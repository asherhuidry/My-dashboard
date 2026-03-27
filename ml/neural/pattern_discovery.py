"""Pattern Discovery Engine — finds hidden structure in the embedding space.

After the GNN encodes the full financial universe into node embeddings,
this module mines those embeddings for:

1. MARKET REGIMES — unsupervised clustering of the embedding space over time.
   Instead of hard-coded regimes (bull/bear), the VAE learns the actual
   structure: "2022 rate-shock regime", "covid crash regime", "AI-bubble regime".

2. NEW CORRELATIONS — pairs of assets that are close in embedding space
   but NOT in the known knowledge graph. These are hidden relationships
   the model discovered purely from co-movement patterns.

3. ANOMALY DETECTION — nodes whose current embedding is far from their
   historical average. A high anomaly score = "this asset is behaving
   unusually relative to its own history AND its graph neighbors."

4. CONTAGION PATHS — when one node goes anomalous, trace which neighbors
   are most likely to follow using attention weights + graph distance.

5. FACTOR DISCOVERY — PCA/ICA on the embedding space to find latent
   factors that explain co-movement. These are the "unknown unknowns"
   of quantitative finance.

Architecture:
  - Variational Autoencoder (VAE) for latent representation
  - Gaussian Mixture Model (GMM) in latent space for regime clustering
  - Mahalanobis distance for anomaly scoring
  - Cosine similarity matrix for hidden correlation discovery
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ── Variational Autoencoder ───────────────────────────────────────────────────

class FinBrainVAE(nn.Module):
    """Variational Autoencoder over node embeddings.

    Learns a compressed latent representation of the financial system state.
    The latent space is what we cluster to find regimes and anomalies.

    Input: (N, embed_dim) — all node embeddings at a point in time.
           Flattened to a single "system state" vector for encoding.

    We use a hierarchical design:
      1. Per-node encoder: compress each node's embedding to latent_dim/2
      2. System encoder: aggregate across all nodes to global state
      3. Decoder: reconstruct from global + local latent codes

    For pattern discovery, we mostly use the LOCAL per-node latent codes.
    The GLOBAL system code captures macro regime (is the whole system in
    risk-on or risk-off mode?).

    Args:
        input_dim: Node embedding dimension.
        latent_dim: Latent code dimension.
        hidden_dim: Encoder/decoder hidden dimension.
    """

    def __init__(
        self,
        input_dim:  int   = 128,
        latent_dim: int   = 32,
        hidden_dim: int   = 128,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Per-node encoder: embed_dim → hidden → (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.mu_head     = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder: latent → embed_dim (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode embeddings to latent distribution parameters.

        Args:
            x: (N, input_dim)

        Returns:
            (mu, logvar) each of shape (N, latent_dim).
        """
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from N(mu, exp(logvar)) using reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # use mean at inference

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent code back to embedding space.

        Args:
            z: (N, latent_dim)

        Returns:
            (N, input_dim) reconstructed embeddings.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: (N, input_dim)

        Returns:
            (x_recon, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decode(z)
        return x_recon, mu, logvar

    @staticmethod
    def loss(x: torch.Tensor, x_recon: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> dict[str, torch.Tensor]:
        """Beta-VAE loss = reconstruction + beta * KL divergence.

        Args:
            x:       Original embeddings.
            x_recon: Reconstructed embeddings.
            mu:      Latent means.
            logvar:  Latent log-variances.
            beta:    KL weight (higher = more disentangled latent space).

        Returns:
            Dict with 'total', 'recon', 'kl' losses.
        """
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total      = recon_loss + beta * kl_loss
        return {"total": total, "recon": recon_loss, "kl": kl_loss}


# ── Regime clustering ─────────────────────────────────────────────────────────

@dataclass
class RegimeCluster:
    """A discovered market regime cluster.

    Attributes:
        regime_id:     Cluster index (0, 1, 2, ...).
        label:         Human-readable label (e.g. 'risk_off_credit_stress').
        centroid:      Mean latent code for this regime.
        typical_dates: Example dates this regime occurred.
        signature:     Top node anomalies characterizing this regime.
        transition_probs: Probability of transitioning to other regimes.
    """
    regime_id:         int
    label:             str
    centroid:          np.ndarray
    typical_dates:     list[str] = field(default_factory=list)
    signature:         dict[str, float] = field(default_factory=dict)
    transition_probs:  dict[int, float] = field(default_factory=dict)


class RegimeDetector:
    """Online GMM-based regime detector operating in VAE latent space.

    Uses a Gaussian Mixture Model to cluster the latent codes over a
    rolling window. New market states are assigned to the nearest cluster;
    if a state is too far from all clusters, it's classified as a new regime.

    The key insight: the VAE latent space is structured so that
    "nearby" states = "similar financial conditions", even if the
    raw asset prices look completely different.

    Args:
        n_components:  Number of regime clusters (K).
        latent_dim:    Dimension of VAE latent space.
        window_size:   Number of historical states to maintain.
    """

    def __init__(
        self,
        n_components: int = 8,
        latent_dim:   int = 32,
        window_size:  int = 756,  # ~3 years
    ) -> None:
        self.n_components = n_components
        self.latent_dim   = latent_dim
        self.window_size  = window_size

        # Cluster centroids (initialized with first observations)
        self.centroids: np.ndarray | None = None
        self.history: list[np.ndarray] = []  # latent codes over time
        self.history_dates: list[str]  = []
        self.current_regime: int       = 0

    def update(self, latent_code: np.ndarray, date: str) -> int:
        """Update regime detector with new latent code.

        Args:
            latent_code: (latent_dim,) current system latent code.
            date:        Date string for this observation.

        Returns:
            Current regime index (0 to n_components-1).
        """
        self.history.append(latent_code)
        self.history_dates.append(date)

        # Keep rolling window
        if len(self.history) > self.window_size:
            self.history.pop(0)
            self.history_dates.pop(0)

        # Fit/update clusters once we have enough history
        if len(self.history) >= self.n_components * 10:
            self._fit_clusters()

        if self.centroids is not None:
            self.current_regime = self._assign_regime(latent_code)

        return self.current_regime

    def _fit_clusters(self) -> None:
        """Fit Gaussian Mixture Model (via K-means init + EM) to history."""
        try:
            from sklearn.mixture import GaussianMixture
            X = np.array(self.history)
            gm = GaussianMixture(
                n_components     = self.n_components,
                covariance_type  = "diag",
                max_iter         = 100,
                random_state     = 42,
                warm_start       = False,
            )
            gm.fit(X)
            self.centroids = gm.means_
            self._gm = gm
        except ImportError:
            # Fallback: simple K-means centroids
            self.centroids = self._simple_kmeans(np.array(self.history))

    def _simple_kmeans(self, X: np.ndarray, n_iter: int = 50) -> np.ndarray:
        """Simple K-means for environments without sklearn."""
        K   = self.n_components
        idx = np.random.choice(len(X), K, replace=False)
        centroids = X[idx].copy()
        for _ in range(n_iter):
            dists    = np.linalg.norm(X[:, None] - centroids[None], axis=-1)  # (N, K)
            labels   = dists.argmin(axis=1)
            for k in range(K):
                mask = labels == k
                if mask.sum() > 0:
                    centroids[k] = X[mask].mean(axis=0)
        return centroids

    def _assign_regime(self, code: np.ndarray) -> int:
        """Assign a latent code to its nearest regime cluster."""
        if hasattr(self, "_gm"):
            return int(self._gm.predict(code.reshape(1, -1))[0])
        dists = np.linalg.norm(self.centroids - code[None], axis=-1)
        return int(dists.argmin())

    def get_regime_probability(self, code: np.ndarray) -> np.ndarray:
        """Get soft assignment probabilities for all regimes.

        Args:
            code: (latent_dim,) latent code.

        Returns:
            (n_components,) probability vector.
        """
        if hasattr(self, "_gm"):
            return self._gm.predict_proba(code.reshape(1, -1))[0]
        # Softmax over negative distances
        if self.centroids is None:
            return np.ones(self.n_components) / self.n_components
        dists = np.linalg.norm(self.centroids - code[None], axis=-1)
        probs = np.exp(-dists / (dists.std() + 1e-9))
        return probs / probs.sum()


# ── Anomaly detection ─────────────────────────────────────────────────────────

@dataclass
class AnomalySignal:
    """Anomaly detected for a specific node at a specific time.

    Attributes:
        node_id:        The asset or indicator with anomalous behavior.
        anomaly_score:  Z-score of reconstruction error (higher = more anomalous).
        regime_shift:   Whether this coincides with a regime transition.
        likely_causes:  Top graph neighbors contributing to the anomaly.
        timestamp:      When this anomaly was detected.
    """
    node_id:       str
    anomaly_score: float
    regime_shift:  bool
    likely_causes: list[str]
    timestamp:     str


class AnomalyDetector:
    """Detects unusual behavior in node embeddings using reconstruction error.

    A node is anomalous if its current VAE reconstruction error is significantly
    higher than its rolling historical average — meaning "the model can't explain
    this node's current behavior given the rest of the graph."

    High reconstruction error = the node is doing something unusual relative
    to its historical patterns AND its connected neighbors.

    This catches:
      - Earnings surprises before they're priced in
      - Sector rotation starts (one sector decouples from usual correlates)
      - Contagion starting (a stress metric spikes unusually)
      - Macro regime breaks (yield curve inverts unusually fast)

    Args:
        window_size: Rolling window for baseline error stats (trading days).
        z_threshold: Z-score threshold for anomaly classification.
    """

    def __init__(
        self,
        window_size:  int   = 252,
        z_threshold:  float = 2.5,
    ) -> None:
        self.window   = window_size
        self.threshold = z_threshold
        self.history: dict[str, list[float]] = {}  # node_id → recon_errors

    def update(
        self,
        node_id:     str,
        recon_error: float,
    ) -> float | None:
        """Update history and return Z-score if sufficient history available.

        Args:
            node_id:     Node identifier.
            recon_error: Current reconstruction error for this node.

        Returns:
            Z-score (positive = above average error = anomalous) or None
            if insufficient history (<30 observations).
        """
        if node_id not in self.history:
            self.history[node_id] = []
        self.history[node_id].append(recon_error)
        if len(self.history[node_id]) > self.window:
            self.history[node_id].pop(0)
        if len(self.history[node_id]) < 30:
            return None
        errors = np.array(self.history[node_id])
        mu, sigma = errors.mean(), errors.std()
        if sigma < 1e-9:
            return 0.0
        return float((recon_error - mu) / sigma)

    def score_all(
        self,
        node_ids:     list[str],
        embeddings:   torch.Tensor,
        vae:          FinBrainVAE,
        graph:        Any | None = None,
    ) -> list[AnomalySignal]:
        """Score all nodes for anomalies.

        Args:
            node_ids:   List of node identifiers matching embeddings rows.
            embeddings: (N, embed_dim) current node embeddings.
            vae:        Trained VAE for reconstruction.
            graph:      FinancialGraph for finding likely causes (optional).

        Returns:
            List of AnomalySignal for nodes exceeding threshold.
        """
        vae.eval()
        with torch.no_grad():
            x_recon, mu, logvar = vae(embeddings)
            # Per-node reconstruction error = MSE
            errors = ((embeddings - x_recon) ** 2).mean(dim=-1).cpu().numpy()

        anomalies = []
        now = datetime.now(tz=timezone.utc).isoformat()

        for i, (nid, err) in enumerate(zip(node_ids, errors)):
            z = self.update(nid, float(err))
            if z is not None and z > self.threshold:
                # Find likely causes: top graph neighbors
                causes = []
                if graph is not None:
                    neighbors = graph.get_neighbors(nid, max_hops=1)
                    causes = neighbors.get("hop_1", [])[:5]

                anomalies.append(AnomalySignal(
                    node_id       = nid,
                    anomaly_score = round(z, 2),
                    regime_shift  = z > self.threshold * 1.5,
                    likely_causes = causes,
                    timestamp     = now,
                ))

        # Sort by anomaly score descending
        anomalies.sort(key=lambda a: a.anomaly_score, reverse=True)
        return anomalies


# ── Hidden correlation discovery ──────────────────────────────────────────────

@dataclass
class HiddenCorrelation:
    """A relationship discovered in embedding space, not in the known graph.

    Attributes:
        node_a:       First node identifier.
        node_b:       Second node identifier.
        similarity:   Cosine similarity in embedding space (0-1).
        known_in_graph: Whether this edge already exists in the knowledge graph.
        suggested_type: Probable relationship type based on node types.
        strength:     'very_strong' | 'strong' | 'moderate'.
    """
    node_a:          str
    node_b:          str
    similarity:      float
    known_in_graph:  bool
    suggested_type:  str
    strength:        str


def discover_hidden_correlations(
    node_ids:     list[str],
    embeddings:   torch.Tensor,
    graph:        Any,
    min_sim:      float = 0.85,
    max_results:  int   = 100,
) -> list[HiddenCorrelation]:
    """Find pairs close in embedding space that aren't in the known graph.

    These are the "unknown unknowns" — relationships the model implicitly
    learned from co-movement patterns but that aren't in our hard-coded
    supply chain / correlation knowledge base.

    Args:
        node_ids:    List of node identifiers (matches embeddings rows).
        embeddings:  (N, embed_dim) node embeddings.
        graph:       FinancialGraph for checking known edges.
        min_sim:     Minimum cosine similarity to consider a relationship.
        max_results: Maximum number of hidden correlations to return.

    Returns:
        Sorted list of HiddenCorrelation (strongest first).
    """
    N = len(node_ids)
    # Normalize embeddings for cosine similarity
    emb_norm = F.normalize(embeddings, dim=-1).cpu()

    # Compute pairwise cosine similarity matrix
    sim_matrix = torch.mm(emb_norm, emb_norm.t()).numpy()  # (N, N)

    # Build set of known edges for O(1) lookup
    known_edges: set[tuple[str, str]] = set()
    for src, dst, _etype, _w in graph._edges:
        src_id = node_ids[src] if src < N else None
        dst_id = node_ids[dst] if dst < N else None
        if src_id and dst_id:
            known_edges.add((src_id, dst_id))
            known_edges.add((dst_id, src_id))

    results: list[HiddenCorrelation] = []

    for i in range(N):
        for j in range(i + 1, N):  # upper triangle only
            sim = float(sim_matrix[i, j])
            if sim < min_sim:
                continue

            id_a, id_b = node_ids[i], node_ids[j]
            if id_a is None or id_b is None:
                continue

            is_known = (id_a, id_b) in known_edges

            # Infer relationship type from node types
            type_a = graph.node_types.get(graph.node_to_idx.get(id_a, -1), "asset")
            type_b = graph.node_types.get(graph.node_to_idx.get(id_b, -1), "asset")

            if type_a == "macro" or type_b == "macro":
                suggested = "macro_impact"
            elif type_a == "sector" or type_b == "sector":
                suggested = "sector_membership"
            else:
                suggested = "hidden_correlation"

            strength = (
                "very_strong" if sim > 0.95
                else "strong"  if sim > 0.90
                else "moderate"
            )

            results.append(HiddenCorrelation(
                node_a         = id_a,
                node_b         = id_b,
                similarity     = round(sim, 4),
                known_in_graph = is_known,
                suggested_type = suggested,
                strength       = strength,
            ))

    # Sort: unknown + strongest first (the real discoveries)
    results.sort(key=lambda r: (not r.known_in_graph, r.similarity), reverse=True)
    return results[:max_results]


# ── Latent factor analysis ─────────────────────────────────────────────────────

def extract_latent_factors(
    embeddings:  torch.Tensor,
    node_ids:    list[str],
    n_factors:   int = 10,
) -> dict[str, Any]:
    """Extract dominant factors from the embedding space using PCA.

    In traditional quant finance, factors are hard-coded: value, momentum,
    quality, size. Here we let the data speak — the top PCA components
    of the learned embeddings ARE the latent factors.

    Factor 1 might be "rate sensitivity": loads heavily on TLT, T10Y2Y, XLU
    Factor 2 might be "AI capex": loads heavily on NVDA, MSFT, AMZN, GOOGL
    Factor 3 might be "energy supercycle": XLE, COP, SLB, DCOILWTICO

    These emerge from the model, not from human pre-specification.

    Args:
        embeddings: (N, embed_dim) node embeddings.
        node_ids:   Node identifiers.
        n_factors:  Number of top factors to extract.

    Returns:
        Dict with factor loadings, explained variance, and top-loading nodes.
    """
    X = embeddings.detach().cpu().numpy()  # (N, D)

    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_factors, min(X.shape) - 1))
        pca.fit(X)
        components   = pca.components_                    # (n_factors, D)
        explained    = pca.explained_variance_ratio_      # (n_factors,)
        projections  = pca.transform(X)                   # (N, n_factors)
    except ImportError:
        # Manual PCA via SVD if sklearn unavailable
        X_centered = X - X.mean(axis=0)
        U, S, Vt   = np.linalg.svd(X_centered, full_matrices=False)
        k          = min(n_factors, len(S))
        components  = Vt[:k]
        total_var   = (S ** 2).sum()
        explained   = (S[:k] ** 2) / total_var
        projections = U[:, :k] * S[:k]

    # Find top nodes loading on each factor
    factors = []
    for f_idx in range(len(explained)):
        loadings = projections[:, f_idx]
        # Top positive and negative loaders
        top_pos_idx = np.argsort(loadings)[-5:][::-1]
        top_neg_idx = np.argsort(loadings)[:5]

        factors.append({
            "factor_id":       f_idx,
            "explained_var":   float(explained[f_idx]),
            "top_positive":    [node_ids[i] for i in top_pos_idx if node_ids[i]],
            "top_negative":    [node_ids[i] for i in top_neg_idx if node_ids[i]],
            "factor_loadings": {node_ids[i]: float(loadings[i])
                                for i in range(len(node_ids)) if node_ids[i]},
        })

    return {
        "n_factors":        len(factors),
        "total_explained":  float(explained.sum()),
        "factors":          factors,
    }


# ── Master pattern discovery pipeline ─────────────────────────────────────────

class PatternDiscoveryEngine:
    """Orchestrates all unsupervised pattern discovery.

    Run after each GNN forward pass with the enriched node embeddings.
    Continuously updates the knowledge base with newly discovered patterns.

    Usage:
        engine = PatternDiscoveryEngine(embed_dim=128, latent_dim=32)
        # Train the VAE on historical embeddings first
        engine.train_vae(historical_embeddings_list, epochs=50)
        # Then run discovery on each new timestep
        results = engine.discover(node_ids, current_embeddings, graph, date)
    """

    def __init__(
        self,
        embed_dim:  int   = 128,
        latent_dim: int   = 32,
        n_regimes:  int   = 8,
        device:     str   = "cpu",
    ) -> None:
        self.vae        = FinBrainVAE(input_dim=embed_dim, latent_dim=latent_dim)
        self.regimes    = RegimeDetector(n_components=n_regimes, latent_dim=latent_dim)
        self.anomalies  = AnomalyDetector(window_size=252, z_threshold=2.5)
        self.device     = device
        self.vae        = self.vae.to(device)
        self._trained   = False

    def train_vae(
        self,
        embedding_snapshots: list[torch.Tensor],  # list of (N, embed_dim) tensors
        epochs: int = 50,
        lr:     float = 1e-3,
        beta:   float = 0.5,
    ) -> list[float]:
        """Train the VAE on historical embedding snapshots.

        Args:
            embedding_snapshots: One tensor per historical timestep.
            epochs:  Training epochs.
            lr:      Learning rate.
            beta:    VAE beta (KL weight).

        Returns:
            Training loss history.
        """
        self.vae.train()
        opt    = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=1e-5)
        sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        losses = []

        # Flatten all snapshots into a single dataset
        all_embs = torch.cat(embedding_snapshots, dim=0).to(self.device)

        for epoch in range(epochs):
            # Random mini-batch
            perm  = torch.randperm(len(all_embs))[:256]
            batch = all_embs[perm]

            x_recon, mu, logvar = self.vae(batch)
            loss_dict = FinBrainVAE.loss(batch, x_recon, mu, logvar, beta=beta)
            total_loss = loss_dict["total"]

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(total_loss))

            if (epoch + 1) % 10 == 0:
                log.debug(
                    "VAE epoch %d/%d: loss=%.4f (recon=%.4f, kl=%.4f)",
                    epoch + 1, epochs,
                    float(total_loss),
                    float(loss_dict["recon"]),
                    float(loss_dict["kl"]),
                )

        self.vae.eval()
        self._trained = True
        return losses

    def discover(
        self,
        node_ids:    list[str],
        embeddings:  torch.Tensor,
        graph:       Any,
        date:        str | None = None,
    ) -> dict[str, Any]:
        """Run full pattern discovery pipeline on current embeddings.

        Args:
            node_ids:   Node identifiers matching embeddings rows.
            embeddings: (N, embed_dim) current GNN node embeddings.
            graph:      FinancialGraph for context.
            date:       Current date string for regime history.

        Returns:
            Dict with: regime, anomalies, hidden_correlations, latent_factors.
        """
        if not self._trained:
            log.warning("VAE not trained yet — running with untrained model")

        date = date or datetime.now(tz=timezone.utc).date().isoformat()
        embs = embeddings.detach().to(self.device)

        # 1. Get latent codes
        self.vae.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encode(embs)
            z          = mu.cpu().numpy()  # use mean (deterministic at inference)

        # 2. Regime detection: use mean of all node codes as system state
        system_code  = z.mean(axis=0)
        regime_idx   = self.regimes.update(system_code, date)
        regime_probs = self.regimes.get_regime_probability(system_code)

        # 3. Anomaly detection
        with torch.no_grad():
            x_recon, _, _ = self.vae(embs)
        errors = ((embs.cpu() - x_recon.cpu()) ** 2).mean(dim=-1).numpy()

        anomaly_signals = []
        for i, (nid, err) in enumerate(zip(node_ids, errors)):
            if nid is None:
                continue
            z_score = self.anomalies.update(nid, float(err))
            if z_score is not None and z_score > self.anomalies.threshold:
                causes = graph.get_neighbors(nid, max_hops=1).get("hop_1", [])[:5]
                anomaly_signals.append(AnomalySignal(
                    node_id       = nid,
                    anomaly_score = round(z_score, 2),
                    regime_shift  = z_score > self.anomalies.threshold * 1.5,
                    likely_causes = causes,
                    timestamp     = datetime.now(tz=timezone.utc).isoformat(),
                ))

        anomaly_signals.sort(key=lambda a: a.anomaly_score, reverse=True)

        # 4. Hidden correlation discovery (expensive — run every N calls)
        hidden_corrs = discover_hidden_correlations(
            node_ids   = node_ids,
            embeddings = embs.cpu(),
            graph      = graph,
            min_sim    = 0.88,
            max_results= 50,
        )
        # Filter to truly hidden (not in known graph)
        new_correlations = [c for c in hidden_corrs if not c.known_in_graph]

        # 5. Latent factor analysis
        factors = extract_latent_factors(mu, node_ids, n_factors=8)

        return {
            "date":           date,
            "regime": {
                "index":       regime_idx,
                "probs":       regime_probs.tolist(),
                "system_code": system_code.tolist()[:8],  # first 8 dims for display
            },
            "anomalies":     [
                {"node_id": a.node_id, "score": a.anomaly_score,
                 "causes": a.likely_causes, "regime_shift": a.regime_shift}
                for a in anomaly_signals[:20]
            ],
            "new_correlations": [
                {"node_a": c.node_a, "node_b": c.node_b,
                 "similarity": c.similarity, "type": c.suggested_type}
                for c in new_correlations[:20]
            ],
            "latent_factors": factors["factors"][:5],
        }
