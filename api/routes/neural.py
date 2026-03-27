"""Neural intelligence API — FinBrainNet graph predictions + pattern discovery.

Endpoints:
  GET  /api/neural/predict/{symbol}      — graph-aware prediction (uses full universe context)
  GET  /api/neural/explain/{symbol}      — attention explanation (which neighbors drove it)
  GET  /api/neural/anomalies             — current anomaly signals across universe
  GET  /api/neural/regime                — current market regime from VAE clustering
  GET  /api/neural/hidden-correlations   — newly discovered correlations
  GET  /api/neural/latent-factors        — discovered latent factors in embedding space
  POST /api/neural/train                 — trigger offline training (admin)
  GET  /api/neural/status                — model status (trained, checkpoint age, etc.)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

log    = logging.getLogger(__name__)
router = APIRouter()

CHECKPOINT_DIR = Path("checkpoints/neural")

# ── Lazy model cache ──────────────────────────────────────────────────────────
# Load once and cache; reload if checkpoint is newer than cache
_model_cache: dict[str, Any] = {}
_graph_cache: dict[str, Any] = {}


def _get_model_and_graph(device: str = "cpu"):
    """Load FinBrainNet and FinancialGraph (cached in memory)."""
    ckpt_path = CHECKPOINT_DIR / "finbrainnet_best.pt"
    if not ckpt_path.exists():
        return None, None

    ckpt_mtime = ckpt_path.stat().st_mtime
    if _model_cache.get("mtime") != ckpt_mtime:
        try:
            from ml.neural.unified_model import load_model
            model, ckpt = load_model(str(ckpt_path), device)
            _model_cache["model"]  = model
            _model_cache["ckpt"]   = ckpt
            _model_cache["mtime"]  = ckpt_mtime
        except Exception as exc:
            log.warning("Failed to load FinBrainNet: %s", exc)
            return None, None

    if "graph" not in _graph_cache:
        try:
            from ml.neural.graph_net import FinancialGraph
            _graph_cache["graph"] = FinancialGraph.build_from_knowledge_base(device=device)
        except Exception as exc:
            log.warning("Failed to build graph: %s", exc)
            return _model_cache.get("model"), None

    return _model_cache.get("model"), _graph_cache.get("graph")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/neural/predict/{symbol}")
async def graph_predict(symbol: str) -> dict[str, Any]:
    """Predict using the full graph-aware model.

    Unlike /api/predict/{symbol} which uses only the asset's own data,
    this endpoint propagates information from supply chain, correlations,
    and macro indicators before predicting.

    Returns:
        direction_prob, return_estimate, confidence, and which neighbors
        most influenced the prediction.
    """
    symbol = symbol.upper()
    model, graph = _get_model_and_graph()

    if model is None:
        # Fallback to rule-based if neural model not trained
        from api.routes.predict import get_prediction
        return await get_prediction(symbol)

    try:
        from ml.neural.train_unified import (
            _build_price_tensor, _build_fundamental_tensor, assemble_node_inputs
        )
        from ml.neural.unified_model import NodeInputs

        # Build inputs just for this symbol + its full graph context
        # (Full graph context comes from the cached node_inputs)
        price_seq = _build_price_tensor(symbol, seq_len=63, n_features=80)
        fund_vec  = _build_fundamental_tensor(symbol)

        if price_seq is None:
            raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

        target_input = NodeInputs(
            node_id      = symbol,
            node_type    = "asset",
            price_seq    = price_seq,
            fundamentals = fund_vec,
        )

        model.eval()
        import torch
        with torch.no_grad():
            result = model([target_input], graph, target_ids=[symbol])

        if not result.node_ids:
            raise HTTPException(status_code=404, detail=f"{symbol} not in graph")

        prob  = float(result.direction_probs[0])
        ret   = float(result.return_estimates[0])
        conf  = float(result.confidence[0])

        # Get attention explanation
        explanation = model.get_attention_explanation(symbol, graph, top_k=10)

        return {
            "symbol":          symbol,
            "direction_prob":  round(prob, 4),
            "direction":       "bullish" if prob > 0.55 else "bearish" if prob < 0.45 else "neutral",
            "return_estimate": round(ret, 2),
            "confidence":      round(conf, 3),
            "model_type":      "finbrainnet_graph",
            "top_influences":  explanation[:5],
            "timestamp":       datetime.now(tz=timezone.utc).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.warning("Graph prediction failed for %s: %s", symbol, exc)
        from api.routes.predict import get_prediction
        return await get_prediction(symbol)


@router.get("/neural/explain/{symbol}")
async def explain_prediction(symbol: str) -> dict[str, Any]:
    """Get full attention explanation for a symbol's prediction.

    Shows which graph neighbors — supply chain partners, macro indicators,
    sector peers — most influenced the model's prediction.

    Returns:
        Ranked list of influences with edge types and weights.
    """
    symbol = symbol.upper()
    model, graph = _get_model_and_graph()

    if model is None or graph is None:
        return {"symbol": symbol, "error": "Model not trained", "influences": []}

    explanation = model.get_attention_explanation(symbol, graph, top_k=15)

    # Enrich with node types
    type_descriptions = {
        "supplies_to":     "Supply chain partner",
        "correlates_with": "Statistically correlated",
        "impacts":         "Macro driver",
        "belongs_to":      "Sector membership",
        "competes_with":   "Competitor",
        "leads":           "Leading indicator",
    }

    enriched = []
    for inf in explanation:
        enriched.append({
            **inf,
            "edge_description": type_descriptions.get(inf["edge_type"], "Related"),
            "node_in_graph":    inf["neighbor_id"] in graph.node_to_idx,
        })

    return {
        "symbol":      symbol,
        "influences":  enriched,
        "graph_context": {
            "total_neighbors": len(graph.get_neighbors(symbol, max_hops=1).get("hop_1", [])),
            "two_hop_neighbors": len(graph.get_neighbors(symbol, max_hops=2).get("hop_2", [])),
        }
    }


@router.get("/neural/anomalies")
async def get_anomalies(min_score: float = Query(2.0)) -> dict[str, Any]:
    """Get current anomaly signals across the universe.

    Anomaly = an asset or macro indicator behaving unusually relative to:
      (1) its own historical patterns
      (2) what its graph neighbors would predict

    High anomaly score = something unusual is happening. Watch these closely.

    Args:
        min_score: Minimum Z-score to include (default 2.0 = 2 std above normal).
    """
    vae_path = CHECKPOINT_DIR / "vae_discovery.pt"
    if not vae_path.exists():
        return {"error": "Pattern discovery model not trained", "anomalies": []}

    try:
        model, graph = _get_model_and_graph()
        if model is None:
            return {"anomalies": [], "error": "Model not available"}

        from ml.neural.train_unified import assemble_node_inputs
        from ml.neural.pattern_discovery import PatternDiscoveryEngine
        import torch

        node_inputs = assemble_node_inputs(graph, verbose=False)
        model.eval()
        with torch.no_grad():
            result = model(node_inputs, graph, device="cpu")

        # Load pattern engine
        from ml.neural.unified_model import MODEL_CONFIG_DEFAULT
        engine = PatternDiscoveryEngine(
            embed_dim = MODEL_CONFIG_DEFAULT["embed_dim"],
            latent_dim = 32,
        )
        engine.vae.load_state_dict(
            torch.load(vae_path, map_location="cpu", weights_only=True)
        )
        engine._trained = True

        discovery = engine.discover(
            node_ids   = result.node_ids,
            embeddings = result.node_embeddings,
            graph      = graph,
        )
        anomalies = [a for a in discovery["anomalies"] if a["score"] >= min_score]

        return {
            "anomalies": anomalies,
            "count":     len(anomalies),
            "regime_index": discovery["regime"]["index"],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
    except Exception as exc:
        log.warning("Anomaly detection failed: %s", exc)
        return {"anomalies": [], "error": str(exc)}


@router.get("/neural/hidden-correlations")
async def get_hidden_correlations(
    min_similarity: float = Query(0.88),
    limit:          int   = Query(30),
) -> dict[str, Any]:
    """Discover hidden correlations from the embedding space.

    These are asset pairs that the model implicitly learned are related
    through co-movement — even though they're not in the supply chain
    or known correlation database.

    These are the truly novel discoveries: relationships that no analyst
    has explicitly programmed, emerging purely from pattern learning.

    Returns:
        List of hidden correlations sorted by similarity (strongest first).
    """
    vae_path = CHECKPOINT_DIR / "vae_discovery.pt"
    if not vae_path.exists():
        return {"error": "Pattern discovery model not trained", "correlations": []}

    try:
        model, graph = _get_model_and_graph()
        if model is None:
            return {"correlations": [], "error": "Model not available"}

        from ml.neural.train_unified import assemble_node_inputs
        from ml.neural.pattern_discovery import discover_hidden_correlations
        import torch

        node_inputs = assemble_node_inputs(graph, verbose=False)
        model.eval()
        with torch.no_grad():
            result = model(node_inputs, graph, device="cpu")

        corrs = discover_hidden_correlations(
            node_ids    = result.node_ids,
            embeddings  = result.node_embeddings,
            graph       = graph,
            min_sim     = min_similarity,
            max_results = limit,
        )

        return {
            "correlations": [
                {
                    "node_a":        c.node_a,
                    "node_b":        c.node_b,
                    "similarity":    c.similarity,
                    "known_in_graph":c.known_in_graph,
                    "type":          c.suggested_type,
                    "strength":      c.strength,
                }
                for c in corrs
            ],
            "new_discoveries": sum(1 for c in corrs if not c.known_in_graph),
            "total":           len(corrs),
        }
    except Exception as exc:
        log.warning("Hidden correlation discovery failed: %s", exc)
        return {"correlations": [], "error": str(exc)}


@router.get("/neural/latent-factors")
async def get_latent_factors() -> dict[str, Any]:
    """Get discovered latent factors driving market co-movement.

    Unlike traditional quant factors (value, momentum, quality — which are
    hand-crafted), these factors are discovered purely from the learned
    embedding space. Factor 1 might be "AI capex cycle", Factor 2 might
    be "rate sensitivity", Factor 3 "energy supercycle" — whatever
    the model actually learned from the data.

    Returns:
        Top factors with explained variance and top-loading nodes.
    """
    try:
        model, graph = _get_model_and_graph()
        if model is None:
            return {"error": "Model not available", "factors": []}

        from ml.neural.train_unified import assemble_node_inputs
        from ml.neural.pattern_discovery import extract_latent_factors
        import torch

        node_inputs = assemble_node_inputs(graph, verbose=False)
        model.eval()
        with torch.no_grad():
            result = model(node_inputs, graph, device="cpu")

        factor_result = extract_latent_factors(
            result.node_embeddings, result.node_ids, n_factors=10
        )

        return factor_result
    except Exception as exc:
        log.warning("Factor extraction failed: %s", exc)
        return {"factors": [], "error": str(exc)}


@router.get("/neural/status")
async def get_model_status() -> dict[str, Any]:
    """Get FinBrainNet training status and checkpoint info."""
    ckpt_path = CHECKPOINT_DIR / "finbrainnet_best.pt"
    vae_path  = CHECKPOINT_DIR / "vae_discovery.pt"

    status = {
        "model_trained":     ckpt_path.exists(),
        "vae_trained":       vae_path.exists(),
        "checkpoint_path":   str(ckpt_path),
        "checkpoint_exists": ckpt_path.exists(),
    }

    if ckpt_path.exists():
        import torch
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            meta = ckpt.get("metadata", {})
            status.update({
                "best_val_acc":    meta.get("val_acc"),
                "trained_epoch":   meta.get("epoch"),
                "model_version":   ckpt.get("version", "unknown"),
                "checkpoint_age_hours": round(
                    (datetime.now().timestamp() - ckpt_path.stat().st_mtime) / 3600, 1
                ),
            })
        except Exception:
            pass

    model, graph = _get_model_and_graph()
    if graph is not None:
        status["graph_nodes"] = graph.n_nodes
        status["graph_edges"] = len(graph._edges)

    return status


@router.post("/neural/train")
async def trigger_training(
    background_tasks: BackgroundTasks,
    mode:   str = "daily",
    epochs: int = 30,
) -> dict[str, Any]:
    """Trigger FinBrainNet training in the background.

    Args:
        mode:   'offline' for full training, 'daily' for online update.
        epochs: Training epochs (only used for offline mode).

    Returns:
        Immediate response confirming the training job was started.
    """
    def _train():
        try:
            from ml.neural.train_unified import run as neural_run
            result = neural_run(mode=mode, epochs=epochs)
            log.info("Training complete: %s", result)
            # Clear model cache so next request loads new checkpoint
            _model_cache.clear()
        except Exception as exc:
            log.error("Training failed: %s", exc)

    background_tasks.add_task(_train)
    return {
        "status":  "started",
        "mode":    mode,
        "epochs":  epochs if mode == "offline" else "N/A",
        "message": "Training running in background. Check /api/neural/status for updates.",
    }
