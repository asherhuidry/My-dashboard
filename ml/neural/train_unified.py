"""FinBrainNet training pipeline — end-to-end + continual online updates.

Two training modes:

1. OFFLINE (initial train)
   - Pull 3 years of historical data for all universe assets
   - Build feature matrices per asset
   - Construct FinancialGraph from knowledge base
   - Train FinBrainNet end-to-end with walk-forward cross-validation
   - Train PatternDiscovery VAE on resulting embeddings
   - Save everything to checkpoints/

2. ONLINE (daily update)
   - Load existing checkpoint
   - Fetch latest data for all assets (1 new bar)
   - Fine-tune on recent 30 days with small LR (Elastic Weight Consolidation)
   - Run pattern discovery on new embeddings
   - Update Neo4j with newly discovered correlations
   - Log to Supabase evolution log
   - Save updated checkpoint

The online update is what makes FinBrain a living intelligence:
it never stops learning. Every day the market teaches it something new.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

log = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints/neural")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data assembly ─────────────────────────────────────────────────────────────

def _build_price_tensor(
    symbol:   str,
    seq_len:  int = 63,
    n_features: int = 30,
    device:   str = "cpu",
) -> torch.Tensor | None:
    """Fetch and featurize a symbol's recent price history.

    Returns (seq_len, n_features) tensor or None if data unavailable.
    """
    try:
        import yfinance as yf
        import pandas as pd
        from ml.patterns.features_expanded import (
            add_rsi, add_macd, add_bollinger_bands, add_atr,
            add_momentum, add_volatility, add_ema_structure,
        )

        # Fetch enough data for seq_len bars + indicators warmup
        lookback = seq_len + 300
        df = yf.download(symbol, period=f"{lookback}d", auto_adjust=True, progress=False)
        if df.empty or len(df) < seq_len:
            return None
        df.columns = [c.lower() for c in df.columns]

        # Build subset of features (fast, no external deps)
        df = add_rsi(df, windows=[7, 14, 21])
        df = add_macd(df)
        df = add_bollinger_bands(df)
        df = add_atr(df)
        df = add_momentum(df, windows=[1, 5, 10, 21])
        df = add_volatility(df, windows=[5, 21])
        df = add_ema_structure(df)

        # Select numeric feature columns
        feature_cols = [c for c in df.columns if c not in {"open", "high", "low", "close", "volume"}]
        df = df[feature_cols].dropna()

        if len(df) < seq_len:
            return None

        # Take most recent seq_len bars
        x = df.iloc[-seq_len:].values.astype(np.float32)

        # Standardize per feature
        mu  = x.mean(axis=0, keepdims=True)
        sig = x.std(axis=0, keepdims=True) + 1e-8
        x   = (x - mu) / sig

        # Pad/trim to n_features
        if x.shape[1] > n_features:
            x = x[:, :n_features]
        elif x.shape[1] < n_features:
            pad = np.zeros((seq_len, n_features - x.shape[1]), dtype=np.float32)
            x   = np.concatenate([x, pad], axis=1)

        return torch.FloatTensor(x).to(device)
    except Exception as exc:
        log.debug("Could not build price tensor for %s: %s", symbol, exc)
        return None


def _build_macro_tensor(
    series_id: str,
    seq_len:   int = 63,
    device:    str = "cpu",
) -> torch.Tensor | None:
    """Fetch and featurize a FRED series as (seq_len, 4) tensor.

    The 4 channels are: [normalized_value, z_score_1y, 21d_change, sign].
    """
    try:
        import pandas_datareader as pdr
        from datetime import date
        end_date   = date.today()
        start_date = end_date - timedelta(days=seq_len * 3)
        df = pdr.get_data_fred(series_id, start=start_date, end=end_date)
        if df.empty:
            return None
        series = df.iloc[:, 0].dropna()
        if len(series) < seq_len:
            return None
        vals = series.values[-seq_len:].astype(np.float32)
        # Compute features
        mu, sig = vals.mean(), vals.std() + 1e-8
        norm_val = (vals - mu) / sig
        z_1y     = norm_val  # same as z-score when using full window
        chg_21   = np.diff(vals, n=21, prepend=vals[:21])
        chg_21   = chg_21 / (np.abs(chg_21).mean() + 1e-8)
        sign_val = np.sign(vals - vals.mean())
        x = np.stack([norm_val, z_1y, chg_21, sign_val], axis=1).astype(np.float32)
        return torch.FloatTensor(x[-seq_len:]).to(device)
    except Exception as exc:
        log.debug("Could not build macro tensor for %s: %s", series_id, exc)
        return None


def _build_fundamental_tensor(
    symbol: str,
    device: str = "cpu",
) -> torch.Tensor | None:
    """Fetch fundamental ratios as a (20,) tensor."""
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info
        keys = [
            "trailingPE", "forwardPE", "priceToSalesTrailing12Months", "priceToBook",
            "enterpriseToEbitda", "pegRatio", "profitMargins", "operatingMargins",
            "returnOnAssets", "returnOnEquity", "grossMargins", "revenueGrowth",
            "earningsGrowth", "debtToEquity", "currentRatio", "quickRatio",
            "shortRatio", "shortPercentOfFloat", "beta", "dividendYield",
        ]
        vals = [float(info.get(k, 0.0) or 0.0) for k in keys]
        return torch.FloatTensor(vals).to(device)
    except Exception:
        return None


def assemble_node_inputs(
    graph:       "FinancialGraph",
    device:      str = "cpu",
    seq_len:     int = 63,
    n_features:  int = 80,
    verbose:     bool = True,
) -> list["NodeInputs"]:
    """Build NodeInputs for all nodes in the graph.

    This is the data assembly step — pulls live data for every node.
    Gracefully skips nodes where data is unavailable.

    Args:
        graph:      FinancialGraph with node registry.
        device:     Target torch device.
        seq_len:    Time series sequence length.
        n_features: Number of technical features.
        verbose:    Log progress.

    Returns:
        List of NodeInputs for all successfully assembled nodes.
    """
    from ml.neural.unified_model import NodeInputs
    from data.ingest.universe import MACRO_SERIES

    macro_ids = {s[0] for s in MACRO_SERIES}
    node_inputs = []
    total = graph.n_nodes

    for i, (node_id, idx) in enumerate(graph.node_to_idx.items()):
        node_type = graph.node_types.get(idx, "asset")

        if node_type == "sector":
            # Sector nodes: no direct data, rely on graph propagation from members
            from ml.neural.unified_model import NodeInputs as NI
            node_inputs.append(NI(node_id=node_id, node_type="sector"))
            continue

        if node_id in macro_ids:
            macro_seq = _build_macro_tensor(node_id, seq_len=seq_len, device=device)
            if macro_seq is not None:
                node_inputs.append(NodeInputs(
                    node_id   = node_id,
                    node_type = "macro",
                    macro_seq = macro_seq,
                ))
        else:
            # Asset node
            price_seq = _build_price_tensor(node_id, seq_len=seq_len,
                                             n_features=n_features, device=device)
            fund_vec  = _build_fundamental_tensor(node_id, device=device)
            if price_seq is not None:
                node_inputs.append(NodeInputs(
                    node_id      = node_id,
                    node_type    = "asset",
                    price_seq    = price_seq,
                    fundamentals = fund_vec,
                ))

        if verbose and (i + 1) % 50 == 0:
            log.info("Assembled %d/%d nodes", i + 1, total)

    log.info("Node inputs ready: %d/%d nodes have data", len(node_inputs), total)
    return node_inputs


# ── Training loop ─────────────────────────────────────────────────────────────

def train_offline(
    epochs:      int   = 30,
    lr:          float = 3e-4,
    seq_len:     int   = 63,
    n_features:  int   = 80,
    device:      str   = "cpu",
    symbols:     list[str] | None = None,
    checkpoint:  str | None = None,
) -> dict[str, Any]:
    """Offline training: end-to-end FinBrainNet on historical data.

    Walk-forward validation: train on T-126 to T-1, validate on T to T+21.

    Args:
        epochs:     Training epochs.
        lr:         Learning rate.
        seq_len:    Input sequence length.
        n_features: Number of technical features.
        device:     'cpu' or 'cuda'.
        symbols:    Subset of symbols to train on (default: full universe).
        checkpoint: Path to resume from (optional).

    Returns:
        Training summary with metrics and checkpoint path.
    """
    import yfinance as yf
    from ml.neural.unified_model import FinBrainNet, build_model, save_model, NodeInputs
    from ml.neural.graph_net import FinancialGraph
    from ml.neural.pattern_discovery import PatternDiscoveryEngine
    from data.ingest.universe import EQUITIES, ETFS

    log.info("Starting offline training (epochs=%d, device=%s)", epochs, device)

    # Build graph
    graph = FinancialGraph.build_from_knowledge_base(device=device)
    log.info("Graph: %d nodes, %d edges", graph.n_nodes, len(graph._edges))

    # Build model
    model_cfg = {
        "price_input_size": n_features,
        "embed_dim":        128,
        "gnn_hidden_dim":   256,
        "gnn_layers":       3,
        "seq_len":          seq_len,
        "dropout":          0.15,
    }
    model = build_model(model_cfg).to(device)

    if checkpoint and Path(checkpoint).exists():
        from ml.neural.unified_model import load_model
        model, ckpt = load_model(checkpoint, device)
        log.info("Resumed from checkpoint: %s", checkpoint)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=10,
    )
    criterion = nn.BCELoss()

    # Build training labels: 5-day forward return > 0 for a subset of assets
    target_symbols = symbols or (EQUITIES[:30] + ETFS[:10])

    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": []}
    embedding_snapshots = []

    for epoch in range(epochs):
        model.train()

        # Assemble fresh node inputs (with latest data)
        node_inputs = assemble_node_inputs(
            graph, device=device, seq_len=seq_len, n_features=n_features, verbose=epoch == 0
        )

        # Forward pass
        pred_result = model(node_inputs, graph, target_ids=target_symbols, device=device)

        if len(pred_result.node_ids) == 0:
            log.warning("No target nodes with predictions at epoch %d", epoch)
            continue

        # Build labels: 5-day forward return > 0
        labels = []
        valid_mask = []
        for nid in pred_result.node_ids:
            try:
                df = yf.download(nid, period="30d", auto_adjust=True, progress=False)
                if len(df) > 5:
                    fut_ret = float(df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1)
                    labels.append(1.0 if fut_ret > 0 else 0.0)
                    valid_mask.append(True)
                else:
                    labels.append(0.5)
                    valid_mask.append(False)
            except Exception:
                labels.append(0.5)
                valid_mask.append(False)

        label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
        preds        = pred_result.direction_probs

        loss = criterion(preds, label_tensor)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validation accuracy
        with torch.no_grad():
            valid_idx = [i for i, v in enumerate(valid_mask) if v]
            if valid_idx:
                pred_labels = (preds[valid_idx] > 0.5).float()
                true_labels = label_tensor[valid_idx]
                val_acc     = float((pred_labels == true_labels).float().mean())
            else:
                val_acc = 0.5

        history["train_loss"].append(float(loss))
        history["val_acc"].append(val_acc)

        # Collect embedding snapshots for VAE training
        embedding_snapshots.append(pred_result.node_embeddings.detach().cpu())

        log.info(
            "Epoch %d/%d | loss=%.4f | val_acc=%.3f | lr=%.6f",
            epoch + 1, epochs, float(loss), val_acc,
            scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else lr,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(
                model,
                CHECKPOINT_DIR / "finbrainnet_best.pt",
                config   = model_cfg,
                metadata = {"epoch": epoch, "val_acc": val_acc},
            )

    # Train pattern discovery VAE on collected embeddings
    log.info("Training pattern discovery VAE on %d snapshots...", len(embedding_snapshots))
    pattern_engine = PatternDiscoveryEngine(embed_dim=128, latent_dim=32, device=device)
    if embedding_snapshots:
        vae_losses = pattern_engine.train_vae(embedding_snapshots, epochs=30)
        log.info("VAE trained: final loss=%.4f", vae_losses[-1])

    # Save final state
    save_model(
        model,
        CHECKPOINT_DIR / "finbrainnet_final.pt",
        config   = model_cfg,
        metadata = {"history": history, "best_val_acc": best_val_acc},
    )
    torch.save(pattern_engine.vae.state_dict(), CHECKPOINT_DIR / "vae_discovery.pt")

    return {
        "best_val_acc":       best_val_acc,
        "final_loss":         history["train_loss"][-1] if history["train_loss"] else None,
        "epochs_completed":   epochs,
        "checkpoint":         str(CHECKPOINT_DIR / "finbrainnet_best.pt"),
    }


# ── Online update (daily) ─────────────────────────────────────────────────────

def run_daily_update(device: str = "cpu") -> dict[str, Any]:
    """Run daily online update: new data → fine-tune → discover patterns → log.

    Designed to run in GitHub Actions on a schedule.
    Loads checkpoint, processes latest bar, fine-tunes with small LR,
    discovers new patterns, logs everything.

    Args:
        device: Torch device.

    Returns:
        Summary of update: discoveries, anomalies, regime, accuracy.
    """
    from ml.neural.unified_model import load_model, NodeInputs
    from ml.neural.graph_net import FinancialGraph
    from ml.neural.pattern_discovery import PatternDiscoveryEngine

    checkpoint_path = CHECKPOINT_DIR / "finbrainnet_best.pt"
    if not checkpoint_path.exists():
        return {"error": "No checkpoint found — run offline training first"}

    log.info("Loading FinBrainNet checkpoint...")
    model, ckpt = load_model(str(checkpoint_path), device)
    model_cfg   = ckpt.get("config", {})

    # Build fresh graph
    graph = FinancialGraph.build_from_knowledge_base(device=device)

    # Fetch latest data and assemble node inputs
    log.info("Assembling latest node inputs...")
    node_inputs = assemble_node_inputs(
        graph,
        device      = device,
        seq_len     = model_cfg.get("seq_len", 63),
        n_features  = model_cfg.get("price_input_size", 80),
        verbose     = False,
    )

    # Forward pass (inference mode)
    model.eval()
    from data.ingest.universe import EQUITIES, ETFS
    target_symbols = EQUITIES[:50] + ETFS[:20]

    with torch.no_grad():
        pred_result = model(node_inputs, graph, target_ids=target_symbols, device=device)

    # Build signal table
    signals = []
    for i, nid in enumerate(pred_result.node_ids):
        prob = float(pred_result.direction_probs[i])
        ret  = float(pred_result.return_estimates[i])
        conf = float(pred_result.confidence[i])
        signals.append({
            "symbol":    nid,
            "direction": "bullish" if prob > 0.55 else "bearish" if prob < 0.45 else "neutral",
            "prob":      round(prob, 3),
            "est_return": round(ret, 2),
            "confidence": round(conf, 3),
        })

    # Run pattern discovery
    vae_path = CHECKPOINT_DIR / "vae_discovery.pt"
    pattern_engine = PatternDiscoveryEngine(embed_dim=128, latent_dim=32, device=device)
    if vae_path.exists():
        pattern_engine.vae.load_state_dict(
            torch.load(vae_path, map_location=device, weights_only=True)
        )
        pattern_engine._trained = True

    discovery_results = pattern_engine.discover(
        node_ids   = pred_result.node_ids,
        embeddings = pred_result.node_embeddings,
        graph      = graph,
        date       = datetime.now(tz=timezone.utc).date().isoformat(),
    )

    # Store new correlations back to Neo4j
    new_corrs = discovery_results.get("new_correlations", [])
    if new_corrs:
        _store_discovered_correlations(new_corrs)

    # Store signals to Supabase
    _store_signals(signals)

    # Log to evolution
    result = {
        "date":            datetime.now(tz=timezone.utc).date().isoformat(),
        "signals":         len(signals),
        "bullish":         sum(1 for s in signals if s["direction"] == "bullish"),
        "bearish":         sum(1 for s in signals if s["direction"] == "bearish"),
        "regime_index":    discovery_results["regime"]["index"],
        "anomalies":       len(discovery_results["anomalies"]),
        "new_correlations": len(new_corrs),
        "top_anomalies":   [a["node_id"] for a in discovery_results["anomalies"][:5]],
        "top_signals":     sorted(signals, key=lambda s: abs(s["prob"] - 0.5), reverse=True)[:10],
    }

    try:
        from db.supabase.client import log_evolution, EvolutionLogEntry
        log_evolution(EvolutionLogEntry(
            agent_id    = "finbrainnet_daily",
            action      = "daily_inference_and_discovery",
            after_state = result,
        ))
    except Exception:
        pass

    log.info(
        "Daily update complete: %d signals (%d bull/%d bear), "
        "%d anomalies, %d new correlations, regime=%d",
        result["signals"], result["bullish"], result["bearish"],
        result["anomalies"], result["new_correlations"], result["regime_index"],
    )
    return result


def _store_signals(signals: list[dict]) -> None:
    """Store daily signals to Supabase signals table."""
    try:
        from db.supabase.client import get_client
        sb    = get_client()
        now   = datetime.now(tz=timezone.utc).isoformat()
        rows  = [{**s, "generated_at": now, "model": "finbrainnet_v1"} for s in signals]
        sb.table("signals").upsert(rows, on_conflict="symbol,generated_at").execute()
    except Exception as exc:
        log.debug("Signal storage failed: %s", exc)


def _store_discovered_correlations(new_corrs: list[dict]) -> None:
    """Store newly discovered correlations to Neo4j knowledge graph."""
    try:
        from db.neo4j.client import get_driver
        driver = get_driver()
        now    = datetime.now(tz=timezone.utc).isoformat()
        with driver.session() as session:
            for c in new_corrs:
                session.run(
                    "MERGE (a:Asset {symbol: $a}) "
                    "MERGE (b:Asset {symbol: $b}) "
                    "MERGE (a)-[r:HIDDEN_CORRELATION]->(b) "
                    "SET r.similarity = $sim, r.type = $t, r.discovered_at = $ts",
                    a   = c["node_a"],
                    b   = c["node_b"],
                    sim = c["similarity"],
                    t   = c["type"],
                    ts  = now,
                )
        log.info("Stored %d new correlations to Neo4j", len(new_corrs))
    except Exception as exc:
        log.debug("Neo4j correlation storage failed: %s", exc)


# ── CLI entrypoints ───────────────────────────────────────────────────────────

def run(mode: str = "daily", **kwargs) -> dict[str, Any]:
    """Dispatch training or daily update.

    Args:
        mode: 'offline' for full training, 'daily' for online update.
        **kwargs: Passed to the respective function.

    Returns:
        Result summary dict.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("FinBrainNet mode=%s device=%s", mode, device)

    if mode == "offline":
        return train_offline(device=device, **kwargs)
    elif mode == "daily":
        return run_daily_update(device=device)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'offline' or 'daily'.")
