"""Bridge between empirical pipeline outputs and the evidence layer.

Converts WalkForwardResult, FoldResult, WalkForwardComparisonResult,
ComparisonResult, and PipelineResult into structured EvidenceItems and
performance Claims.  All generated objects are *proposed* — nothing is
auto-promoted.

This module is deliberately thin.  It translates data from existing result
objects; it does not re-run computations or call external services.

Key functions
-------------
evidence_from_fold_result(fr, symbol)      → EvidenceItem
evidence_from_wf_result(wf)               → list[EvidenceItem]
evidence_from_pipeline_result(result)     → EvidenceItem
claims_from_comparison(result)            → list[ClaimBundle]
claims_from_wf_comparison(result)         → list[ClaimBundle]
save_evidence_bundle(store, items, bundles) → SaveResult

``ClaimBundle`` is a ``(Claim, list[evidence_ids])`` named tuple — it keeps
the claim and the evidence IDs that should be linked to it together.

Usage::

    from ml.evidence.bridge import (
        evidence_from_wf_result,
        claims_from_wf_comparison,
        save_evidence_bundle,
    )
    from ml.evidence import ClaimStore

    store = ClaimStore()

    # After a walk-forward comparison run:
    items  = evidence_from_wf_result(results["baseline"])
    items += evidence_from_wf_result(results["mlp"])
    bundles = claims_from_wf_comparison(wf_comparison_result)
    saved = save_evidence_bundle(store, items, bundles)
    print(saved.claim_ids, saved.evidence_ids)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from ml.evidence.schema import (
    Claim,
    ClaimStatus,
    ClaimType,
    EvidenceItem,
    EvidenceSourceType,
)

if TYPE_CHECKING:
    from ml.evidence.store import ClaimStore
    from ml.validation.wf_runner import FoldResult, WalkForwardResult


# ── Result types ───────────────────────────────────────────────────────────────

class ClaimBundle(NamedTuple):
    """A Claim paired with the evidence IDs that should be linked to it."""
    claim:        Claim
    evidence_ids: list[str]


@dataclass
class SaveResult:
    """Return value of save_evidence_bundle.

    Attributes:
        evidence_ids: IDs of all EvidenceItems that were persisted.
        claim_ids:    IDs of all Claims that were persisted.
        skipped_evidence: IDs of items that were skipped (already existed).
        skipped_claims:   IDs of claims that were skipped (already existed).
    """
    evidence_ids:      list[str]
    claim_ids:         list[str]
    skipped_evidence:  list[str]
    skipped_claims:    list[str]


# ── EvidenceItem factories ─────────────────────────────────────────────────────

def evidence_from_fold_result(
    fr:     "FoldResult",
    symbol: str,
) -> EvidenceItem:
    """Convert a single walk-forward FoldResult into an EvidenceItem.

    The evidence captures: model type, symbol, fold index, key metrics
    (accuracy, auc, f1), backtest figures (hit_rate, sharpe,
    cumulative_return, benchmark_return), and the per-fold promotion
    recommendation.

    Args:
        fr:     A FoldResult from ``run_walk_forward_model``.
        symbol: Ticker symbol the fold was run on.

    Returns:
        An EvidenceItem with ``source_type=EXPERIMENT`` (when an
        experiment_id is present) or ``BACKTEST`` (fold-only run).
    """
    m  = fr.metrics
    bt = fr.backtest_summary

    acc     = m.get("accuracy", 0.0)
    hit     = bt.get("hit_rate", 0.0)
    cum_ret = bt.get("cumulative_return", 0.0)
    bm_ret  = bt.get("benchmark_return", 0.0)
    sharpe  = bt.get("sharpe", 0.0)
    beat_bm = cum_ret > bm_ret

    summary = (
        f"Fold {fr.fold_idx}: {fr.model_type} on {symbol.upper()} — "
        f"acc={acc:.3f}  hit={hit:.3f}  "
        f"ret={cum_ret:+.1%} ({'beat' if beat_bm else 'missed'} bm {bm_ret:+.1%})  "
        f"sharpe={sharpe:.2f}  "
        f"promo={'yes' if fr.promotion_recommended else 'no'}"
    )

    src_type = (
        EvidenceSourceType.EXPERIMENT
        if fr.experiment_id
        else EvidenceSourceType.BACKTEST
    )
    src_ref = fr.experiment_id or f"wf_fold_{fr.fold_idx}_{fr.model_type}_{symbol.upper()}"

    structured: dict[str, Any] = {
        "symbol":        symbol.upper(),
        "model_type":    fr.model_type,
        "fold_idx":      fr.fold_idx,
        "n_train":       fr.fold_spec.n_train,
        "n_val":         fr.fold_spec.n_val,
        "n_test":        fr.fold_spec.n_test,
        "test_start":    fr.fold_spec.test_date_start,
        "test_end":      fr.fold_spec.test_date_end,
        "accuracy":      round(acc, 4),
        "auc":           round(m.get("auc", 0.0), 4),
        "f1":            round(m.get("f1", 0.0), 4),
        "hit_rate":      round(hit, 4),
        "sharpe":        round(sharpe, 4),
        "cumulative_return":  round(cum_ret, 4),
        "benchmark_return":   round(bm_ret, 4),
        "beat_benchmark":     beat_bm,
        "promotion_recommended": fr.promotion_recommended,
    }

    return EvidenceItem.new(
        source_type     = src_type,
        source_ref      = src_ref,
        summary         = summary,
        structured_data = structured,
    )


def evidence_from_wf_result(
    wf: "WalkForwardResult",
) -> list[EvidenceItem]:
    """Convert a WalkForwardResult into a list of EvidenceItems.

    Returns one EvidenceItem per completed fold, plus one aggregate summary
    item.  The aggregate captures mean/std statistics across all folds.

    Args:
        wf: A WalkForwardResult from ``run_walk_forward_model``.

    Returns:
        List of EvidenceItems in fold order, with the aggregate appended last.
        Returns an empty list if no folds completed.
    """
    if not wf.fold_results:
        return []

    items: list[EvidenceItem] = []

    # Per-fold items
    for fr in wf.fold_results:
        items.append(evidence_from_fold_result(fr, wf.symbol))

    # Aggregate summary
    import numpy as np
    accs    = [fr.metrics.get("accuracy", 0.0) for fr in wf.fold_results]
    hits    = [fr.backtest_summary.get("hit_rate", 0.0) for fr in wf.fold_results]
    sharpes = [fr.backtest_summary.get("sharpe", 0.0) for fr in wf.fold_results]
    rets    = [fr.backtest_summary.get("cumulative_return", 0.0) for fr in wf.fold_results]
    bm_rets = [fr.backtest_summary.get("benchmark_return", 0.0) for fr in wf.fold_results]
    n_beat  = sum(1 for r, b in zip(rets, bm_rets) if r > b)
    n_promo = sum(1 for fr in wf.fold_results if fr.promotion_recommended)

    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs, ddof=0))
    mean_hit = float(np.mean(hits))

    agg_summary = (
        f"Walk-forward aggregate: {wf.model_type} on {wf.symbol.upper()} — "
        f"{wf.n_folds_run}/{wf.n_folds_total} folds completed  "
        f"mean_acc={mean_acc:.3f}+/-{std_acc:.3f}  "
        f"mean_hit={mean_hit:.3f}  "
        f"beat_bm={n_beat}/{wf.n_folds_run}  "
        f"fold_promo={n_promo}/{wf.n_folds_run}"
    )
    agg_structured: dict[str, Any] = {
        "symbol":          wf.symbol.upper(),
        "model_type":      wf.model_type,
        "n_folds_total":   wf.n_folds_total,
        "n_folds_run":     wf.n_folds_run,
        "mean_accuracy":   round(mean_acc, 4),
        "std_accuracy":    round(std_acc, 4),
        "mean_hit_rate":   round(mean_hit, 4),
        "mean_sharpe":     round(float(np.mean(sharpes)), 4),
        "mean_cumulative_return": round(float(np.mean(rets)), 4),
        "n_folds_beat_benchmark": n_beat,
        "n_folds_promo_recommended": n_promo,
        "wf_config":       wf.config.to_dict(),
        "generated_at":    wf.generated_at,
    }

    items.append(
        EvidenceItem.new(
            source_type     = EvidenceSourceType.BACKTEST,
            source_ref      = f"wf_{wf.symbol.upper()}_{wf.model_type}_{wf.generated_at[:10]}",
            summary         = agg_summary,
            structured_data = agg_structured,
        )
    )

    return items


def evidence_from_pipeline_result(result: Any) -> EvidenceItem:
    """Convert a single-split PipelineResult into an EvidenceItem.

    Args:
        result: A ``PipelineResult`` from ``run_pipeline`` (train_mlp.py,
                train_lstm.py, or baseline.py).  Must have ``experiment_id``,
                ``symbol``, ``metrics``, ``backtest_summary``, and
                ``promotion_recommended`` attributes.

    Returns:
        An EvidenceItem with ``source_type=EXPERIMENT``.
    """
    m  = result.metrics
    bt = result.backtest_summary

    acc     = m.get("accuracy", 0.0)
    hit     = bt.get("hit_rate", 0.0)
    cum_ret = bt.get("cumulative_return", 0.0)
    bm_ret  = bt.get("benchmark_return", 0.0)
    sharpe  = bt.get("sharpe", 0.0)
    beat_bm = cum_ret > bm_ret

    # Infer model_type from experiment_id prefix or attributes
    model_type = getattr(result, "model_type", "unknown")

    summary = (
        f"Single-split {model_type} on {result.symbol.upper()} — "
        f"acc={acc:.3f}  hit={hit:.3f}  "
        f"ret={cum_ret:+.1%} ({'beat' if beat_bm else 'missed'} bm {bm_ret:+.1%})  "
        f"sharpe={sharpe:.2f}  "
        f"promo={'yes' if result.promotion_recommended else 'no'}"
    )

    structured: dict[str, Any] = {
        "symbol":               result.symbol.upper(),
        "model_type":           model_type,
        "experiment_id":        result.experiment_id,
        "accuracy":             round(acc, 4),
        "auc":                  round(m.get("auc", 0.0), 4),
        "f1":                   round(m.get("f1", 0.0), 4),
        "hit_rate":             round(hit, 4),
        "sharpe":               round(sharpe, 4),
        "cumulative_return":    round(cum_ret, 4),
        "benchmark_return":     round(bm_ret, 4),
        "beat_benchmark":       beat_bm,
        "promotion_recommended": result.promotion_recommended,
    }

    return EvidenceItem.new(
        source_type     = EvidenceSourceType.EXPERIMENT,
        source_ref      = result.experiment_id,
        summary         = summary,
        structured_data = structured,
    )


# ── Claim factories ────────────────────────────────────────────────────────────

def claims_from_comparison(
    result: Any,
) -> list[ClaimBundle]:
    """Generate performance Claims from a single-split ComparisonResult.

    Produces:
    - One PERFORMANCE claim per model (subject=model, predicate=achieves_score,
      object=dataset_version).
    - One PERFORMANCE claim for the comparison winner (predicate=leads_comparison).

    All claims are PROPOSED with conservative confidence.  Claims for
    models that pass their promotion thresholds get slightly higher
    confidence and a note encouraging human review.

    Args:
        result: A ``ComparisonResult`` from ``run_comparison``.

    Returns:
        List of ``ClaimBundle(claim, evidence_ids)`` where evidence_ids is
        empty — callers link evidence after calling
        ``evidence_from_pipeline_result`` on the individual model results.
    """
    bundles: list[ClaimBundle] = []

    for r in result.results:
        score   = getattr(r, "composite_score", 0.0)
        promo   = r.promotion_recommended
        conf    = 0.65 if promo else 0.45
        reasons = getattr(r, "promotion_reasons", [])

        uncertainty = (
            "Single test split; walk-forward validation not performed."
        )
        counterpoints = [
            "Single-split results may not generalise to other time windows."
        ]
        if not promo:
            counterpoints.append(
                f"Did not meet promotion thresholds: {'; '.join(reasons[-2:])}"
                if reasons else "Did not meet promotion thresholds."
            )

        claim = Claim.new(
            claim_type        = ClaimType.PERFORMANCE,
            subject           = f"model:{r.model_type}_on_{result.symbol.upper()}",
            predicate         = "achieves_composite_score",
            obj               = (
                f"score:{score:.4f}_dataset:{r.dataset_version[:12]}"
            ),
            confidence        = conf,
            uncertainty_notes = uncertainty,
            counterpoints     = counterpoints,
            tags              = [
                "single_split", result.symbol.upper(), r.model_type,
                "promoted" if promo else "not_promoted",
            ],
            notes             = (
                f"Composite score {score:.4f}, rank {r.rank}. "
                f"{'Promotion recommended.' if promo else 'Promotion not recommended.'}"
            ),
        )
        # Link to experiment evidence by ref
        bundles.append(ClaimBundle(claim=claim, evidence_ids=[]))

    # Winner claim
    if result.winner and len(result.results) > 1:
        runners_up = [
            r.model_type for r in result.results
            if r.model_type != result.winner
        ]
        winner_claim = Claim.new(
            claim_type        = ClaimType.PERFORMANCE,
            subject           = f"model:{result.winner}_on_{result.symbol.upper()}",
            predicate         = "leads_single_split_comparison",
            obj               = (
                f"vs:{'+'.join(runners_up)}_dataset:{result.dataset_version[:12]}"
            ),
            confidence        = 0.55,
            uncertainty_notes = (
                "Based on a single comparison run; rankings are not stable "
                "across different time windows."
            ),
            counterpoints     = [
                "Composite score differences may be within noise for short test windows."
            ],
            tags              = [
                "comparison_winner", "single_split",
                result.symbol.upper(), result.winner,
            ],
            notes             = (
                f"{result.winner} ranked #1 in single-split comparison "
                f"on {result.symbol.upper()} ({result.generated_at[:10]})."
            ),
        )
        bundles.append(ClaimBundle(claim=winner_claim, evidence_ids=[]))

    return bundles


def claims_from_wf_comparison(
    result: Any,
) -> list[ClaimBundle]:
    """Generate performance Claims from a WalkForwardComparisonResult.

    Produces:
    - One PERFORMANCE claim per model with fold aggregate statistics.
    - One PERFORMANCE claim for the walk-forward winner.

    Confidence is derived from the evaluation recommendation for each model's
    FoldAggregate:

    - strong support (mean_acc >= 0.55 AND beat_bm on majority) → 0.70
    - moderate (mean_acc >= 0.52) → 0.58
    - weak → 0.42

    All claims remain PROPOSED.

    Args:
        result: A ``WalkForwardComparisonResult`` from
                ``run_comparison(walk_forward=True)``.

    Returns:
        List of ``ClaimBundle(claim, evidence_ids)`` where evidence_ids
        is initially empty (populated by the caller or by
        ``save_evidence_bundle`` when items and bundles are passed together).
    """
    from ml.validation.wf_aggregation import wf_promotion_recommend

    bundles: list[ClaimBundle] = []
    cfg     = result.wf_config
    n_folds = cfg.n_folds

    for model_type, agg in result.aggregates.items():
        promo_rec = result.promotions.get(model_type)
        overall_ok = promo_rec.overall_recommended if promo_rec else False

        # Derive confidence from aggregate statistics
        majority_beat = agg.n_folds_beat_benchmark >= (n_folds + 1) // 2
        if overall_ok:
            conf = 0.70
        elif agg.mean_accuracy >= 0.52 and majority_beat:
            conf = 0.58
        else:
            conf = 0.42

        uncertainty = (
            f"Based on {agg.n_folds} walk-forward folds "
            f"(std_accuracy={agg.std_accuracy:.3f}). "
            "Past fold performance does not guarantee future results."
        )
        counterpoints: list[str] = []
        if agg.std_accuracy > 0.06:
            counterpoints.append(
                f"High accuracy variance across folds (std={agg.std_accuracy:.3f}) "
                "suggests instability."
            )
        if not majority_beat:
            counterpoints.append(
                f"Beat benchmark on only {agg.n_folds_beat_benchmark}/{agg.n_folds} "
                "folds — no consistent edge over buy-and-hold."
            )
        if promo_rec and not overall_ok:
            failed = [
                c["criterion"] for c in promo_rec.criteria
                if c.get("gate") and not c.get("passed")
            ]
            if failed:
                counterpoints.append(
                    f"Failed promotion gate(s): {', '.join(failed)}."
                )

        claim = Claim.new(
            claim_type = ClaimType.PERFORMANCE,
            subject    = f"model:{model_type}_on_{result.symbol.upper()}",
            predicate  = "achieves_wf_mean_accuracy",
            obj        = (
                f"acc:{agg.mean_accuracy:.4f}_folds:{agg.n_folds}"
                f"_window:{cfg.window}"
            ),
            confidence        = conf,
            uncertainty_notes = uncertainty,
            counterpoints     = counterpoints,
            tags              = [
                "walk_forward",
                result.symbol.upper(),
                model_type,
                "wf_promoted" if overall_ok else "wf_not_promoted",
            ],
            notes = (
                f"Walk-forward: mean_acc={agg.mean_accuracy:.3f}+/-{agg.std_accuracy:.3f}  "
                f"mean_hit={agg.mean_hit_rate:.3f}  "
                f"mean_sharpe={agg.mean_sharpe:.2f}  "
                f"beat_bm={agg.n_folds_beat_benchmark}/{agg.n_folds}  "
                f"fold_promo={agg.n_folds_promo_recommended}/{agg.n_folds}  "
                f"composite={agg.mean_composite_score:.4f}+/-{agg.std_composite_score:.4f}"
            ),
        )
        bundles.append(ClaimBundle(claim=claim, evidence_ids=[]))

    # Winner claim
    if result.winner and len(result.aggregates) > 1:
        others = [m for m in result.aggregates if m != result.winner]
        w_agg  = result.aggregates[result.winner]
        winner_claim = Claim.new(
            claim_type = ClaimType.PERFORMANCE,
            subject    = f"model:{result.winner}_on_{result.symbol.upper()}",
            predicate  = "leads_walk_forward_comparison",
            obj        = (
                f"vs:{'+'.join(sorted(others))}"
                f"_{result.symbol.upper()}"
                f"_{result.generated_at[:10]}"
            ),
            confidence        = 0.60,
            uncertainty_notes = (
                "Walk-forward ranking is based on mean composite score across "
                f"{cfg.n_folds} folds; may not be stable with different data windows."
            ),
            counterpoints     = [
                "Composite score differences between models may be small relative "
                "to fold-to-fold variance."
            ],
            tags = [
                "wf_comparison_winner", "walk_forward",
                result.symbol.upper(), result.winner,
            ],
            notes = (
                f"{result.winner} achieved highest mean composite score in "
                f"walk-forward comparison on {result.symbol.upper()} "
                f"({result.generated_at[:10]}, {cfg.n_folds} folds)."
            ),
        )
        bundles.append(ClaimBundle(claim=winner_claim, evidence_ids=[]))

    return bundles


# ── Store integration ─────────────────────────────────────────────────────────

def save_evidence_bundle(
    store:   "ClaimStore",
    items:   list[EvidenceItem],
    bundles: list[ClaimBundle],
) -> SaveResult:
    """Persist EvidenceItems and Claims into a ClaimStore.

    Evidence items are added first.  Then each claim in ``bundles`` is added,
    and its ``evidence_ids`` (if any) are linked.  Items or claims that
    already exist in the store (duplicate IDs) are skipped with a warning
    rather than raising.

    After adding all items and claims, an attempt is made to link each
    bundle's evidence_ids to its claim.  Missing evidence IDs are logged
    and skipped.

    Args:
        store:   A ``ClaimStore`` instance.
        items:   EvidenceItems to persist.
        bundles: ``ClaimBundle`` pairs to persist and link.

    Returns:
        A ``SaveResult`` with lists of persisted and skipped IDs.
    """
    import logging
    log = logging.getLogger(__name__)

    saved_ev:   list[str] = []
    skipped_ev: list[str] = []

    for item in items:
        try:
            store.add_evidence(item)
            saved_ev.append(item.evidence_id)
        except ValueError:
            skipped_ev.append(item.evidence_id)
            log.debug("EvidenceItem %s already exists — skipped.", item.evidence_id)

    saved_cl:   list[str] = []
    skipped_cl: list[str] = []

    for bundle in bundles:
        claim = bundle.claim
        try:
            store.add_claim(claim)
            saved_cl.append(claim.claim_id)
        except ValueError:
            skipped_cl.append(claim.claim_id)
            log.debug("Claim %s already exists — skipped.", claim.claim_id)
            continue

        for eid in bundle.evidence_ids:
            if eid in saved_ev or eid in skipped_ev:
                try:
                    store.link_evidence(claim.claim_id, eid)
                except (KeyError, Exception) as exc:
                    log.warning(
                        "Could not link evidence %s to claim %s: %s",
                        eid, claim.claim_id, exc,
                    )

    return SaveResult(
        evidence_ids     = saved_ev,
        claim_ids        = saved_cl,
        skipped_evidence = skipped_ev,
        skipped_claims   = skipped_cl,
    )


def bridge_wf_comparison(
    result: Any,
    store:  "ClaimStore",
    wf_results: "dict[str, WalkForwardResult] | None" = None,
) -> SaveResult:
    """One-call helper: convert a WalkForwardComparisonResult + optional per-model
    WalkForwardResults into evidence + claims and save everything to the store.

    If ``wf_results`` is provided (the dict returned by ``run_walk_forward``),
    per-fold EvidenceItems are generated and linked to the per-model
    performance claims.

    Args:
        result:     A ``WalkForwardComparisonResult``.
        store:      A ``ClaimStore`` instance to persist into.
        wf_results: Optional dict[model_type, WalkForwardResult] from
                    ``run_walk_forward``.  When provided, generates per-fold
                    EvidenceItems and links them to their model's claim.

    Returns:
        A ``SaveResult`` with the IDs of all persisted items and claims.
    """
    all_items: list[EvidenceItem] = []

    # Per-model evidence from fold results
    model_evidence_ids: dict[str, list[str]] = {}
    if wf_results:
        for model_type, wf in wf_results.items():
            ev_list = evidence_from_wf_result(wf)
            all_items.extend(ev_list)
            model_evidence_ids[model_type] = [e.evidence_id for e in ev_list]

    # Generate claims; populate evidence_ids from the per-model items
    raw_bundles = claims_from_wf_comparison(result)
    bundles: list[ClaimBundle] = []
    for bundle in raw_bundles:
        claim = bundle.claim
        # Extract model_type from subject (format: "model:{type}_on_{symbol}")
        subject_parts = claim.subject.split("_on_")
        model_type = subject_parts[0].replace("model:", "") if subject_parts else ""
        ev_ids = model_evidence_ids.get(model_type, [])
        bundles.append(ClaimBundle(claim=claim, evidence_ids=ev_ids))

    return save_evidence_bundle(store, all_items, bundles)


def bridge_comparison(
    result:          Any,
    store:           "ClaimStore",
    pipeline_results: "list[Any] | None" = None,
) -> SaveResult:
    """One-call helper: convert a ComparisonResult into evidence + claims and save.

    If ``pipeline_results`` is provided (list of PipelineResult objects),
    per-model EvidenceItems are generated and linked to each model's claim.

    Args:
        result:           A ``ComparisonResult`` from ``run_comparison``.
        store:            A ``ClaimStore`` instance.
        pipeline_results: Optional list of ``PipelineResult`` objects.  When
                          provided, each is converted to an EvidenceItem and
                          linked to its corresponding model claim.

    Returns:
        A ``SaveResult`` with the IDs of all persisted items and claims.
    """
    all_items: list[EvidenceItem] = []
    model_evidence_ids: dict[str, list[str]] = {}

    if pipeline_results:
        for pr in pipeline_results:
            ev = evidence_from_pipeline_result(pr)
            all_items.append(ev)
            mt = getattr(pr, "model_type", "unknown")
            model_evidence_ids.setdefault(mt, []).append(ev.evidence_id)

    raw_bundles = claims_from_comparison(result)
    bundles: list[ClaimBundle] = []
    for bundle in raw_bundles:
        claim = bundle.claim
        subject_parts = claim.subject.split("_on_")
        model_type = subject_parts[0].replace("model:", "") if subject_parts else ""
        ev_ids = model_evidence_ids.get(model_type, [])
        bundles.append(ClaimBundle(claim=claim, evidence_ids=ev_ids))

    return save_evidence_bundle(store, all_items, bundles)
