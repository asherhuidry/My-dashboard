"""Factory helpers for common claim types.

These are thin constructors that enforce consistent naming conventions for
subjects, predicates, and objects.  They do not add business logic or
hard-code thresholds — they just make it easy to create well-formed claims
without having to remember the (subject, predicate, object) format.

Naming conventions
------------------
Subjects:
    ``"source:<source_id>"``        e.g. ``"source:FRED_GDP"``
    ``"feature:<feature_name>"``    e.g. ``"feature:rsi_14"``
    ``"asset:<ticker>"``            e.g. ``"asset:AAPL"``
    ``"model:<model_type>"``        e.g. ``"model:mlp"``

Predicates:
    ``"is_useful_for"``             a source/feature adds value
    ``"improves_accuracy_on"``      a feature measurably helps
    ``"is_correlated_with"``        two assets move together
    ``"is_causal_for"``             a proposed causal relationship
    ``"meets_performance_bar"``     a model passes a defined threshold
    (or any other plain-English verb phrase)

Objects:
    ``"domain:<domain>"``           e.g. ``"domain:macro_regime"``
    ``"model:<type>_on_<symbol>"``  e.g. ``"model:mlp_on_AAPL"``
    ``"dataset:<version>"``         e.g. ``"dataset:v1.2.3"``
    ``"asset:<ticker>"``            e.g. ``"asset:SPY"``
    (or any free-form description)

Usage::

    from ml.evidence.templates import (
        source_usefulness_claim,
        feature_usefulness_claim,
        relationship_claim,
    )

    c1 = source_usefulness_claim(
        source_id   = "FRED_GDP",
        useful_for  = "macro_regime",
        notes       = "GDP growth correlates with broad equity direction",
        confidence  = 0.55,
    )

    c2 = feature_usefulness_claim(
        feature_name     = "rsi_14",
        model_type       = "mlp",
        dataset_version  = "abc123",
        improvement_note = "accuracy +0.03 on AAPL 3-fold walk-forward",
        confidence       = 0.6,
    )

    c3 = relationship_claim(
        asset_a    = "AAPL",
        asset_b    = "QQQ",
        criterion  = "rolling_60d_correlation > 0.85",
        confidence = 0.7,
    )
"""
from __future__ import annotations

from ml.evidence.schema import Claim, ClaimType


def source_usefulness_claim(
    source_id:         str,
    useful_for:        str,
    notes:             str        = "",
    confidence:        float      = 0.5,
    uncertainty_notes: str        = "",
    counterpoints:     list[str]  | None = None,
    tags:              list[str]  | None = None,
) -> Claim:
    """Create a claim that a data source is useful for a given purpose.

    Produces a claim with the triple::

        ("source:<source_id>", "is_useful_for", "domain:<useful_for>")

    Args:
        source_id:         Identifier of the data source (e.g. ``"FRED_GDP"``).
        useful_for:        Domain or purpose (e.g. ``"macro_regime"``).
        notes:             Free-text context or rationale.
        confidence:        Human-set confidence in [0.0, 1.0].
        uncertainty_notes: Known caveats or unknowns.
        counterpoints:     Known counterarguments.
        tags:              String labels for filtering (e.g. ``["macro"]``).

    Returns:
        A new Claim in PROPOSED status.

    Example::

        c = source_usefulness_claim(
            source_id  = "FRED_GDP",
            useful_for = "macro_regime",
            confidence = 0.55,
        )
        # c.subject == "source:FRED_GDP"
        # c.predicate == "is_useful_for"
        # c.object == "domain:macro_regime"
    """
    return Claim.new(
        claim_type        = ClaimType.SOURCE_USEFULNESS,
        subject           = f"source:{source_id}",
        predicate         = "is_useful_for",
        obj               = f"domain:{useful_for}",
        confidence        = confidence,
        uncertainty_notes = uncertainty_notes,
        counterpoints     = counterpoints,
        tags              = tags,
        notes             = notes,
    )


def feature_usefulness_claim(
    feature_name:      str,
    model_type:        str,
    dataset_version:   str,
    improvement_note:  str        = "",
    confidence:        float      = 0.5,
    uncertainty_notes: str        = "",
    counterpoints:     list[str]  | None = None,
    tags:              list[str]  | None = None,
) -> Claim:
    """Create a claim that a feature improves a model on a given dataset.

    Produces a claim with the triple::

        ("feature:<feature_name>",
         "improves_accuracy_on",
         "model:<model_type>_dataset:<dataset_version>")

    Args:
        feature_name:     Feature column name (e.g. ``"rsi_14"``).
        model_type:       Model family (e.g. ``"mlp"``).
        dataset_version:  Dataset version hash or label.
        improvement_note: Free-text description of the observed improvement.
        confidence:       Human-set confidence in [0.0, 1.0].
        uncertainty_notes: Known caveats.
        counterpoints:    Known counterarguments.
        tags:             String labels.

    Returns:
        A new Claim in PROPOSED status.

    Example::

        c = feature_usefulness_claim(
            feature_name    = "rsi_14",
            model_type      = "mlp",
            dataset_version = "abc123",
            improvement_note = "accuracy +0.03 over 3-fold walk-forward",
            confidence      = 0.6,
        )
        # c.subject    == "feature:rsi_14"
        # c.predicate  == "improves_accuracy_on"
        # c.object     == "model:mlp_dataset:abc123"
    """
    return Claim.new(
        claim_type        = ClaimType.FEATURE_USEFULNESS,
        subject           = f"feature:{feature_name}",
        predicate         = "improves_accuracy_on",
        obj               = f"model:{model_type}_dataset:{dataset_version}",
        confidence        = confidence,
        uncertainty_notes = uncertainty_notes,
        counterpoints     = counterpoints,
        tags              = tags,
        notes             = improvement_note,
    )


def relationship_claim(
    asset_a:           str,
    asset_b:           str,
    criterion:         str,
    predicate:         str        = "is_correlated_with",
    notes:             str        = "",
    confidence:        float      = 0.5,
    uncertainty_notes: str        = "",
    counterpoints:     list[str]  | None = None,
    tags:              list[str]  | None = None,
) -> Claim:
    """Create a claim that two assets are meaningfully related under a criterion.

    Produces a claim with the triple::

        ("asset:<asset_a>", "<predicate>", "asset:<asset_b>_criterion:<criterion>")

    Args:
        asset_a:          First asset ticker or identifier.
        asset_b:          Second asset ticker or identifier.
        criterion:        Description of the relationship criterion
                          (e.g. ``"rolling_60d_correlation > 0.85"``).
        predicate:        Relationship verb (default ``"is_correlated_with"``).
        notes:            Free-text context.
        confidence:       Human-set confidence in [0.0, 1.0].
        uncertainty_notes: Known caveats.
        counterpoints:    Known counterarguments.
        tags:             String labels.

    Returns:
        A new Claim in PROPOSED status.

    Example::

        c = relationship_claim(
            asset_a   = "AAPL",
            asset_b   = "QQQ",
            criterion = "rolling_60d_correlation > 0.85",
            confidence = 0.7,
        )
        # c.subject   == "asset:AAPL"
        # c.predicate == "is_correlated_with"
        # c.object    == "asset:QQQ_criterion:rolling_60d_correlation > 0.85"
    """
    return Claim.new(
        claim_type        = ClaimType.RELATIONSHIP,
        subject           = f"asset:{asset_a}",
        predicate         = predicate,
        obj               = f"asset:{asset_b}_criterion:{criterion}",
        confidence        = confidence,
        uncertainty_notes = uncertainty_notes,
        counterpoints     = counterpoints,
        tags              = tags,
        notes             = notes,
    )


def performance_claim(
    model_type:        str,
    symbol:            str,
    dataset_version:   str,
    metric:            str,
    threshold:         float,
    observed:          float,
    notes:             str        = "",
    confidence:        float      = 0.5,
    uncertainty_notes: str        = "",
    counterpoints:     list[str]  | None = None,
    tags:              list[str]  | None = None,
) -> Claim:
    """Create a claim that a model meets a performance threshold on a dataset.

    Produces a claim with the triple::

        ("model:<model_type>_on_<symbol>",
         "meets_performance_bar",
         "metric:<metric>_threshold:<threshold>_dataset:<dataset_version>")

    Args:
        model_type:       Model family (e.g. ``"mlp"``).
        symbol:           Asset the model was evaluated on.
        dataset_version:  Dataset version hash or label.
        metric:           Metric name (e.g. ``"mean_accuracy"``).
        threshold:        The bar the model must clear.
        observed:         The observed metric value.
        notes:            Free-text context.
        confidence:       Human-set confidence in [0.0, 1.0].
        uncertainty_notes: Known caveats.
        counterpoints:    Known counterarguments.
        tags:             String labels.

    Returns:
        A new Claim in PROPOSED status.
    """
    auto_note = (
        f"Observed {metric}={observed:.4f} vs threshold {threshold:.4f} "
        f"on {symbol} (dataset {dataset_version})."
    )
    if notes:
        auto_note = auto_note + " " + notes
    return Claim.new(
        claim_type        = ClaimType.PERFORMANCE,
        subject           = f"model:{model_type}_on_{symbol}",
        predicate         = "meets_performance_bar",
        obj               = (
            f"metric:{metric}_threshold:{threshold:.4f}"
            f"_dataset:{dataset_version}"
        ),
        confidence        = confidence,
        uncertainty_notes = uncertainty_notes,
        counterpoints     = counterpoints,
        tags              = tags,
        notes             = auto_note,
    )
