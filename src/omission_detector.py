"""
omission_detector.py
--------------------
The intellectual heart of the Earnings Call Silence Mapping strategy.

Core insight
------------
This module detects *topic omissions* — the deliberate absence of language
a company has historically used.  The key distinction from sentiment analysis:

    We do not score tone or language.
    We score the structural absence of topics relative to a company's
    own historical baseline.

A CEO who routinely discusses "supply chain optimisation" in every quarterly
call but suddenly drops the phrase entirely is signalling something that a
pure sentiment scorer would miss entirely.  The silence IS the signal.

Omission severity weighting
----------------------------
``omission_severity = (1 − presence_ratio) × avg_tfidf_weight × core_vocab_multiplier``

* ``presence_ratio`` — how much of the term's historical weight is present now.
* ``avg_tfidf_weight`` — baseline importance (terms the company emphasises heavily
  are more informative when omitted).
* ``core_vocab_multiplier`` — 1.5× if the term appeared in *every* trailing
  quarter (core vocabulary).  Omitting a term used in every single prior call
  is far more informative than omitting one used inconsistently.

This severity formulation is fully unit-testable with synthetic inputs: any
combination of (current_vector, standard_vocab_df, feature_names) can be
constructed to verify the detection logic deterministically.
"""

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

# Multiplier applied to core vocabulary omissions
_CORE_VOCAB_MULTIPLIER: float = 1.5


# ---------------------------------------------------------------------------
# 1. detect_omissions
# ---------------------------------------------------------------------------

def detect_omissions(
    current_vector: np.ndarray,
    standard_vocab_df: pd.DataFrame,
    feature_names: list,
) -> pd.DataFrame:
    """Detect topic omissions for a single company-quarter.

    For each term in the standard vocabulary, compare its current TF-IDF
    weight against its historical average.  Flag as an omission event if the
    presence ratio falls below ``OMISSION_THRESHOLD``.

    Parameters
    ----------
    current_vector : np.ndarray
        1-D TF-IDF vector for the company's most recent quarter
        (length == MAX_FEATURES, aligned with ``feature_names``).
    standard_vocab_df : pd.DataFrame
        Standard vocabulary DataFrame from ``build_standard_vocabulary``.
        Required columns: ``term``, ``avg_tfidf_weight``, ``core_vocab``.
    feature_names : list[str]
        Ordered vocabulary list from ``build_tfidf_matrix``.

    Returns
    -------
    pd.DataFrame
        One row per vocabulary term.  Columns:
        ``term``, ``avg_tfidf_weight``, ``current_tfidf``,
        ``presence_ratio``, ``omission_event``, ``core_vocab``,
        ``omission_severity``.

    Notes
    -----
    **Core vocabulary multiplier rationale**: a term present in all four
    trailing quarters has been part of the company's *canonical* communication
    for a full year.  Its sudden disappearance is not random variation — it is
    an editorial decision.  We weight this 1.5× in the severity score to
    reflect its higher informational value.  This is the key modelling
    assumption and should be tested in sensitivity analysis.
    """
    feature_index = {term: idx for idx, term in enumerate(feature_names)}
    rows: list[dict] = []

    for _, vocab_row in standard_vocab_df.iterrows():
        term = vocab_row["term"]
        avg_weight = float(vocab_row["avg_tfidf_weight"])
        is_core = bool(vocab_row.get("core_vocab", False))

        # Look up current TF-IDF weight
        term_idx = feature_index.get(term)
        if term_idx is None:
            current_tfidf = 0.0
        else:
            current_tfidf = float(current_vector[term_idx])

        # Presence ratio: how much of historical weight is still present?
        if avg_weight > 0:
            presence_ratio = current_tfidf / avg_weight
        else:
            presence_ratio = 1.0  # undefined; treat as present

        # Omission event flag
        omission_event = presence_ratio < config.OMISSION_THRESHOLD

        # Severity: weighted by historical importance and core-vocab status
        multiplier = _CORE_VOCAB_MULTIPLIER if is_core else 1.0
        if omission_event:
            severity = (1.0 - presence_ratio) * avg_weight * multiplier
        else:
            severity = 0.0

        rows.append(
            {
                "term": term,
                "avg_tfidf_weight": avg_weight,
                "current_tfidf": current_tfidf,
                "presence_ratio": presence_ratio,
                "omission_event": omission_event,
                "core_vocab": is_core,
                "omission_severity": severity,
            }
        )

    omission_df = pd.DataFrame(rows)
    return omission_df


# ---------------------------------------------------------------------------
# 2. compute_omission_score
# ---------------------------------------------------------------------------

def compute_omission_score(omission_df: pd.DataFrame) -> float:
    """Aggregate omission events into a single normalised score.

    The score is the sum of omission severity values for all omission events,
    normalised by the total number of vocabulary terms.  This normalisation
    allows fair comparison across companies with different vocabulary sizes.

    Parameters
    ----------
    omission_df : pd.DataFrame
        Output of ``detect_omissions``.

    Returns
    -------
    float
        Omission score ∈ [0, ∞).  Higher values indicate more and/or more
        severe topic omissions relative to the company's historical baseline.
        A score of 0 means no omissions were detected.
    """
    n_terms = len(omission_df)
    if n_terms == 0:
        return 0.0

    total_severity = omission_df.loc[
        omission_df["omission_event"], "omission_severity"
    ].sum()

    return float(total_severity / n_terms)


# ---------------------------------------------------------------------------
# 3. run_omission_detection_universe
# ---------------------------------------------------------------------------

def run_omission_detection_universe(
    corpus_df: pd.DataFrame,
    tfidf_matrix,
    feature_names: list,
    vocabularies: dict,
) -> pd.DataFrame:
    """Run omission detection across all company-quarters in the universe.

    Parameters
    ----------
    corpus_df : pd.DataFrame
        Preprocessed corpus.
    tfidf_matrix : scipy sparse matrix
        Full corpus TF-IDF matrix.
    feature_names : list[str]
        Vocabulary list.
    vocabularies : dict
        Nested dict ``{cik: {(quarter, year): vocabulary_df}}``.

    Returns
    -------
    pd.DataFrame
        Master signal DataFrame.  Columns: ``cik``, ``ticker``, ``quarter``,
        ``year``, ``earnings_date``, ``omission_score``, ``n_omission_events``,
        ``n_core_vocab_omissions``, ``top_omitted_terms``.
        Saved to ``data/processed/omission_signals.csv``.
    """
    from src.tfidf_engine import get_company_quarter_vector

    fit_df = corpus_df[corpus_df["coverage_quality"] == "transcript"].copy()
    signal_rows: list[dict] = []

    for cik, quarter_vocabs in vocabularies.items():
        company_rows = fit_df[fit_df["cik"].astype(str) == str(cik)]
        if company_rows.empty:
            continue
        ticker = company_rows["ticker"].iloc[0]

        for (quarter, year), vocab_df in quarter_vocabs.items():
            current_vec = get_company_quarter_vector(
                tfidf_matrix, corpus_df, cik, quarter, year
            )
            if current_vec is None:
                continue

            # Match earnings date from corpus
            date_rows = company_rows[
                (company_rows["quarter"] == quarter)
                & (company_rows["year"].astype(int) == int(year))
            ]
            earnings_date = (
                date_rows["filing_date"].iloc[0] if not date_rows.empty else None
            )

            # Detect omissions
            omission_df = detect_omissions(current_vec, vocab_df, feature_names)
            score = compute_omission_score(omission_df)

            n_omission_events = int(omission_df["omission_event"].sum())
            n_core_omissions = int(
                omission_df[omission_df["omission_event"] & omission_df["core_vocab"]].shape[0]
            )

            # Top 5 omitted terms by severity
            top_omitted = (
                omission_df[omission_df["omission_event"]]
                .sort_values("omission_severity", ascending=False)
                .head(5)["term"]
                .tolist()
            )

            signal_rows.append(
                {
                    "cik": cik,
                    "ticker": ticker,
                    "quarter": quarter,
                    "year": year,
                    "earnings_date": earnings_date,
                    "omission_score": score,
                    "n_omission_events": n_omission_events,
                    "n_core_vocab_omissions": n_core_omissions,
                    "top_omitted_terms": json.dumps(top_omitted),
                }
            )

            logger.debug(
                "%s %s/%d — score: %.4f, events: %d, core: %d",
                ticker,
                quarter,
                year,
                score,
                n_omission_events,
                n_core_omissions,
            )

    signal_df = pd.DataFrame(signal_rows)
    if signal_df.empty:
        logger.warning("No omission signals generated — check corpus and vocabulary coverage")
        return signal_df

    signal_df = signal_df.sort_values(["year", "quarter", "ticker"]).reset_index(drop=True)
    signal_df.to_csv(config.OMISSION_SIGNALS_PATH, index=False)
    logger.info(
        "Omission signals saved: %d rows — %s", len(signal_df), config.OMISSION_SIGNALS_PATH
    )
    return signal_df


# ---------------------------------------------------------------------------
# 4. classify_omission_regime
# ---------------------------------------------------------------------------

def classify_omission_regime(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Assign omission deciles and signal labels cross-sectionally each quarter.

    Decile ranking is computed within each calendar quarter across all companies.
    This cross-sectional normalisation means the signal is always relative to
    peers, not to a fixed absolute threshold.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of ``run_omission_detection_universe``.

    Returns
    -------
    pd.DataFrame
        Enriched with columns:
        ``omission_decile`` (1=lowest omission, 10=highest) and
        ``signal_label`` ("HIGH_OMISSION", "NEUTRAL", or "LOW_OMISSION").

    Notes
    -----
    Signal directions:

    * ``HIGH_OMISSION`` (deciles 9–10) → **SHORT** candidate.
      Management is avoiding topics they historically emphasised — a bearish
      information asymmetry signal.
    * ``LOW_OMISSION`` (deciles 1–2) → **LONG** candidate.
      Language is consistent with historical baseline — no negative silence.
    * ``NEUTRAL`` (deciles 3–8) → no directional bet.
    """
    signal_df = signal_df.copy()

    def _decile_rank(group: pd.DataFrame) -> pd.Series:
        return pd.qcut(
            group["omission_score"],
            q=10,
            labels=False,
            duplicates="drop",
        ).add(1)  # 1-indexed

    # Cross-sectional decile by quarter-year
    signal_df["omission_decile"] = signal_df.groupby(["year", "quarter"], group_keys=False).apply(
        _decile_rank
    )

    # Handle cases where qcut fails (too few companies in a quarter)
    signal_df["omission_decile"] = signal_df["omission_decile"].fillna(5).astype(int)

    # Signal label
    def _label(decile: int) -> str:
        if decile >= 9:
            return "HIGH_OMISSION"
        if decile <= 2:
            return "LOW_OMISSION"
        return "NEUTRAL"

    signal_df["signal_label"] = signal_df["omission_decile"].apply(_label)

    logger.info(
        "Signal label distribution:\n%s",
        signal_df["signal_label"].value_counts().to_string(),
    )
    return signal_df
