"""Offline tests for omission detection, point-in-time vocabulary and the backtest."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import config
from src.backtest import run_long_short_backtest
from src.omission_detector import (
    classify_omission_regime,
    compute_omission_score,
    detect_omissions,
)
from src.text_processor import extract_ngrams
from src.tfidf_engine import build_standard_vocabulary


# ------------------------------------------------------------ omission logic
def _vocab():
    return pd.DataFrame(
        {
            "term": ["supply_chain", "gross_margin", "guidance"],
            "avg_tfidf_weight": [0.20, 0.10, 0.05],
            "core_vocab": [True, False, False],
        }
    )


def test_detect_omissions_flags_terms_below_threshold():
    feats = ["supply_chain", "gross_margin", "guidance"]
    cur = np.array([0.0, 0.10, 0.05])  # supply_chain vanished
    out = detect_omissions(cur, _vocab(), feats).set_index("term")
    assert bool(out.loc["supply_chain", "omission_event"])
    assert not out.loc[["gross_margin", "guidance"], "omission_event"].any()
    assert out.loc["supply_chain", "presence_ratio"] == 0.0


def test_core_vocab_omission_is_weighted_more():
    feats = ["supply_chain", "gross_margin", "guidance"]
    vocab = _vocab()
    vocab["avg_tfidf_weight"] = 0.10
    out = detect_omissions(np.zeros(3), vocab, feats).set_index("term")
    assert out.loc["supply_chain", "omission_severity"] > out.loc["gross_margin", "omission_severity"]
    assert out.loc["gross_margin", "omission_severity"] == pytest.approx(0.10)  # (1-0)*0.10*1.0


def test_term_missing_from_current_features_counts_as_omitted():
    out = detect_omissions(np.array([0.5]), _vocab(), ["other_term"])
    assert out["omission_event"].all()


def test_omission_score_normalised_and_zero_when_none():
    feats = ["supply_chain", "gross_margin", "guidance"]
    none = detect_omissions(np.array([0.2, 0.1, 0.05]), _vocab(), feats)
    assert compute_omission_score(none) == 0.0
    some = detect_omissions(np.zeros(3), _vocab(), feats)
    assert compute_omission_score(some) == pytest.approx(some["omission_severity"].sum() / 3)
    assert compute_omission_score(pd.DataFrame()) == 0.0


# ---------------------------------------------- classification (cross-section)
def _signals(seed=0, quarters=2, n=40):
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(1, quarters + 1):
        for i in range(n):
            rows.append({"year": 2022, "quarter": f"Q{q}", "ticker": f"T{i}",
                         "omission_score": rng.random(), "ret_30d": rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_regime_labels_follow_deciles_within_each_quarter():
    out = classify_omission_regime(_signals())
    for _, g in out.groupby(["year", "quarter"]):
        hi = g[g["signal_label"] == "HIGH_OMISSION"]["omission_score"].min()
        lo = g[g["signal_label"] == "LOW_OMISSION"]["omission_score"].max()
        mid = g[g["signal_label"] == "NEUTRAL"]["omission_score"]
        assert lo < mid.min() and mid.max() < hi
        assert set(g["omission_decile"]).issuperset({1, 10})


def test_regime_is_independent_of_other_quarters():
    """A quarter's labels must not change when later quarters are added/removed."""
    full = classify_omission_regime(_signals(quarters=3))
    q1_only = classify_omission_regime(_signals(quarters=3).query("quarter == 'Q1'"))
    a = full.query("quarter == 'Q1'").sort_values("ticker")["signal_label"].to_numpy()
    b = q1_only.sort_values("ticker")["signal_label"].to_numpy()
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------ point-in-time
def test_standard_vocabulary_excludes_current_quarter():
    quarters = ["Q1", "Q2", "Q3", "Q4", "Q1", "Q2"]
    years = [2021, 2021, 2021, 2021, 2022, 2022]
    corpus = pd.DataFrame(
        {"cik": "1", "quarter": quarters, "year": years, "coverage_quality": "transcript"}
    )
    mat = np.zeros((6, 3))
    mat[:5, 0] = 0.10          # term A in history
    mat[:5, 1] = 0.05
    mat[5, :] = [9.0, 9.0, 9.0]  # "current" quarter: huge, must be ignored
    vocab = build_standard_vocabulary(
        "1", corpus, sparse.csr_matrix(mat), ["a", "b", "c"], trailing_quarters=4
    ).set_index("term")
    assert vocab.loc["a", "avg_tfidf_weight"] == pytest.approx(0.10)
    assert "c" not in vocab.index  # only present in the current quarter
    assert bool(vocab.loc["a", "core_vocab"])


def test_standard_vocabulary_needs_history():
    corpus = pd.DataFrame(
        {"cik": "1", "quarter": ["Q1", "Q2"], "year": [2021, 2021], "coverage_quality": "transcript"}
    )
    out = build_standard_vocabulary("1", corpus, sparse.csr_matrix(np.ones((2, 3))), ["a", "b", "c"], 4)
    assert out is None


# ----------------------------------------------------------------- backtest
def test_long_short_backtest_signs_and_counts():
    df = pd.DataFrame(
        {
            "year": 2022,
            "quarter": "Q1",
            "ticker": list("ABCD"),
            "signal_label": ["LOW_OMISSION", "LOW_OMISSION", "HIGH_OMISSION", "NEUTRAL"],
            "ret_30d": [0.04, 0.02, -0.01, 0.50],
        }
    )
    res = run_long_short_backtest(df)
    row = res.iloc[0]
    assert row["long_ret"] == pytest.approx(0.03)
    assert row["short_ret"] == pytest.approx(-0.01)
    assert row["ls_ret"] == pytest.approx(0.04)  # neutral name ignored
    assert (row["n_long"], row["n_short"]) == (2, 1)


def test_backtest_skips_quarters_with_empty_book():
    df = pd.DataFrame(
        {"year": 2022, "quarter": "Q1", "ticker": ["A"], "signal_label": ["LOW_OMISSION"], "ret_30d": [0.1]}
    )
    assert run_long_short_backtest(df).empty


def test_extract_ngrams():
    assert extract_ngrams(["a", "b", "c"], (2, 3)) == ["a_b", "b_c", "a_b_c"]
    assert extract_ngrams(["a"], (2, 3)) == []
