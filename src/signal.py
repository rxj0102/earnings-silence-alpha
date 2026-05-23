"""
signal.py
---------
Statistical validation of the omission signal against forward stock returns.

This module fetches post-earnings returns via yfinance, then runs a suite of
statistical tests to validate the omission → return relationship:

1. t-test: HIGH_OMISSION group vs LOW_OMISSION group
2. Information Coefficient (Spearman rank IC)
3. OLS regression with sector and quarter fixed effects
4. Sector-level breakdown

Thesis on return horizon
------------------------
The 30-day POST_EARNINGS_WINDOW is intentional.  The research hypothesis is
that the market *fails to price* the omission signal at the earnings announcement.
The 1-day reaction captures the market's *immediate* response to headline
numbers; the 30-day window captures post-earnings drift as fundamental reality
slowly surfaces.  If the 30-day IC is stronger than the 1-day IC, that is
consistent with the drift hypothesis — i.e. the signal is genuinely alpha-
generating, not just a proxy for existing sell-side expectations.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: fetch adjusted close prices via yfinance
# ---------------------------------------------------------------------------

def _fetch_price_data(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Fetch adjusted close prices for a list of tickers via yfinance.

    Parameters
    ----------
    tickers : list[str]
        Exchange ticker symbols.
    start : str
        ISO start date (YYYY-MM-DD).
    end : str
        ISO end date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Wide DataFrame: index = date, columns = ticker, values = adj close.
    """
    try:
        import yfinance as yf

        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]] if "Close" in data.columns else data
        return close
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        return pd.DataFrame()


def _compute_return(prices: pd.DataFrame, ticker: str, date: str, window: int) -> Optional[float]:
    """Compute the window-day forward return for a ticker from a given date.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price DataFrame (index = date, columns = tickers).
    ticker : str
        Ticker symbol.
    date : str
        Start date (ISO format).
    window : int
        Return window in calendar days.

    Returns
    -------
    float or None
        Forward return (decimal), or None if data is insufficient.
    """
    if ticker not in prices.columns:
        return None

    series = prices[ticker].dropna()
    date_idx = pd.Timestamp(date)

    # Find the first available trading day on or after `date`
    after = series[series.index >= date_idx]
    if len(after) < 2:
        return None

    start_price = after.iloc[0]

    # Find the trading day closest to `date + window` calendar days
    target = date_idx + pd.Timedelta(days=window)
    end_candidates = series[series.index >= target]
    if end_candidates.empty:
        # Use last available price
        end_price = series.iloc[-1]
    else:
        end_price = end_candidates.iloc[0]

    if start_price == 0 or pd.isna(start_price) or pd.isna(end_price):
        return None

    return float(end_price / start_price - 1)


# ---------------------------------------------------------------------------
# 1. fetch_post_earnings_returns
# ---------------------------------------------------------------------------

def fetch_post_earnings_returns(
    signal_df: pd.DataFrame,
    window: int = config.POST_EARNINGS_WINDOW,
) -> pd.DataFrame:
    """Compute post-earnings returns for all signal rows.

    Fetches adjusted close prices from yfinance and computes 1-day, 5-day,
    and 30-day forward returns anchored to each earnings filing date.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Output of ``classify_omission_regime``.  Must contain columns
        ``ticker``, ``earnings_date``.
    window : int, optional
        Primary return window in calendar days (default ``POST_EARNINGS_WINDOW = 30``).

    Returns
    -------
    pd.DataFrame
        signal_df enriched with ``ret_1d``, ``ret_5d``, ``ret_30d``.
    """
    signal_df = signal_df.copy()
    tickers = signal_df["ticker"].unique().tolist()

    # Determine global date range
    valid_dates = signal_df["earnings_date"].dropna()
    if valid_dates.empty:
        logger.warning("No valid earnings dates — cannot fetch returns")
        signal_df["ret_1d"] = np.nan
        signal_df["ret_5d"] = np.nan
        signal_df["ret_30d"] = np.nan
        return signal_df

    min_date = pd.to_datetime(valid_dates).min() - pd.Timedelta(days=5)
    max_date = pd.to_datetime(valid_dates).max() + pd.Timedelta(days=window + 10)

    logger.info(
        "Fetching prices for %d tickers: %s to %s",
        len(tickers),
        min_date.date(),
        max_date.date(),
    )
    prices = _fetch_price_data(tickers, str(min_date.date()), str(max_date.date()))

    ret_1d: list = []
    ret_5d: list = []
    ret_30d: list = []

    for _, row in signal_df.iterrows():
        date = row.get("earnings_date")
        ticker = row.get("ticker")

        if pd.isna(date) or prices.empty:
            ret_1d.append(np.nan)
            ret_5d.append(np.nan)
            ret_30d.append(np.nan)
            continue

        ret_1d.append(_compute_return(prices, ticker, str(date), 1))
        ret_5d.append(_compute_return(prices, ticker, str(date), 5))
        ret_30d.append(_compute_return(prices, ticker, str(date), window))

    signal_df["ret_1d"] = ret_1d
    signal_df["ret_5d"] = ret_5d
    signal_df["ret_30d"] = ret_30d

    n_valid = signal_df["ret_30d"].notna().sum()
    logger.info("Return data populated: %d/%d rows have valid ret_30d", n_valid, len(signal_df))
    return signal_df


# ---------------------------------------------------------------------------
# 2. run_ttest_validation
# ---------------------------------------------------------------------------

def run_ttest_validation(signal_df: pd.DataFrame) -> dict:
    """Run two-sample t-tests comparing HIGH_OMISSION vs LOW_OMISSION groups.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``signal_label``, ``ret_1d``, ``ret_5d``, ``ret_30d``.

    Returns
    -------
    dict
        Keys: ``"ret_1d"``, ``"ret_5d"``, ``"ret_30d"``.
        Each value is a dict with keys ``t_stat``, ``p_value``,
        ``high_mean``, ``low_mean``, ``n_high``, ``n_low``.

    Notes
    -----
    The 30-day window tests the drift hypothesis: we expect HIGH_OMISSION
    firms to have *lower* subsequent returns as the market gradually prices
    in the information asymmetry signalled by topic omission.  A statistically
    significant negative mean for HIGH_OMISSION vs positive for LOW_OMISSION
    would support the research hypothesis.
    """
    high = signal_df[signal_df["signal_label"] == "HIGH_OMISSION"]
    low = signal_df[signal_df["signal_label"] == "LOW_OMISSION"]

    results: dict = {}
    horizons = {"ret_1d": "1-Day", "ret_5d": "5-Day", "ret_30d": "30-Day"}

    header = f"{'Horizon':<10} {'HIGH mean':>12} {'LOW mean':>12} {'t-stat':>10} {'p-value':>10} {'N(H)':>8} {'N(L)':>8}"
    print("\n── T-Test: HIGH_OMISSION vs LOW_OMISSION ──────────────────────────────")
    print(header)
    print("─" * len(header))

    for col, label in horizons.items():
        h_vals = high[col].dropna()
        l_vals = low[col].dropna()

        if len(h_vals) < 3 or len(l_vals) < 3:
            logger.warning("Insufficient data for t-test on %s", col)
            results[col] = {
                "t_stat": np.nan,
                "p_value": np.nan,
                "high_mean": h_vals.mean() if len(h_vals) else np.nan,
                "low_mean": l_vals.mean() if len(l_vals) else np.nan,
                "n_high": len(h_vals),
                "n_low": len(l_vals),
            }
            continue

        t_stat, p_val = stats.ttest_ind(h_vals, l_vals, equal_var=False)
        res = {
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "high_mean": float(h_vals.mean()),
            "low_mean": float(l_vals.mean()),
            "n_high": int(len(h_vals)),
            "n_low": int(len(l_vals)),
        }
        results[col] = res

        print(
            f"{label:<10} {res['high_mean']:>11.4f} {res['low_mean']:>12.4f} "
            f"{t_stat:>10.3f} {p_val:>10.4f} {res['n_high']:>8} {res['n_low']:>8}"
        )

    print()
    return results


# ---------------------------------------------------------------------------
# 3. compute_information_coefficient
# ---------------------------------------------------------------------------

def compute_information_coefficient(signal_df: pd.DataFrame) -> dict:
    """Compute Spearman rank IC between omission_score and forward returns.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``omission_score``, ``ret_1d``, ``ret_5d``,
        ``ret_30d``.

    Returns
    -------
    dict
        Keys: ``"ret_1d"``, ``"ret_5d"``, ``"ret_30d"``.
        Each value: dict with ``ic``, ``p_value``.

    Notes
    -----
    We expect **negative** IC (higher omission score → lower returns).
    The IC measures the monotonic rank relationship — it is robust to outlier
    returns unlike Pearson correlation.  An IC of -0.05 to -0.10 would be
    economically meaningful for a cross-sectional equity signal.
    """
    horizons = {"ret_1d": "1-Day", "ret_5d": "5-Day", "ret_30d": "30-Day"}
    results: dict = {}

    print("\n── Information Coefficient (Spearman Rank IC) ─────────────────────────")
    print(f"{'Horizon':<10} {'IC':>10} {'p-value':>12} {'N':>8}")
    print("─" * 44)

    for col, label in horizons.items():
        valid = signal_df[["omission_score", col]].dropna()
        if len(valid) < 5:
            logger.warning("Insufficient data for IC on %s", col)
            results[col] = {"ic": np.nan, "p_value": np.nan}
            continue

        ic, p_val = stats.spearmanr(valid["omission_score"], valid[col])
        results[col] = {"ic": float(ic), "p_value": float(p_val)}
        print(f"{label:<10} {ic:>10.4f} {p_val:>12.4f} {len(valid):>8}")

    print()
    return results


# ---------------------------------------------------------------------------
# 4. run_ols_regression
# ---------------------------------------------------------------------------

def run_ols_regression(signal_df: pd.DataFrame):
    """OLS regression of ret_30d on omission score with fixed effects.

    Specification::

        ret_30d ~ omission_score + n_core_vocab_omissions + sector_FE + quarter_FE

    Estimated with heteroskedasticity-robust standard errors (HC3).

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame.  Must contain ``ret_30d``, ``omission_score``,
        ``n_core_vocab_omissions``, and optionally ``sector`` and ``quarter``.

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        Full OLS result object.  ``.summary()`` prints the regression table.

    Notes
    -----
    The key coefficient is ``omission_score`` — we expect a negative, statistically
    significant estimate.  HC3 robust standard errors are appropriate here because
    we anticipate heteroskedasticity from cross-sectional returns.
    """
    reg_df = signal_df.dropna(subset=["ret_30d", "omission_score"]).copy()
    if len(reg_df) < 10:
        logger.warning("Insufficient data for OLS regression (%d rows)", len(reg_df))
        return None

    y = reg_df["ret_30d"]

    X = reg_df[["omission_score", "n_core_vocab_omissions"]].copy()

    # Sector fixed effects (dummy encoding)
    if "sector" in reg_df.columns:
        sector_dummies = pd.get_dummies(reg_df["sector"], prefix="sector", drop_first=True)
        X = pd.concat([X, sector_dummies], axis=1)

    # Quarter fixed effects
    if "quarter" in reg_df.columns:
        quarter_dummies = pd.get_dummies(reg_df["quarter"], prefix="q", drop_first=True)
        X = pd.concat([X, quarter_dummies], axis=1)

    X = sm.add_constant(X).astype(float)

    model = sm.OLS(y, X)
    result = model.fit(cov_type="HC3")

    print("\n── OLS Regression: ret_30d ~ omission_score + controls ────────────────")
    print(result.summary())
    print()
    return result


# ---------------------------------------------------------------------------
# 5. compute_sector_breakdown
# ---------------------------------------------------------------------------

def compute_sector_breakdown(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Compute signal statistics broken down by sector.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``sector``, ``omission_score``,
        ``signal_label``, ``ret_30d``.

    Returns
    -------
    pd.DataFrame
        Sector-level summary: ``mean_omission_score``, ``mean_ret_30d_high``,
        ``ic_sector``, ``n_obs``.

    Notes
    -----
    **Sector heterogeneity hypothesis**: the omission signal is likely stronger
    in sectors with complex, multi-dimensional business narratives — Technology
    (competitive moats, R&D cycles), Industrials (supply chains, backlog), and
    Healthcare (pipeline assets, regulatory timelines).  In simpler business
    models such as Utilities or REITs, quarterly language is more formulaic
    and omissions carry less incremental information.  This hypothesis should
    be tested in a larger universe extension.
    """
    if "sector" not in signal_df.columns:
        logger.warning("'sector' column not found — returning empty sector breakdown")
        return pd.DataFrame()

    rows: list[dict] = []
    for sector, grp in signal_df.groupby("sector"):
        high_grp = grp[grp["signal_label"] == "HIGH_OMISSION"]

        # Sector IC
        valid = grp[["omission_score", "ret_30d"]].dropna()
        if len(valid) >= 5:
            ic, _ = stats.spearmanr(valid["omission_score"], valid["ret_30d"])
        else:
            ic = np.nan

        rows.append(
            {
                "sector": sector,
                "mean_omission_score": float(grp["omission_score"].mean()),
                "mean_ret_30d_high_omission": float(
                    high_grp["ret_30d"].mean() if not high_grp.empty else np.nan
                ),
                "ic_sector": float(ic),
                "n_obs": int(len(grp)),
                "n_high_omission": int(len(high_grp)),
            }
        )

    breakdown_df = pd.DataFrame(rows).sort_values("ic_sector")
    print("\n── Sector Breakdown ────────────────────────────────────────────────────")
    print(breakdown_df.to_string(index=False))
    return breakdown_df
