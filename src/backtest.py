"""
backtest.py
-----------
Long-short portfolio backtest for the earnings call silence signal.

Strategy construction
---------------------
Each quarter:
  - **SHORT book**: equal-weight the HIGH_OMISSION decile (deciles 9–10).
  - **LONG book**: equal-weight the LOW_OMISSION decile (deciles 1–2).
  - Net long-short return = mean(LONG) − mean(SHORT).
  - Rebalancing aligned to earnings seasons: Jan, Apr, Jul, Oct.

Two variants are implemented:
1. **Unconstrained**: raw long-short without sector neutralisation.
2. **Sector-neutral**: long/short constructed *within* each sector,
   eliminating sector-level beta exposure and isolating pure omission alpha.

The Sharpe ratio comparison between the two variants is a key diagnostic:
if the sector-neutral Sharpe is meaningfully higher, the unconstrained strategy
is partly picking up sector momentum rather than stock-specific omission effects.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper: quarterly return pivot
# ---------------------------------------------------------------------------

def _build_quarter_returns(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot signal_df into a period × company returns structure.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with returns attached (from ``fetch_post_earnings_returns``).

    Returns
    -------
    pd.DataFrame
        Wide DataFrame indexed by (year, quarter) with ticker columns.
        Cell values are ret_30d forward returns.
    """
    required = {"ticker", "year", "quarter", "ret_30d", "signal_label"}
    missing = required - set(signal_df.columns)
    if missing:
        raise ValueError(f"signal_df missing columns: {missing}")

    pivot = signal_df.pivot_table(
        index=["year", "quarter"],
        columns="ticker",
        values="ret_30d",
        aggfunc="first",
    )
    return pivot


# ---------------------------------------------------------------------------
# 1. run_long_short_backtest
# ---------------------------------------------------------------------------

def run_long_short_backtest(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Run the unconstrained long-short backtest.

    Each quarter:
    - LONG = equal-weight LOW_OMISSION companies (deciles 1–2).
    - SHORT = equal-weight HIGH_OMISSION companies (deciles 9–10).
    - Net return = mean(LONG ret_30d) − mean(SHORT ret_30d).

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``signal_label``, ``ret_30d``, ``year``, ``quarter``,
        ``ticker``.

    Returns
    -------
    pd.DataFrame
        Quarterly time series with columns:
        ``year``, ``quarter``, ``long_ret``, ``short_ret``, ``ls_ret``,
        ``n_long``, ``n_short``.
    """
    rows: list[dict] = []

    for (year, quarter), grp in signal_df.groupby(["year", "quarter"]):
        long_grp = grp[grp["signal_label"] == "LOW_OMISSION"]["ret_30d"].dropna()
        short_grp = grp[grp["signal_label"] == "HIGH_OMISSION"]["ret_30d"].dropna()

        if long_grp.empty or short_grp.empty:
            continue

        long_ret = float(long_grp.mean())
        short_ret = float(short_grp.mean())
        ls_ret = long_ret - short_ret

        rows.append(
            {
                "year": year,
                "quarter": quarter,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "ls_ret": ls_ret,
                "n_long": int(len(long_grp)),
                "n_short": int(len(short_grp)),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["year", "quarter"]).reset_index(drop=True)
    logger.info(
        "Backtest complete: %d periods, mean LS return = %.4f",
        len(results_df),
        results_df["ls_ret"].mean() if not results_df.empty else np.nan,
    )
    return results_df


# ---------------------------------------------------------------------------
# 2. compute_performance_metrics
# ---------------------------------------------------------------------------

def compute_performance_metrics(backtest_results_df: pd.DataFrame) -> dict:
    """Compute risk-adjusted performance metrics for the backtest.

    Parameters
    ----------
    backtest_results_df : pd.DataFrame
        Output of ``run_long_short_backtest`` or ``run_sector_neutral_backtest``.

    Returns
    -------
    dict
        Keys: ``annualized_return``, ``annualized_sharpe``, ``max_drawdown``,
        ``max_drawdown_start``, ``max_drawdown_end``, ``win_rate``,
        ``calmar_ratio``, ``n_periods``.
    """
    if backtest_results_df.empty or "ls_ret" not in backtest_results_df.columns:
        logger.warning("Empty backtest results — cannot compute metrics")
        return {}

    returns = backtest_results_df["ls_ret"].dropna()
    n = len(returns)

    # Annualisation: 4 quarters per year
    periods_per_year = 4.0
    ann_return = float(returns.mean() * periods_per_year)
    ann_vol = float(returns.std() * np.sqrt(periods_per_year))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else np.nan

    # Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    dd_end_idx = drawdown.idxmin()
    dd_start_idx = cum_returns[:dd_end_idx].idxmax()

    try:
        dd_start = backtest_results_df.loc[dd_start_idx, ["year", "quarter"]].to_dict()
        dd_end = backtest_results_df.loc[dd_end_idx, ["year", "quarter"]].to_dict()
    except Exception:
        dd_start = dd_end = {}

    win_rate = float((returns > 0).mean())
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else np.nan

    metrics = {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "annualized_sharpe": sharpe,
        "max_drawdown": max_dd,
        "max_drawdown_start": dd_start,
        "max_drawdown_end": dd_end,
        "win_rate": win_rate,
        "calmar_ratio": calmar,
        "n_periods": n,
    }

    # Pretty print
    print("\n── Performance Metrics ─────────────────────────────────────────────────")
    print(f"  Annualized Return  : {ann_return:>10.2%}")
    print(f"  Annualized Vol     : {ann_vol:>10.2%}")
    print(f"  Sharpe Ratio       : {sharpe:>10.3f}")
    print(f"  Max Drawdown       : {max_dd:>10.2%}")
    print(f"  Calmar Ratio       : {calmar:>10.3f}")
    print(f"  Win Rate           : {win_rate:>10.2%}")
    print(f"  Periods            : {n:>10}")
    print()
    return metrics


# ---------------------------------------------------------------------------
# 3. run_sector_neutral_backtest
# ---------------------------------------------------------------------------

def run_sector_neutral_backtest(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Run the sector-neutral long-short backtest.

    Long/short books are constructed *within* each sector each quarter.
    This removes sector-level beta and isolates pure omission alpha.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``sector`` column.

    Returns
    -------
    pd.DataFrame
        Same structure as ``run_long_short_backtest`` output, representing
        the sector-neutral strategy return series.

    Notes
    -----
    **When to prefer sector-neutral**: if the unconstrained strategy's alpha
    largely disappears when sector-neutralised, the omission signal is
    predominantly capturing sector-level dynamics (e.g. all tech companies
    omitting "supply chain" during a semiconductor downcycle) rather than
    stock-specific management evasion.  Sector-neutral isolation is the cleaner
    test of the paper's core thesis.
    """
    if "sector" not in signal_df.columns:
        logger.warning("'sector' column missing — falling back to unconstrained backtest")
        return run_long_short_backtest(signal_df)

    rows: list[dict] = []

    for (year, quarter), period_grp in signal_df.groupby(["year", "quarter"]):
        long_rets: list = []
        short_rets: list = []

        for sector, sector_grp in period_grp.groupby("sector"):
            l = sector_grp[sector_grp["signal_label"] == "LOW_OMISSION"]["ret_30d"].dropna()
            s = sector_grp[sector_grp["signal_label"] == "HIGH_OMISSION"]["ret_30d"].dropna()

            if not l.empty:
                long_rets.extend(l.tolist())
            if not s.empty:
                short_rets.extend(s.tolist())

        if not long_rets or not short_rets:
            continue

        long_ret = float(np.mean(long_rets))
        short_ret = float(np.mean(short_rets))
        ls_ret = long_ret - short_ret

        rows.append(
            {
                "year": year,
                "quarter": quarter,
                "long_ret": long_ret,
                "short_ret": short_ret,
                "ls_ret": ls_ret,
                "n_long": len(long_rets),
                "n_short": len(short_rets),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["year", "quarter"]).reset_index(drop=True)
    logger.info(
        "Sector-neutral backtest complete: %d periods, mean LS return = %.4f",
        len(results_df),
        results_df["ls_ret"].mean() if not results_df.empty else np.nan,
    )
    return results_df
