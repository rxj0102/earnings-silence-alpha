"""
visualize.py
------------
All publication-quality charts for the Earnings Call Silence Mapping paper.

Charts are saved to ``outputs/`` at 300 DPI in PNG format.
Style: seaborn-whitegrid throughout.

Chart inventory
---------------
1. plot_omission_score_distribution  — histogram coloured by signal label
2. plot_topic_presence_matrix        — heatmap (Figure 1 of the paper)
3. plot_omission_score_vs_return     — scatter with OLS trendline
4. plot_return_by_decile             — bar chart by omission decile
5. plot_cumulative_pnl               — cumulative LS PnL with drawdown shading
6. plot_top_omitted_terms_wordcloud  — word cloud of most frequently omitted terms
"""

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE_WIDE = (14, 6)
FIGSIZE_SQUARE = (10, 8)
FIGSIZE_TALL = (10, 10)
DPI = 300
SIGNAL_PALETTE = {
    "HIGH_OMISSION": "#d62728",   # red  → short
    "NEUTRAL": "#aec7e8",         # light blue → neutral
    "LOW_OMISSION": "#1f77b4",    # blue → long
}


def _save(fig: plt.Figure, filename: str) -> Path:
    """Save figure to outputs/ and close it."""
    path = config.OUTPUTS_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. plot_omission_score_distribution
# ---------------------------------------------------------------------------

def plot_omission_score_distribution(signal_df: pd.DataFrame) -> Path:
    """Histogram of omission scores coloured by signal label.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``omission_score`` and ``signal_label``.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    for label in ["LOW_OMISSION", "NEUTRAL", "HIGH_OMISSION"]:
        subset = signal_df[signal_df["signal_label"] == label]["omission_score"]
        ax.hist(
            subset,
            bins=30,
            alpha=0.6,
            color=SIGNAL_PALETTE[label],
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Omission Score", fontsize=13)
    ax.set_ylabel("Frequency", fontsize=13)
    ax.set_title(
        "Distribution of Earnings Call Omission Scores by Signal Label",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(title="Signal Label", fontsize=11)
    ax.axvline(
        signal_df["omission_score"].median(),
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Median",
    )

    return _save(fig, "01_omission_score_distribution.png")


# ---------------------------------------------------------------------------
# 2. plot_topic_presence_matrix
# ---------------------------------------------------------------------------

def plot_topic_presence_matrix(
    cik: str,
    corpus_df: pd.DataFrame,
    vocabularies: dict,
    tfidf_matrix,
    feature_names: list,
) -> Path:
    """Heatmap showing topic presence intensity over time (Figure 1 of paper).

    X-axis: quarters (8 most recent).
    Y-axis: top 20 standard vocabulary terms.
    Colour: TF-IDF intensity.  Red cells = omission events in most recent quarter.

    Parameters
    ----------
    cik : str
        Company CIK.
    corpus_df : pd.DataFrame
        Preprocessed corpus.
    vocabularies : dict
        Standard vocabularies dict.
    tfidf_matrix : sparse matrix
        Full corpus TF-IDF matrix.
    feature_names : list[str]
        Vocabulary list.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    from src.tfidf_engine import get_company_quarter_vector
    from src.omission_detector import detect_omissions

    fit_df = corpus_df[corpus_df["coverage_quality"] == "transcript"].copy()
    company_df = fit_df[fit_df["cik"].astype(str) == str(cik)].sort_values(
        ["year", "quarter"]
    )
    if company_df.empty or cik not in vocabularies:
        logger.warning("Cannot plot topic matrix for CIK %s — insufficient data", cik)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return _save(fig, f"02_topic_presence_{cik}.png")

    # Use the 8 most recent quarters
    recent = company_df.tail(config.LOOKBACK_QUARTERS)
    quarters_list = [
        f"{row['quarter']}/{int(row['year'])}" for _, row in recent.iterrows()
    ]

    # Get the standard vocabulary from the most recent entry
    cik_vocabs = vocabularies.get(cik, {})
    if not cik_vocabs:
        logger.warning("No vocabulary for CIK %s", cik)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No vocabulary built", ha="center", va="center")
        return _save(fig, f"02_topic_presence_{cik}.png")

    latest_key = sorted(cik_vocabs.keys(), key=lambda k: (k[1], k[0]))[-1]
    std_vocab = cik_vocabs[latest_key]
    terms = std_vocab["term"].tolist()[:config.STANDARD_VOCAB_SIZE]

    # Build presence matrix
    feature_index = {t: i for i, t in enumerate(feature_names)}
    matrix_data = np.zeros((len(terms), len(recent)))

    for col_idx, (_, row) in enumerate(recent.iterrows()):
        vec = get_company_quarter_vector(
            tfidf_matrix, corpus_df, cik, row["quarter"], int(row["year"])
        )
        if vec is not None:
            for row_idx, term in enumerate(terms):
                t_idx = feature_index.get(term)
                if t_idx is not None:
                    matrix_data[row_idx, col_idx] = vec[t_idx]

    # Identify omission events in the last quarter
    last_vec = get_company_quarter_vector(
        tfidf_matrix, corpus_df, cik,
        recent.iloc[-1]["quarter"], int(recent.iloc[-1]["year"])
    )
    omission_mask = np.zeros(len(terms), dtype=bool)
    if last_vec is not None:
        omission_df = detect_omissions(last_vec, std_vocab, feature_names)
        for i, term in enumerate(terms):
            row = omission_df[omission_df["term"] == term]
            if not row.empty:
                omission_mask[i] = bool(row.iloc[0]["omission_event"])

    # Build custom colormap: white → blue for presence, red for omission
    presence_matrix = pd.DataFrame(matrix_data, index=terms, columns=quarters_list)

    fig, ax = plt.subplots(figsize=(FIGSIZE_TALL[0] + 2, FIGSIZE_TALL[1]))
    cmap_blue = sns.light_palette("#1f77b4", as_cmap=True)
    sns.heatmap(
        presence_matrix,
        ax=ax,
        cmap=cmap_blue,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "TF-IDF Weight", "shrink": 0.6},
        annot=False,
    )

    # Overlay red for omission cells in the last column
    for row_idx, is_omission in enumerate(omission_mask):
        if is_omission:
            ax.add_patch(
                mpatches.Rectangle(
                    (len(quarters_list) - 1, row_idx),
                    1,
                    1,
                    fill=True,
                    color="#d62728",
                    alpha=0.85,
                )
            )

    # Tick styling
    ax.set_yticklabels(
        [t.replace("_", " ") for t in terms], rotation=0, fontsize=9
    )
    ax.set_xticklabels(quarters_list, rotation=45, ha="right", fontsize=9)

    ticker = company_df["ticker"].iloc[0] if "ticker" in company_df.columns else cik
    ax.set_title(
        f"Topic Presence Matrix — {ticker}\n"
        "(Red = Omission Event in Most Recent Quarter)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Quarter", fontsize=11)
    ax.set_ylabel("Standard Vocabulary Term", fontsize=11)

    # Legend patches
    blue_patch = mpatches.Patch(color="#1f77b4", label="Present (TF-IDF weight)")
    white_patch = mpatches.Patch(color="white", ec="gray", label="Absent")
    red_patch = mpatches.Patch(color="#d62728", label="Omission Event")
    ax.legend(handles=[blue_patch, white_patch, red_patch], loc="upper left", fontsize=9)

    return _save(fig, f"02_topic_presence_{ticker}.png")


# ---------------------------------------------------------------------------
# 3. plot_omission_score_vs_return
# ---------------------------------------------------------------------------

def plot_omission_score_vs_return(signal_df: pd.DataFrame) -> Path:
    """Scatter plot: omission score vs 30-day return, OLS trendline, IC annotation.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``omission_score``, ``ret_30d``, ``sector``.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    from scipy import stats

    plot_df = signal_df.dropna(subset=["omission_score", "ret_30d"]).copy()
    if plot_df.empty:
        logger.warning("No data for scatter plot")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return _save(fig, "03_omission_vs_return.png")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    sectors = plot_df["sector"].unique() if "sector" in plot_df.columns else ["All"]
    palette = sns.color_palette("tab10", len(sectors))

    for i, sector in enumerate(sectors):
        sub = plot_df[plot_df["sector"] == sector] if "sector" in plot_df.columns else plot_df
        ax.scatter(
            sub["omission_score"],
            sub["ret_30d"],
            color=palette[i],
            alpha=0.6,
            s=40,
            label=sector,
        )

    # OLS trendline
    x = plot_df["omission_score"].values
    y = plot_df["ret_30d"].values
    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line,
        intercept + slope * x_line,
        color="black",
        linewidth=2,
        linestyle="--",
        label=f"OLS (slope={slope:.3f})",
    )

    # IC annotation
    ic, ic_p = stats.spearmanr(x, y)
    ax.text(
        0.02, 0.95,
        f"Spearman IC = {ic:.3f}  (p={ic_p:.3f})",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Omission Score", fontsize=13)
    ax.set_ylabel("30-Day Post-Earnings Return", fontsize=13)
    ax.set_title(
        "Omission Score vs 30-Day Post-Earnings Return",
        fontsize=14,
        fontweight="bold",
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-")
    ax.legend(title="Sector", fontsize=9, loc="upper right")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))

    return _save(fig, "03_omission_vs_return.png")


# ---------------------------------------------------------------------------
# 4. plot_return_by_decile
# ---------------------------------------------------------------------------

def plot_return_by_decile(signal_df: pd.DataFrame) -> Path:
    """Bar chart: mean 30-day return by omission decile.

    This is the key validation chart.  A monotonically decreasing pattern
    (decile 1 = highest return, decile 10 = lowest return) would strongly
    support the omission signal hypothesis.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``omission_decile`` and ``ret_30d``.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    plot_df = signal_df.dropna(subset=["omission_decile", "ret_30d"]).copy()
    if plot_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        return _save(fig, "04_return_by_decile.png")

    decile_ret = (
        plot_df.groupby("omission_decile")["ret_30d"]
        .agg(["mean", "sem", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = [
        "#1f77b4" if d <= 2 else ("#d62728" if d >= 9 else "#aec7e8")
        for d in decile_ret["omission_decile"]
    ]
    bars = ax.bar(
        decile_ret["omission_decile"],
        decile_ret["mean"],
        color=colors,
        width=0.7,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.errorbar(
        decile_ret["omission_decile"],
        decile_ret["mean"],
        yerr=1.96 * decile_ret["sem"],
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.2,
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Omission Score Decile (1 = Lowest Omission → 10 = Highest)", fontsize=12)
    ax.set_ylabel("Mean 30-Day Post-Earnings Return", fontsize=12)
    ax.set_title(
        "Post-Earnings Return by Omission Decile\n(Blue = LONG, Red = SHORT)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))

    # Legend
    long_patch = mpatches.Patch(color="#1f77b4", label="LONG (Deciles 1–2)")
    neutral_patch = mpatches.Patch(color="#aec7e8", label="NEUTRAL (Deciles 3–8)")
    short_patch = mpatches.Patch(color="#d62728", label="SHORT (Deciles 9–10)")
    ax.legend(handles=[long_patch, neutral_patch, short_patch], fontsize=10)

    return _save(fig, "04_return_by_decile.png")


# ---------------------------------------------------------------------------
# 5. plot_cumulative_pnl
# ---------------------------------------------------------------------------

def plot_cumulative_pnl(
    backtest_df: pd.DataFrame,
    sector_neutral_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Cumulative long-short PnL chart with drawdown shading.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        Output of ``run_long_short_backtest``.
    sector_neutral_df : pd.DataFrame, optional
        Output of ``run_sector_neutral_backtest`` for comparison overlay.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    if backtest_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No backtest data", ha="center", va="center")
        return _save(fig, "05_cumulative_pnl.png")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    def _plot_cum_pnl(df: pd.DataFrame, ax: plt.Axes, label: str, color: str) -> None:
        cum = (1 + df["ls_ret"]).cumprod()
        ax.plot(range(len(cum)), cum.values, color=color, linewidth=2, label=label)
        # Drawdown shading
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        ax.fill_between(
            range(len(cum)),
            cum.values,
            rolling_max.values,
            alpha=0.15,
            color=color,
        )

    _plot_cum_pnl(backtest_df, ax1, "Unconstrained L/S", "#1f77b4")
    if sector_neutral_df is not None and not sector_neutral_df.empty:
        _plot_cum_pnl(sector_neutral_df, ax1, "Sector-Neutral L/S", "#ff7f0e")

    ax1.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("Cumulative Return (Base = 1.0)", fontsize=12)
    ax1.set_title(
        "Earnings Silence Alpha — Long/Short Cumulative PnL",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(fontsize=11)

    # Drawdown panel
    cum = (1 + backtest_df["ls_ret"]).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    ax2.fill_between(range(len(dd)), dd.values, 0, color="#d62728", alpha=0.6)
    ax2.set_ylabel("Drawdown", fontsize=11)
    ax2.set_xlabel("Quarter Index", fontsize=12)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))

    fig.tight_layout()
    return _save(fig, "05_cumulative_pnl.png")


# ---------------------------------------------------------------------------
# 6. plot_top_omitted_terms_wordcloud
# ---------------------------------------------------------------------------

def plot_top_omitted_terms_wordcloud(signal_df: pd.DataFrame) -> Path:
    """Word cloud of the most frequently omitted terms in HIGH_OMISSION companies.

    Aggregates ``top_omitted_terms`` lists from all HIGH_OMISSION rows and
    weights each term by its frequency of omission across the universe.
    This is the most intuitive chart for a non-quantitative audience — it
    answers: "What topics are management teams going quiet about?"

    Parameters
    ----------
    signal_df : pd.DataFrame
        Signal DataFrame with ``signal_label`` and ``top_omitted_terms``.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud not installed — skipping word cloud chart")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Install wordcloud: pip install wordcloud", ha="center", va="center")
        return _save(fig, "06_omitted_terms_wordcloud.png")

    high_df = signal_df[signal_df["signal_label"] == "HIGH_OMISSION"].copy()
    if high_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No HIGH_OMISSION rows", ha="center", va="center")
        return _save(fig, "06_omitted_terms_wordcloud.png")

    term_freq: dict = {}
    for _, row in high_df.iterrows():
        try:
            terms = json.loads(row["top_omitted_terms"])
        except (json.JSONDecodeError, TypeError):
            continue
        for term in terms:
            display_term = term.replace("_", " ")
            term_freq[display_term] = term_freq.get(display_term, 0) + 1

    if not term_freq:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No omitted terms found", ha="center", va="center")
        return _save(fig, "06_omitted_terms_wordcloud.png")

    wc = WordCloud(
        width=1400,
        height=700,
        background_color="white",
        colormap="RdBu_r",
        max_words=80,
        prefer_horizontal=0.8,
        collocations=False,
    ).generate_from_frequencies(term_freq)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "Most Frequently Omitted Topics — HIGH_OMISSION Companies\n"
        "(Term size ∝ frequency of omission across universe)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    return _save(fig, "06_omitted_terms_wordcloud.png")
