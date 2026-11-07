# Earnings Call Silence Mapping via Transcript Topic Omission

> **Academic Research** | Python · NLP · Quant Finance | Not Investment Advice

---

## Abstract

Earnings calls are one of the most information-dense events in the corporate calendar. The finance literature has largely focused on *what* management says — sentiment, readability, tone, and keyword frequency. This project takes the opposite approach: it maps **what management does not say** relative to their own historical baseline.

The behavioural finance motivation is straightforward: when a CEO who has discussed "supply chain optimisation" in every single quarterly call for two years suddenly omits the phrase entirely, the silence is more informative than any positive spin could be. Topic omissions in prepared remarks — the carefully scripted, legally reviewed portion of the call — are deliberate editorial choices. They are the dog that didn't bark. This repository implements a systematic, data-driven framework to detect and trade on those silences.

The methodological contribution is a company-specific, history-relative topic omission score built from TF-IDF representations of SEC 8-K filings. For each company, we construct a *standard vocabulary* — the top TF-IDF-weighted bigrams and trigrams over the trailing four quarters — and flag any term that drops below 20% of its historical average weight in the current quarter as an *omission event*. The omission score aggregates severity across all flagged terms, with additional weight for terms that appeared in every prior quarter (*core vocabulary*). We validate the signal against 30-day post-earnings stock returns across a 20-company universe using t-tests, Spearman rank IC, OLS regression with fixed effects, and a long-short backtest.

---

## Core Hypothesis

**H₀**: Topic omissions in earnings call prepared remarks carry no information about subsequent stock returns. The mean 30-day post-earnings return is equal for high-omission and low-omission companies.

**H₁**: Companies with high topic omission scores (relative to their own historical baseline) exhibit significantly lower 30-day post-earnings returns than companies with low omission scores. The market under-reacts to the silence signal, creating post-earnings drift.

---

## Key Innovation

**This is not sentiment analysis.**

| Approach | What It Measures |
|---|---|
| Sentiment NLP (Loughran-McDonald) | Positive / negative tone of language used |
| Topic modelling (LDA) | Latent themes present in the document |
| Readability scores (Fog, Flesch) | Linguistic complexity of disclosure |
| **Omission Mapping (this repo)** | **Structural *absence* of topics vs. company's own baseline** |

We do not score tone or language. We score the structural absence of topics relative to a company's own historical baseline. A company can say nothing negative and still have a high omission score — precisely because it said nothing. This is the key distinction.

---

## Methodology

### Pipeline

```
SEC EDGAR EDGAR 8-K Filings
        │
        ▼
BeautifulSoup HTML Parsing
(fetch_transcript_text)
        │
        ▼
Prepared Remarks Extraction
(split_transcript_sections)
 ─ CEO/CFO scripted section only ─
        │
        ▼
TF-IDF Corpus Construction
(build_tfidf_matrix)
 ─ fitted on full universe ─
        │
        ▼
Standard Vocabulary Building
(build_standard_vocabulary)
 ─ trailing 4-quarter average ─
 ─ top 20 bigrams/trigrams ─
        │
        ▼
Omission Detection
(detect_omissions)
 ─ presence_ratio < 0.20 → event ─
        │
        ▼
Omission Score Aggregation
(compute_omission_score)
 ─ severity-weighted sum ─
        │
        ▼
Cross-Sectional Decile Ranking
(classify_omission_regime)
 ─ HIGH/NEUTRAL/LOW per quarter ─
        │
        ▼
Return Attribution + Backtest
(signal.py, backtest.py)
 ─ 30-day post-earnings window ─
```

### Omission Score Formula

For a given company-quarter, let $V$ be the standard vocabulary of $N$ terms. For each term $i \in V$:

$$\text{presence\_ratio}_i = \frac{\text{TF-IDF}_{i,t}}{\overline{\text{TF-IDF}}_{i,t-1:t-4}}$$

$$\text{omission\_event}_i = \mathbf{1}\left[\text{presence\_ratio}_i < \theta\right] \quad \theta = 0.20$$

$$\text{Omission Score} = \frac{1}{N} \sum_{i \in V} \text{Severity}_i \cdot \mathbf{1}[\text{omission\_event}_i]$$

### Omission Severity Formula

$$\text{Severity}_i = (1 - \text{presence\_ratio}_i) \times \overline{\text{TF-IDF}}_{i} \times \underbrace{m_i}_{\text{core vocab multiplier}}$$

where:
$$m_i = \begin{cases} 1.5 & \text{if term } i \text{ appeared in all 4 trailing quarters (core vocab)} \\ 1.0 & \text{otherwise} \end{cases}$$

**Intuition**: the severity score rewards three dimensions simultaneously — how completely absent the term is (1 − presence\_ratio), how important the term was historically (avg TF-IDF weight), and whether it was a regular fixture of the company's communication (core vocab multiplier).

---

## Repository Structure

```
earnings-silence-alpha/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py                          ← all paths, hyperparameters, endpoints
│
├── data/
│   ├── raw/
│   │   └── transcripts/               ← cached .txt files (gitignored)
│   ├── processed/                     ← CSV/pkl artefacts (gitignored)
│   └── reference/
│       ├── company_universe.csv       ← 20-company S&P 500 universe
│       └── stopwords_finance.txt      ← 80+ earnings-call stopwords
│
├── outputs/                           ← charts saved at 300 DPI (gitignored)
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb ← end-to-end walkthrough
│
└── src/
    ├── __init__.py
    ├── transcript_scraper.py          ← EDGAR data acquisition
    ├── text_processor.py              ← NLP cleaning and preprocessing
    ├── tfidf_engine.py                ← TF-IDF matrix + standard vocabulary
    ├── omission_detector.py           ← core signal generation (omission events)
    ├── signal.py                      ← return attribution + statistical tests
    ├── backtest.py                    ← long-short portfolio simulation
    └── visualize.py                   ← all publication-quality charts
```

---

## How to Run

### 1. Clone and set up environment

```bash
git clone https://github.com/rxj0102/earnings-silence-alpha.git
cd earnings-silence-alpha
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### 2. Smoke test scraper for one company

```bash
python -m src.transcript_scraper --cik 0000320193 --ticker AAPL --start-year 2020
```

### 3. Run the full pipeline

```python
import pandas as pd
from src.transcript_scraper import build_universe_corpus
from src.text_processor import preprocess_corpus
from src.tfidf_engine import build_tfidf_matrix, build_all_vocabularies
from src.omission_detector import run_omission_detection_universe, classify_omission_regime
from src.signal import fetch_post_earnings_returns, run_ttest_validation, compute_information_coefficient
from src.backtest import run_long_short_backtest, compute_performance_metrics
import config

# 1. Load universe
universe = pd.read_csv(config.COMPANY_UNIVERSE_PATH)

# 2. Acquire transcripts (rate-limited, will take ~15 min for full universe)
corpus = build_universe_corpus(universe)

# 3. Preprocess
corpus_pp = preprocess_corpus(corpus)

# 4. TF-IDF
tfidf_matrix, vectorizer, feature_names = build_tfidf_matrix(corpus_pp)

# 5. Standard vocabularies
vocabularies = build_all_vocabularies(corpus_pp, tfidf_matrix, feature_names)

# 6. Omission signals
signal_df = run_omission_detection_universe(corpus_pp, tfidf_matrix, feature_names, vocabularies)
signal_df = classify_omission_regime(signal_df)

# 7. Return attribution
signal_df = fetch_post_earnings_returns(signal_df)

# 8. Validation
run_ttest_validation(signal_df)
compute_information_coefficient(signal_df)

# 9. Backtest
bt = run_long_short_backtest(signal_df)
compute_performance_metrics(bt)
```

### 4. Explore interactively

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

---

## Key Findings

*[Placeholder — populate after running the full pipeline on live EDGAR data.]*

| Metric | Value |
|---|---|
| 30-Day IC (Spearman) | TBD |
| t-stat (HIGH vs LOW, 30d) | TBD |
| Annualised Sharpe (L/S) | TBD |
| Max Drawdown | TBD |
| Sector-Neutral Sharpe | TBD |

---

## Limitations

1. **EDGAR coverage gaps**: Not all 8-K filings contain full earnings call transcripts. Many companies file only the press release or a summary. The `coverage_quality` column filters to documents >2000 words, but coverage varies significantly by company and year.

2. **Section detection failure rate**: Splitting prepared remarks from Q&A relies on heuristic pattern matching. For filings without clear section headers, the full text is used — diluting the signal by including analyst questions that may re-introduce omitted topics.

3. **Look-ahead bias risk**: Standard vocabulary construction uses a strictly trailing window (t-1 through t-4). No future information enters the baseline. However, the TF-IDF vectoriser is fit on the *full* corpus including future quarters — a mild look-ahead bias in IDF weights. A production system would need rolling vectoriser refitting.

4. **Survivorship bias**: The 20-company universe consists of current S&P 500 constituents. Companies that were delisted, merged, or experienced financial distress are excluded — biasing the universe toward survivors with stable disclosure patterns.

5. **Multiple comparison problem**: Testing three return horizons (1d, 5d, 30d) inflates the Type I error rate. Apply Bonferroni correction (α/3 ≈ 0.017) when assessing p-values. The 30-day horizon is the pre-specified primary test; the others are exploratory.

---

## Academic References

- Larcker, D. F., & Zakolyukina, A. A. (2012). Detecting Deceptive Discussions in Conference Calls. *Journal of Accounting Research*, 50(2), 495–540.
- Lee, Y. J. (2016). Talking About Good News: CEO Communication in Earnings Conference Calls. *The Accounting Review*, 91(3), 871–896.
- Tetlock, P. C., Saar-Tsechansky, M., & Macskassy, S. (2008). More Than Words: Quantifying Language to Measure Firms' Fundamentals. *The Journal of Finance*, 63(3), 1437–1467.
- Li, F. (2008). Annual Report Readability, Current Earnings, and Earnings Persistence. *Journal of Accounting and Economics*, 45(2–3), 221–247.
- Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35–65.

---

## Disclaimer

This repository is academic research only. All code, analyses, and results are provided for educational and research purposes. Nothing in this repository constitutes investment advice. Past performance (whether backtested or real) does not guarantee future results. The author(s) have no fiduciary relationship with the reader.

## Known limitations

- **TF-IDF is fit on the whole corpus.** The per-company baseline vocabulary correctly uses only the trailing quarters, but the IDF weights come from a vectoriser fitted on every transcript, including ones after the signal date. Fitting IDF on an expanding window (or per quarter) removes this mild look-ahead.
