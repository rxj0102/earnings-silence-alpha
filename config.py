"""
config.py
---------
Central configuration for the Earnings Call Silence Mapping project.

All paths, hyperparameters, and API endpoints are defined here.
Import this module in all other source files to avoid hardcoded constants.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Reference file paths
# ---------------------------------------------------------------------------
COMPANY_UNIVERSE_PATH = REFERENCE_DIR / "company_universe.csv"
STOPWORDS_PATH = REFERENCE_DIR / "stopwords_finance.txt"

# ---------------------------------------------------------------------------
# Processed artefact paths
# ---------------------------------------------------------------------------
UNIVERSE_CORPUS_PATH = PROCESSED_DIR / "universe_corpus.csv"
CORPUS_PREPROCESSED_PATH = PROCESSED_DIR / "corpus_preprocessed.csv"
TFIDF_VECTORIZER_PATH = PROCESSED_DIR / "tfidf_vectorizer.pkl"
STANDARD_VOCABULARIES_PATH = PROCESSED_DIR / "standard_vocabularies.pkl"
OMISSION_SIGNALS_PATH = PROCESSED_DIR / "omission_signals.csv"

# ---------------------------------------------------------------------------
# SEC EDGAR endpoints
# ---------------------------------------------------------------------------
EDGAR_FULL_TEXT_SEARCH = (
    "https://efts.sec.gov/LATEST/search-index?q={query}"
    "&dateRange=custom&startdt={start}&enddt={end}&forms=8-K"
)
EDGAR_BASE_URL = "https://www.sec.gov"
EDGAR_SEARCH_URL = (
    "https://efts.sec.gov/LATEST/search-index?q=%22{cik}%22"
    "&dateRange=custom&startdt={start}&enddt={end}&forms=8-K"
)
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_FILING_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/index.json"
)

# ---------------------------------------------------------------------------
# Strategy hyperparameters
# ---------------------------------------------------------------------------
LOOKBACK_QUARTERS: int = 8          # transcript history per company
STANDARD_VOCAB_SIZE: int = 20       # top N ngrams forming company vocabulary
OMISSION_THRESHOLD: float = 0.20    # term drops below 20 % of trailing avg → omission
TRAILING_AVG_WINDOW: int = 4        # quarters used to build standard vocabulary
MIN_QUARTERS_REQUIRED: int = 6      # minimum history before signal is valid
POST_EARNINGS_WINDOW: int = 30      # days to measure drift after earnings

# ---------------------------------------------------------------------------
# NLP / TF-IDF parameters
# ---------------------------------------------------------------------------
NGRAM_RANGE: tuple = (2, 3)         # bigrams and trigrams only
MIN_DF: int = 2                     # minimum document frequency for TF-IDF
MAX_FEATURES: int = 500             # vocabulary cap for TF-IDF matrix

# ---------------------------------------------------------------------------
# Network / scraping parameters
# ---------------------------------------------------------------------------
REQUEST_DELAY_SECONDS: float = 1.5   # pause between EDGAR requests (SEC ToS)
MAX_RETRIES: int = 3                 # exponential backoff retries
BACKOFF_BASE: float = 2.0            # base seconds for exponential backoff
EDGAR_MAX_RESULTS_PER_PAGE: int = 10 # EDGAR pagination page size

# ---------------------------------------------------------------------------
# Coverage quality thresholds (word count)
# ---------------------------------------------------------------------------
TRANSCRIPT_WORD_THRESHOLD: int = 2000    # full transcript if > 2000 words
PRESS_RELEASE_WORD_THRESHOLD: int = 500  # press release if 500-2000 words
# < 500 words → "insufficient"

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------
HIGH_OMISSION_DECILES: tuple = (9, 10)   # SHORT book
LOW_OMISSION_DECILES: tuple = (1, 2)     # LONG book
EARNINGS_SEASON_MONTHS: tuple = (1, 4, 7, 10)  # quarterly rebalance months
BENCHMARK_TICKER: str = "SPY"

# ---------------------------------------------------------------------------
# Ensure output directories exist at import time
# ---------------------------------------------------------------------------
for _dir in [TRANSCRIPTS_DIR, PROCESSED_DIR, REFERENCE_DIR, OUTPUTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
