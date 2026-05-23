"""
transcript_scraper.py
---------------------
SEC EDGAR data acquisition layer for earnings call transcripts.

The primary data source is SEC EDGAR's full-text search API and the EDGAR
submissions endpoint (data.sec.gov/submissions/CIK{cik}.json).  All raw
transcript text is cached to disk so re-runs incur zero network cost.

Usage (CLI smoke test)::

    python -m src.transcript_scraper --cik 0000320193 --ticker AAPL

Notes
-----
EDGAR 8-K filings do not always contain the full earnings call transcript.
Many companies file the transcript as exhibit 99.1; others file only the
earnings press release.  The ``coverage_quality`` column downstream
distinguishes full transcripts (>2000 words), press releases (500-2000
words), and insufficient filings (<500 words).  Only "transcript" rows
should be used in signal generation.

EDGAR rate-limit compliance: SEC ToS forbids more than 10 requests/second.
This module sleeps REQUEST_DELAY_SECONDS between every network call and
implements exponential backoff on failures.
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quarter → date-range mapping helpers
# ---------------------------------------------------------------------------
_QUARTER_DATE_RANGES: dict[str, tuple[str, str]] = {
    "Q1": ("{year}-01-01", "{year}-04-15"),
    "Q2": ("{year}-04-01", "{year}-07-15"),
    "Q3": ("{year}-07-01", "{year}-10-15"),
    "Q4": ("{year}-10-01", "{year+1}-01-31"),
}


def _quarter_to_date_range(quarter: str, year: int) -> tuple[str, str]:
    """Convert a quarter string and year to a (start_date, end_date) tuple.

    Parameters
    ----------
    quarter : str
        One of "Q1", "Q2", "Q3", "Q4".
    year : int
        Calendar year (e.g. 2022).

    Returns
    -------
    tuple[str, str]
        ISO date strings (YYYY-MM-DD) bounding the earnings release window.
        The window is intentionally wider than the quarter end to capture
        lagged filings.
    """
    if quarter not in _QUARTER_DATE_RANGES:
        raise ValueError(f"quarter must be one of Q1/Q2/Q3/Q4, got: {quarter!r}")
    start_tmpl, end_tmpl = _QUARTER_DATE_RANGES[quarter]
    start = start_tmpl.format(year=year)
    if quarter == "Q4":
        end = end_tmpl.format(**{"year+1": year + 1})
    else:
        end = end_tmpl.format(year=year)
    return start, end


def _zero_pad_cik(cik: str) -> str:
    """Left-pad CIK to 10 digits as required by EDGAR."""
    return str(cik).zfill(10)


def _rate_limit() -> None:
    """Sleep REQUEST_DELAY_SECONDS to respect EDGAR ToS (≤10 req/s)."""
    time.sleep(config.REQUEST_DELAY_SECONDS)


def _get_with_backoff(url: str, **kwargs) -> requests.Response:
    """HTTP GET with exponential backoff on network failures.

    Parameters
    ----------
    url : str
        Target URL.
    **kwargs
        Forwarded to ``requests.get``.

    Returns
    -------
    requests.Response

    Raises
    ------
    requests.HTTPError
        If all retries are exhausted.
    """
    headers = kwargs.pop(
        "headers",
        {
            "User-Agent": "EarningsSilenceAlpha/0.1 research@example.com",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json, text/html, */*",
        },
    )
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            _rate_limit()
            resp = requests.get(url, headers=headers, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == config.MAX_RETRIES:
                logger.error("All retries exhausted for %s: %s", url, exc)
                raise
            wait = config.BACKOFF_BASE ** (attempt + 1)
            logger.warning(
                "Request failed (attempt %d/%d) for %s: %s — retrying in %.0fs",
                attempt + 1,
                config.MAX_RETRIES,
                url,
                exc,
                wait,
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# 1. search_edgar_8k
# ---------------------------------------------------------------------------

def search_edgar_8k(cik: str, quarter: str, year: int) -> list[dict]:
    """Search EDGAR for 8-K filings for a given CIK in a given quarter/year.

    Uses the EDGAR submissions JSON endpoint (data.sec.gov) which returns the
    full filing history for a company.  Filters to 8-K filings whose filing
    date falls within the quarter window.

    Parameters
    ----------
    cik : str
        SEC CIK identifier (with or without leading zeros).
    quarter : str
        One of "Q1", "Q2", "Q3", "Q4".
    year : int
        Calendar year.

    Returns
    -------
    list[dict]
        Each dict contains: ``accession_number``, ``filing_date``,
        ``document_url``.  Empty list if no filings found.
    """
    padded_cik = _zero_pad_cik(cik)
    start_date, end_date = _quarter_to_date_range(quarter, year)
    url = config.EDGAR_SUBMISSIONS_URL.format(cik=padded_cik)
    logger.info("Searching EDGAR 8-K filings: CIK=%s %s/%d", cik, quarter, year)

    try:
        resp = _get_with_backoff(url)
        data = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch EDGAR submissions for CIK %s: %s", cik, exc)
        return []

    filings = data.get("filings", {}).get("recent", {})
    if not filings:
        logger.warning("No filings data in EDGAR response for CIK %s", cik)
        return []

    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    results: list[dict] = []

    for form, date, accession in zip(forms, dates, accessions):
        if form not in ("8-K", "8-K/A"):
            continue
        if not (start_date <= date <= end_date):
            continue
        # Normalise accession number: remove dashes for URL construction
        acc_nodash = accession.replace("-", "")
        document_url = (
            f"{config.EDGAR_BASE_URL}/Archives/edgar/data/"
            f"{cik}/{acc_nodash}/{accession}-index.htm"
        )
        results.append(
            {
                "accession_number": accession,
                "filing_date": date,
                "document_url": document_url,
            }
        )

    logger.info("Found %d 8-K filing(s) for CIK %s in %s/%d", len(results), cik, quarter, year)
    return results


# ---------------------------------------------------------------------------
# 2. fetch_8k_document_index
# ---------------------------------------------------------------------------

def fetch_8k_document_index(accession_number: str, cik: str) -> list[dict]:
    """Fetch the filing index and identify the primary transcript document.

    Targets the EDGAR filing index JSON at::

        /Archives/edgar/data/{cik}/{accession_nodash}/index.json

    Prioritises documents by:
    1. Description contains "transcript" or "earnings" (case-insensitive)
    2. Exhibit type 99.1 (press release / supplemental data)
    3. Primary document (.htm or .txt extension)

    Parameters
    ----------
    accession_number : str
        EDGAR accession number (with dashes, e.g. ``0000320193-23-000077``).
    cik : str
        SEC CIK identifier.

    Returns
    -------
    list[dict]
        Each dict: ``filename``, ``document_type``, ``description``, ``url``.
    """
    acc_nodash = accession_number.replace("-", "")
    index_url = (
        f"{config.EDGAR_BASE_URL}/Archives/edgar/data/"
        f"{cik}/{acc_nodash}/index.json"
    )
    logger.debug("Fetching filing index: %s", index_url)

    try:
        resp = _get_with_backoff(index_url)
        data = resp.json()
    except Exception as exc:
        logger.error("Failed to fetch filing index %s: %s", index_url, exc)
        return []

    documents = data.get("directory", {}).get("item", [])
    results: list[dict] = []

    for doc in documents:
        filename = doc.get("name", "")
        doc_type = doc.get("type", "")
        description = doc.get("name", "")  # EDGAR index.json uses name field

        ext = Path(filename).suffix.lower()
        if ext not in (".htm", ".html", ".txt"):
            continue

        url = (
            f"{config.EDGAR_BASE_URL}/Archives/edgar/data/"
            f"{cik}/{acc_nodash}/{filename}"
        )
        results.append(
            {
                "filename": filename,
                "document_type": doc_type,
                "description": description,
                "url": url,
            }
        )

    # Prioritise by relevance
    def _priority(doc: dict) -> int:
        desc_lower = doc["description"].lower()
        fname_lower = doc["filename"].lower()
        if "transcript" in desc_lower or "transcript" in fname_lower:
            return 0
        if "earnings" in desc_lower:
            return 1
        if "ex99" in fname_lower or "ex-99" in fname_lower:
            return 2
        return 3

    results.sort(key=_priority)
    return results


# ---------------------------------------------------------------------------
# 3. fetch_transcript_text
# ---------------------------------------------------------------------------

_SEC_HEADER_PATTERN = re.compile(
    r"SECURITIES AND EXCHANGE COMMISSION|UNITED STATES\s+SECURITIES",
    re.IGNORECASE,
)
_CACHE_DIR = config.TRANSCRIPTS_DIR


def fetch_transcript_text(
    document_url: str,
    cache_key: str,
) -> str:
    """Fetch raw HTML from EDGAR and return clean plaintext.

    Caches the result to ``data/raw/transcripts/{cache_key}.txt``.
    On subsequent calls the cached file is returned directly (zero network
    cost).

    Parameters
    ----------
    document_url : str
        Full URL to the EDGAR HTML document.
    cache_key : str
        Unique filename stem for the cache file (e.g. ``"320193_Q1_2023"``).

    Returns
    -------
    str
        Clean plaintext with HTML tags, SEC boilerplate headers, and EDGAR
        injection artefacts removed.  Empty string on fetch failure.
    """
    cache_path = _CACHE_DIR / f"{cache_key}.txt"

    if cache_path.exists():
        logger.debug("Cache hit: %s", cache_path)
        return cache_path.read_text(encoding="utf-8", errors="replace")

    logger.info("Fetching document: %s", document_url)
    try:
        resp = _get_with_backoff(document_url)
        html = resp.text
    except Exception as exc:
        logger.error("Failed to fetch %s: %s", document_url, exc)
        return ""

    # Parse HTML
    soup = BeautifulSoup(html, "lxml")
    # Remove script / style noise
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    text = soup.get_text(separator="\n")

    # Strip SEC filing header (lines before the actual document body)
    lines = text.splitlines()
    body_start = 0
    for i, line in enumerate(lines):
        if _SEC_HEADER_PATTERN.search(line):
            # Skip forward until a blank line signals end of header block
            for j in range(i, min(i + 50, len(lines))):
                if lines[j].strip() == "":
                    body_start = j + 1
                    break
            break

    cleaned = "\n".join(lines[body_start:]).strip()

    # Persist to cache
    cache_path.write_text(cleaned, encoding="utf-8")
    logger.debug("Cached transcript to %s", cache_path)
    return cleaned


# ---------------------------------------------------------------------------
# 4. build_transcript_corpus
# ---------------------------------------------------------------------------

_COVERAGE_QUALITY_MAP = {
    "transcript": lambda wc: wc > config.TRANSCRIPT_WORD_THRESHOLD,
    "press_release": lambda wc: config.PRESS_RELEASE_WORD_THRESHOLD <= wc <= config.TRANSCRIPT_WORD_THRESHOLD,
}


def _classify_coverage(word_count: int) -> str:
    """Return coverage quality label based on word count."""
    if word_count > config.TRANSCRIPT_WORD_THRESHOLD:
        return "transcript"
    if word_count >= config.PRESS_RELEASE_WORD_THRESHOLD:
        return "press_release"
    return "insufficient"


def build_transcript_corpus(
    cik: str,
    ticker: str,
    company_name: str,
    start_year: int = 2018,
) -> pd.DataFrame:
    """Build a transcript corpus DataFrame for one company.

    Orchestrates the full acquisition pipeline::

        search_edgar_8k → fetch_8k_document_index → fetch_transcript_text

    Parameters
    ----------
    cik : str
        SEC CIK identifier.
    ticker : str
        Exchange ticker symbol (used for logging only).
    company_name : str
        Human-readable company name.
    start_year : int, optional
        Earliest year to collect (default 2018).

    Returns
    -------
    pd.DataFrame
        Columns: ``cik``, ``ticker``, ``company_name``, ``quarter``, ``year``,
        ``filing_date``, ``transcript_text``, ``word_count``,
        ``fetch_status``, ``coverage_quality``.

    Notes
    -----
    EDGAR 8-K filings do not always contain the full earnings call transcript.
    Many companies file the transcript as exhibit 99.1; others only file the
    press release.  The ``coverage_quality`` column signals usability:
    "transcript" (>2000 words), "press_release" (500-2000), or "insufficient"
    (<500).  Only "transcript" rows should feed downstream analysis.
    """
    import datetime

    current_year = datetime.date.today().year
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    rows: list[dict] = []

    for year in range(start_year, current_year + 1):
        for quarter in quarters:
            filings = search_edgar_8k(cik, quarter, year)
            if not filings:
                rows.append(
                    {
                        "cik": cik,
                        "ticker": ticker,
                        "company_name": company_name,
                        "quarter": quarter,
                        "year": year,
                        "filing_date": None,
                        "transcript_text": "",
                        "word_count": 0,
                        "fetch_status": "not_found",
                        "coverage_quality": "insufficient",
                    }
                )
                continue

            # Take the first (most relevant) filing per quarter
            filing = filings[0]
            docs = fetch_8k_document_index(filing["accession_number"], cik)

            if not docs:
                rows.append(
                    {
                        "cik": cik,
                        "ticker": ticker,
                        "company_name": company_name,
                        "quarter": quarter,
                        "year": year,
                        "filing_date": filing["filing_date"],
                        "transcript_text": "",
                        "word_count": 0,
                        "fetch_status": "not_found",
                        "coverage_quality": "insufficient",
                    }
                )
                continue

            # Attempt documents in priority order
            text = ""
            fetch_status = "not_found"
            for doc in docs:
                cache_key = f"{cik}_{quarter}_{year}"
                try:
                    text = fetch_transcript_text(doc["url"], cache_key)
                    if text.strip():
                        fetch_status = "success"
                        break
                except Exception as exc:
                    logger.warning("Parse error for %s: %s", doc["url"], exc)
                    fetch_status = "parse_error"

            word_count = len(text.split()) if text else 0
            rows.append(
                {
                    "cik": cik,
                    "ticker": ticker,
                    "company_name": company_name,
                    "quarter": quarter,
                    "year": year,
                    "filing_date": filing["filing_date"],
                    "transcript_text": text,
                    "word_count": word_count,
                    "fetch_status": fetch_status,
                    "coverage_quality": _classify_coverage(word_count),
                }
            )

    corpus_df = pd.DataFrame(rows)

    # Persist per-company corpus
    save_path = config.PROCESSED_DIR / f"{cik}_corpus.csv"
    corpus_df.to_csv(save_path, index=False)
    logger.info(
        "Built corpus for %s: %d transcript(s), %d parse error(s) — saved to %s",
        ticker,
        (corpus_df.fetch_status == "success").sum(),
        (corpus_df.fetch_status == "parse_error").sum(),
        save_path,
    )
    return corpus_df


# ---------------------------------------------------------------------------
# 5. build_universe_corpus
# ---------------------------------------------------------------------------

def build_universe_corpus(universe_df: pd.DataFrame) -> pd.DataFrame:
    """Build and save the master transcript corpus for the entire universe.

    Parameters
    ----------
    universe_df : pd.DataFrame
        The company universe (columns: ``ticker``, ``company_name``,
        ``sector``, ``cik``).

    Returns
    -------
    pd.DataFrame
        Concatenated corpus for all companies, saved to
        ``data/processed/universe_corpus.csv``.
    """
    all_corpora: list[pd.DataFrame] = []

    for _, row in universe_df.iterrows():
        cik = str(row["cik"])
        ticker = row["ticker"]
        company_name = row["company_name"]
        logger.info("Processing universe company: %s (%s)", ticker, cik)

        try:
            company_corpus = build_transcript_corpus(cik, ticker, company_name)
            n_transcripts = (company_corpus.fetch_status == "success").sum()
            n_errors = (company_corpus.fetch_status == "parse_error").sum()
            logger.info(
                "Built corpus for %s: %d transcripts found, %d parse errors",
                ticker,
                n_transcripts,
                n_errors,
            )
            all_corpora.append(company_corpus)
        except Exception as exc:
            logger.error("Failed to build corpus for %s: %s", ticker, exc)

    if not all_corpora:
        logger.warning("No corpora built — returning empty DataFrame")
        return pd.DataFrame()

    universe_corpus = pd.concat(all_corpora, ignore_index=True)
    universe_corpus.to_csv(config.UNIVERSE_CORPUS_PATH, index=False)
    logger.info(
        "Universe corpus saved: %d rows, %d companies — %s",
        len(universe_corpus),
        universe_corpus["ticker"].nunique(),
        config.UNIVERSE_CORPUS_PATH,
    )
    return universe_corpus


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test EDGAR scraper for a single company."
    )
    parser.add_argument("--cik", required=True, help="SEC CIK (e.g. 0000320193)")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--company", default="", help="Company name (optional)")
    parser.add_argument(
        "--start-year", type=int, default=2020, help="Start year (default 2020)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logger.info(
        "Running scraper smoke test — CIK: %s, ticker: %s", args.cik, args.ticker
    )
    df = build_transcript_corpus(
        cik=args.cik,
        ticker=args.ticker,
        company_name=args.company or args.ticker,
        start_year=args.start_year,
    )
    print("\n── Corpus Sample ──────────────────────────────────────────")
    print(df[["ticker", "quarter", "year", "word_count", "coverage_quality"]].to_string())
    print(f"\nTotal rows: {len(df)}")
    print(f"Transcripts (>2000 words): {(df.coverage_quality == 'transcript').sum()}")
