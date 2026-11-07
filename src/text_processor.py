"""
text_processor.py
-----------------
All text cleaning, section splitting, tokenisation and n-gram extraction
for earnings call transcripts.

Pipeline per document::

    raw_text
        → clean_transcript()          # lowercase, de-number, strip boilerplate
        → split_transcript_sections() # prepared_remarks | qa_section | full
        → tokenize_and_filter()       # NLTK word_tokenize + stopword removal
        → extract_ngrams()            # bigrams and trigrams as underscore strings

The ``prepared_remarks`` section is the *primary signal source*.  CEOs and
CFOs carefully edit and rehearse prepared remarks; they cannot control ad-hoc
Q&A responses to the same degree.  Topic omissions in prepared remarks are
therefore more deliberate and more informative than Q&A omissions.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy NLTK imports (downloaded on first use)
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    """Download required NLTK corpora if not already present."""
    import nltk

    for resource in ("punkt", "stopwords", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                logger.info("Downloading NLTK resource: %s", resource)
                nltk.download(resource, quiet=True)


# ---------------------------------------------------------------------------
# 1. load_finance_stopwords
# ---------------------------------------------------------------------------

def load_finance_stopwords() -> set:
    """Load and merge custom financial stopwords with NLTK English stopwords.

    Returns
    -------
    set
        Combined set of lowercase stopword strings.  Multi-word phrases are
        retained for phrase-level removal in ``clean_transcript``.

    Notes
    -----
    The custom stopwords file at ``data/reference/stopwords_finance.txt``
    contains earnings call boilerplate ("good morning", "next question"),
    legal filler ("forward looking statements", "safe harbor"), and high-
    frequency financial phrases ("per share", "basis points") that dominate
    TF-IDF vocabularies but carry no alpha.
    """
    _ensure_nltk_resources()
    from nltk.corpus import stopwords as nltk_stopwords

    # NLTK English base set
    combined: set = set(nltk_stopwords.words("english"))

    # Custom financial stopwords
    stopwords_path = config.STOPWORDS_PATH
    if not stopwords_path.exists():
        logger.warning("Stopwords file not found at %s — using NLTK only", stopwords_path)
        return combined

    with open(stopwords_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            combined.add(line.lower())

    logger.debug("Loaded %d stopwords total", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

# SEC filing header detection
_SEC_HEADER_RE = re.compile(
    r"UNITED STATES\s+SECURITIES AND EXCHANGE|SECURITIES AND EXCHANGE COMMISSION",
    re.IGNORECASE,
)

# Q&A section boundary markers
_QA_BOUNDARY_RE = re.compile(
    r"(?:QUESTION[\s\-]*AND[\s\-]*ANSWER|Q\s*&\s*A\s+SESSION"
    r"|QUESTIONS?\s+AND\s+ANSWERS?|QUESTION[\s\-]+PERIOD"
    r"|^\s*Operator\s*[:]\s*(?:We will|We are|At this time|Ladies))",
    re.IGNORECASE | re.MULTILINE,
)

# Standalone numbers (keep year-ranges and hyphenated terms)
_STANDALONE_DIGIT_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|x|X)?\b")

# Punctuation except hyphens within words and apostrophes
_PUNCT_RE = re.compile(r"[^\w\s\-']")

# Runs of whitespace
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# 2. clean_transcript
# ---------------------------------------------------------------------------

def clean_transcript(raw_text: str) -> str:
    """Normalise raw transcript text for downstream NLP.

    Steps applied in order:

    1. Detect and strip SEC filing boilerplate header.
    2. Remove Q&A section boundary separator lines.
    3. Lowercase.
    4. Remove standalone numbers / figures (preserves hyphenated terms).
    5. Remove punctuation except intra-word hyphens.
    6. Collapse whitespace.

    Parameters
    ----------
    raw_text : str
        Raw text as returned by ``fetch_transcript_text``.

    Returns
    -------
    str
        Cleaned, normalised plaintext.
    """
    if not raw_text:
        return ""

    text = raw_text

    # 1. Strip SEC header block
    match = _SEC_HEADER_RE.search(text)
    if match:
        # Find the first blank line after the header signature
        tail = text[match.start():]
        blank_pos = tail.find("\n\n")
        if blank_pos != -1:
            text = text[match.start() + blank_pos:].strip()

    # 2. Remove Q&A separator lines (keep content, just remove the header line)
    text = _QA_BOUNDARY_RE.sub("", text)

    # 3. Lowercase
    text = text.lower()

    # 4. Remove standalone digits / figures
    text = _STANDALONE_DIGIT_RE.sub(" ", text)

    # 5. Remove punctuation except intra-word hyphens and apostrophes
    text = _PUNCT_RE.sub(" ", text)

    # 6. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# 3. split_transcript_sections
# ---------------------------------------------------------------------------

_PREPARED_MARKERS = re.compile(
    r"(?:prepared\s+remarks?|presentation|management\s+discussion|"
    r"opening\s+remarks?|results\s+of\s+operations)",
    re.IGNORECASE,
)

_QA_OPEN_MARKERS = re.compile(
    r"(?:question[\s\-]and[\s\-]answer|q\s*&\s*a|"
    r"we\s+will\s+now\s+begin\s+the\s+question|"
    r"we\s+are\s+now\s+ready\s+for\s+(?:your\s+)?questions?|"
    r"^operator\s*:)",
    re.IGNORECASE | re.MULTILINE,
)


def split_transcript_sections(raw_text: str) -> dict:
    """Split an earnings call transcript into prepared remarks and Q&A.

    Detection heuristic: scan for Q&A boundary markers (operator cues, explicit
    "Question-and-Answer Session" headers).  Everything before the first marker
    is treated as prepared remarks; everything after is the Q&A.

    Parameters
    ----------
    raw_text : str
        Raw or lightly cleaned transcript text.

    Returns
    -------
    dict
        Keys: ``prepared_remarks``, ``qa_section``, ``full``.
        If detection fails, all three values are set to the full text and a
        warning is logged.

    Notes
    -----
    **Why prepared_remarks is the primary signal source**: CEOs and CFOs
    carefully draft and rehearse their prepared statements in advance —
    they are often approved by legal and IR teams.  Topic omissions in this
    section are deliberate choices.  Analysts' follow-up questions in the
    Q&A often force management to address topics they omitted, so Q&A may
    *dilute* the omission signal.  This module therefore weights the
    prepared_remarks section in downstream processing.
    """
    result = {"prepared_remarks": raw_text, "qa_section": "", "full": raw_text}

    if not raw_text:
        return result

    # Search for Q&A boundary
    match = _QA_OPEN_MARKERS.search(raw_text)
    if match:
        split_pos = match.start()
        # Require that prepared remarks block is non-trivially long
        if split_pos > 500:
            result["prepared_remarks"] = raw_text[:split_pos].strip()
            result["qa_section"] = raw_text[split_pos:].strip()
            logger.debug(
                "Section split succeeded at char %d (total %d chars)",
                split_pos,
                len(raw_text),
            )
            return result

    logger.warning(
        "Section split FAILED — no Q&A boundary detected (text length %d). "
        "Full text used for all sections.",
        len(raw_text),
    )
    return result


# ---------------------------------------------------------------------------
# 4. tokenize_and_filter
# ---------------------------------------------------------------------------

def tokenize_and_filter(text: str, stopwords: set) -> list:
    """Tokenise text and remove stopwords and short tokens.

    Parameters
    ----------
    text : str
        Cleaned plaintext.
    stopwords : set
        Set of lowercase stopword strings.

    Returns
    -------
    list[str]
        Filtered list of lowercase tokens (length ≥ 3, not in stopwords).
    """
    if not text:
        return []

    _ensure_nltk_resources()
    from nltk.tokenize import word_tokenize

    tokens = word_tokenize(text.lower())
    filtered = [
        tok
        for tok in tokens
        if len(tok) >= 3
        and tok not in stopwords
        and tok.replace("-", "").replace("'", "").isalpha()
    ]
    return filtered


# ---------------------------------------------------------------------------
# 5. extract_ngrams
# ---------------------------------------------------------------------------

def extract_ngrams(tokens: list, n_range: tuple = (2, 3)) -> list:
    """Generate bigrams and trigrams from a token list.

    Parameters
    ----------
    tokens : list[str]
        Filtered token list from ``tokenize_and_filter``.
    n_range : tuple, optional
        (min_n, max_n) — default ``(2, 3)`` produces bigrams and trigrams.

    Returns
    -------
    list[str]
        Flat list of n-gram strings joined by underscores.
        E.g. ``["supply_chain_disruption", "gross_margin", ...]``.
    """
    ngrams: list = []
    min_n, max_n = n_range
    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            gram = "_".join(tokens[i : i + n])
            ngrams.append(gram)
    return ngrams


# ---------------------------------------------------------------------------
# 6. preprocess_corpus
# ---------------------------------------------------------------------------

def preprocess_corpus(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Apply full NLP pipeline to every row of the transcript corpus.

    Only processes rows where ``coverage_quality == "transcript"`` (>2000 words).
    All other rows are passed through with empty NLP columns.

    Parameters
    ----------
    corpus_df : pd.DataFrame
        Output of ``build_universe_corpus`` (or a per-company corpus).

    Returns
    -------
    pd.DataFrame
        Original DataFrame enriched with columns:
        ``cleaned_text``, ``prepared_remarks_text``,
        ``qa_text``, ``ngrams_list`` (JSON-serialised list of ngram strings).

    Notes
    -----
    Processing is logged with per-company timing to aid performance profiling.
    """
    stopwords = load_finance_stopwords()
    corpus_df = corpus_df.copy()

    # Initialise new columns
    for col in ["cleaned_text", "prepared_remarks_text", "qa_text", "ngrams_list"]:
        corpus_df[col] = ""

    companies = corpus_df["ticker"].unique()

    for ticker in companies:
        mask = (corpus_df["ticker"] == ticker) & (
            corpus_df["coverage_quality"] == "transcript"
        )
        if mask.sum() == 0:
            logger.info("Skipping %s — no full transcripts available", ticker)
            continue

        t_start = time.time()
        indices = corpus_df[mask].index

        for idx in indices:
            raw = corpus_df.at[idx, "transcript_text"]

            # Step 1: clean
            cleaned = clean_transcript(raw)

            # Step 2: split sections
            sections = split_transcript_sections(raw)
            prepared_clean = clean_transcript(sections["prepared_remarks"])

            # Step 3 + 4: tokenise and extract ngrams (from prepared remarks)
            tokens = tokenize_and_filter(prepared_clean, stopwords)
            ngrams = extract_ngrams(tokens, n_range=config.NGRAM_RANGE)

            corpus_df.at[idx, "cleaned_text"] = cleaned
            corpus_df.at[idx, "prepared_remarks_text"] = prepared_clean
            corpus_df.at[idx, "qa_text"] = clean_transcript(sections["qa_section"])
            corpus_df.at[idx, "ngrams_list"] = json.dumps(ngrams)

        elapsed = time.time() - t_start
        logger.info(
            "Preprocessed %s: %d transcript(s) in %.1fs", ticker, mask.sum(), elapsed
        )

    corpus_df.to_csv(config.CORPUS_PREPROCESSED_PATH, index=False)
    logger.info("Preprocessed corpus saved to %s", config.CORPUS_PREPROCESSED_PATH)
    return corpus_df
