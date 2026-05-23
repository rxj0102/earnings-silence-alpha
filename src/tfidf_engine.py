"""
tfidf_engine.py
---------------
TF-IDF computation layer for building company-level standard vocabularies.

Design decisions
----------------
* The TF-IDF vectoriser is fitted on the **full corpus** (all companies, all
  quarters) so that IDF weights reflect global document frequency.  This means
  a term that is common across the entire universe gets a lower IDF weight —
  exactly what we want, since we care about *company-specific* language.
* ``sublinear_tf=True`` applies log(1 + tf) normalisation, damping the
  outsized influence of terms that appear hundreds of times per transcript.
* The n-gram analyser uses underscore-joined strings produced by
  ``text_processor.extract_ngrams`` (not sklearn's internal tokeniser) to
  leverage the finance-aware stopword removal already applied upstream.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. build_tfidf_matrix
# ---------------------------------------------------------------------------

def build_tfidf_matrix(
    corpus_df: pd.DataFrame,
    text_column: str = "prepared_remarks_text",
) -> tuple:
    """Fit a TF-IDF matrix on the full preprocessed corpus.

    The vectoriser is fit on the entire corpus (all companies, all quarters)
    so that IDF weights are globally calibrated.  Company-quarter vectors are
    then rows of the resulting sparse matrix.

    Parameters
    ----------
    corpus_df : pd.DataFrame
        Preprocessed corpus (output of ``preprocess_corpus``).  Must contain
        ``text_column`` and ``coverage_quality`` columns.
    text_column : str, optional
        Column containing the text to vectorise (default ``"prepared_remarks_text"``).

    Returns
    -------
    tuple
        * ``tfidf_matrix`` — scipy sparse array (n_docs × MAX_FEATURES)
        * ``vectorizer`` — fitted ``TfidfVectorizer`` instance
        * ``feature_names`` — list of vocabulary strings

    Notes
    -----
    The vectoriser is persisted to ``data/processed/tfidf_vectorizer.pkl`` for
    reproducibility.  Load it with ``pickle.load`` to transform new documents
    without refitting.
    """
    # Only use rows with full transcript coverage
    fit_df = corpus_df[corpus_df["coverage_quality"] == "transcript"].copy()
    if fit_df.empty:
        raise ValueError(
            "No transcript-quality rows in corpus_df. "
            "Run build_universe_corpus() and preprocess_corpus() first."
        )

    texts = fit_df[text_column].fillna("").tolist()
    logger.info(
        "Fitting TF-IDF on %d documents (column: %s)", len(texts), text_column
    )

    vectorizer = TfidfVectorizer(
        ngram_range=config.NGRAM_RANGE,
        max_features=config.MAX_FEATURES,
        min_df=config.MIN_DF,
        sublinear_tf=True,
        analyzer="word",
        token_pattern=r"[a-z][a-z_\-]{2,}",
        strip_accents="unicode",
        lowercase=True,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out().tolist()

    logger.info(
        "TF-IDF matrix shape: %s — vocabulary size: %d",
        tfidf_matrix.shape,
        len(feature_names),
    )

    # Persist vectoriser
    with open(config.TFIDF_VECTORIZER_PATH, "wb") as fh:
        pickle.dump(vectorizer, fh)
    logger.info("Vectoriser saved to %s", config.TFIDF_VECTORIZER_PATH)

    return tfidf_matrix, vectorizer, feature_names


# ---------------------------------------------------------------------------
# 2. get_company_quarter_vector
# ---------------------------------------------------------------------------

def get_company_quarter_vector(
    tfidf_matrix,
    corpus_df: pd.DataFrame,
    cik: str,
    quarter: str,
    year: int,
) -> Optional[np.ndarray]:
    """Extract the TF-IDF vector for a specific company-quarter.

    Parameters
    ----------
    tfidf_matrix : scipy sparse matrix
        Output of ``build_tfidf_matrix``.
    corpus_df : pd.DataFrame
        The **same** filtered DataFrame that was passed to ``build_tfidf_matrix``
        (transcript-quality rows only, same row order).
    cik : str
        Company CIK.
    quarter : str
        One of "Q1", "Q2", "Q3", "Q4".
    year : int
        Calendar year.

    Returns
    -------
    np.ndarray or None
        1-D array of length ``MAX_FEATURES``.  ``None`` if the company-quarter
        is not present in the matrix.
    """
    fit_df = corpus_df[corpus_df["coverage_quality"] == "transcript"].reset_index(drop=True)
    mask = (
        (fit_df["cik"].astype(str) == str(cik))
        & (fit_df["quarter"] == quarter)
        & (fit_df["year"].astype(int) == int(year))
    )
    matching = fit_df[mask]

    if matching.empty:
        logger.warning(
            "No TF-IDF vector found for CIK %s %s/%d", cik, quarter, year
        )
        return None

    row_idx = matching.index[0]
    vector = np.asarray(tfidf_matrix[row_idx].todense()).flatten()
    return vector


# ---------------------------------------------------------------------------
# 3. build_standard_vocabulary
# ---------------------------------------------------------------------------

def build_standard_vocabulary(
    cik: str,
    corpus_df: pd.DataFrame,
    tfidf_matrix,
    feature_names: list,
    trailing_quarters: int = config.TRAILING_AVG_WINDOW,
) -> Optional[pd.DataFrame]:
    """Construct the standard vocabulary for one company from trailing quarters.

    The standard vocabulary is a ranked list of the company's most
    characteristic n-grams averaged across the trailing window.  It represents
    the *baseline* language fingerprint against which future transcripts are
    compared.

    Parameters
    ----------
    cik : str
        Company CIK.
    corpus_df : pd.DataFrame
        Preprocessed corpus (transcript-quality rows, same order as matrix).
    tfidf_matrix : scipy sparse matrix
        Full corpus TF-IDF matrix.
    feature_names : list[str]
        Ordered vocabulary list from ``build_tfidf_matrix``.
    trailing_quarters : int, optional
        Number of historical quarters to average (default ``TRAILING_AVG_WINDOW = 4``).

    Returns
    -------
    pd.DataFrame or None
        Columns: ``term``, ``avg_tfidf_weight``, ``rank``, ``quarters_present``,
        ``core_vocab``.  ``None`` if insufficient history.

    Notes
    -----
    **Trailing window design prevents look-ahead bias**: the standard
    vocabulary for quarter *t* is built exclusively from quarters *t-1* through
    *t-trailing_quarters*.  The current quarter is never included in its own
    baseline.  This is crucial for signal validity.

    ``quarters_present`` counts how many of the trailing quarters had
    ``avg_tfidf > 0.01`` for a given term.  ``core_vocab=True`` flags terms
    that appeared in *every* trailing quarter — omitting these is particularly
    informative.
    """
    fit_df = corpus_df[corpus_df["coverage_quality"] == "transcript"].copy()
    company_df = fit_df[fit_df["cik"].astype(str) == str(cik)].copy()
    company_df = company_df.sort_values(["year", "quarter"]).reset_index(drop=True)

    if len(company_df) < trailing_quarters + 1:
        logger.debug(
            "Insufficient history for CIK %s: %d rows, need %d",
            cik,
            len(company_df),
            trailing_quarters + 1,
        )
        return None

    # Use the most recent `trailing_quarters` rows *excluding* the last (current)
    history = company_df.iloc[-(trailing_quarters + 1) : -1]

    # Build the corresponding original matrix row indices
    fit_df_reset = fit_df.reset_index(drop=True)
    history_vectors: list[np.ndarray] = []
    for _, row in history.iterrows():
        q_vec = get_company_quarter_vector(
            tfidf_matrix, corpus_df, cik, row["quarter"], int(row["year"])
        )
        if q_vec is not None:
            history_vectors.append(q_vec)

    if not history_vectors:
        return None

    stacked = np.vstack(history_vectors)           # shape: (n_quarters, n_features)
    avg_weights = stacked.mean(axis=0)              # element-wise average
    quarters_present = (stacked > 0.01).sum(axis=0)  # how many quarters term appeared

    # Build DataFrame ranked by average TF-IDF weight
    vocab_df = pd.DataFrame(
        {
            "term": feature_names,
            "avg_tfidf_weight": avg_weights,
            "quarters_present": quarters_present,
        }
    )
    vocab_df = vocab_df[vocab_df["avg_tfidf_weight"] > 0].copy()
    vocab_df = vocab_df.sort_values("avg_tfidf_weight", ascending=False).head(
        config.STANDARD_VOCAB_SIZE
    )
    vocab_df["rank"] = range(1, len(vocab_df) + 1)
    vocab_df["core_vocab"] = vocab_df["quarters_present"] == trailing_quarters
    vocab_df = vocab_df.reset_index(drop=True)

    return vocab_df


# ---------------------------------------------------------------------------
# 4. build_all_vocabularies
# ---------------------------------------------------------------------------

def build_all_vocabularies(
    corpus_df: pd.DataFrame,
    tfidf_matrix,
    feature_names: list,
) -> dict:
    """Build standard vocabularies for every company-quarter with enough history.

    Parameters
    ----------
    corpus_df : pd.DataFrame
        Preprocessed corpus.
    tfidf_matrix : scipy sparse matrix
        Full corpus TF-IDF matrix.
    feature_names : list[str]
        Vocabulary list.

    Returns
    -------
    dict
        Nested dict: ``{cik: {(quarter, year): vocabulary_df}}``.
        Saved to ``data/processed/standard_vocabularies.pkl``.
    """
    fit_df = corpus_df[corpus_df["coverage_quality"] == "transcript"].copy()
    ciks = fit_df["cik"].astype(str).unique()
    vocabularies: dict = {}

    for cik in ciks:
        company_df = fit_df[fit_df["cik"].astype(str) == cik].sort_values(
            ["year", "quarter"]
        )
        ticker = company_df["ticker"].iloc[0] if "ticker" in company_df.columns else cik

        if len(company_df) < config.MIN_QUARTERS_REQUIRED:
            logger.info(
                "Skipping %s (%s): only %d quarters of data (need %d)",
                ticker,
                cik,
                len(company_df),
                config.MIN_QUARTERS_REQUIRED,
            )
            continue

        cik_vocabs: dict = {}
        # For each quarter (starting from MIN_QUARTERS_REQUIRED-th), build vocabulary
        rows = company_df.reset_index(drop=True)
        for i in range(config.TRAILING_AVG_WINDOW, len(rows)):
            current_row = rows.iloc[i]
            quarter = current_row["quarter"]
            year = int(current_row["year"])
            vocab_df = build_standard_vocabulary(
                cik, corpus_df, tfidf_matrix, feature_names
            )
            if vocab_df is not None:
                cik_vocabs[(quarter, year)] = vocab_df

        if cik_vocabs:
            vocabularies[cik] = cik_vocabs
            logger.info(
                "Built %d vocabularies for %s (%s)", len(cik_vocabs), ticker, cik
            )

    # Persist
    with open(config.STANDARD_VOCABULARIES_PATH, "wb") as fh:
        pickle.dump(vocabularies, fh)
    logger.info(
        "Standard vocabularies saved: %d companies — %s",
        len(vocabularies),
        config.STANDARD_VOCABULARIES_PATH,
    )
    return vocabularies
