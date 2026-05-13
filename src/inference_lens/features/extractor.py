"""
Feature extraction for Inference-Lens.

Computes a suite of quality-signal features for each LLM response.
Features are computed once and cached to data/processed/features.parquet
so every downstream experiment loads in seconds instead of recomputing.

Feature groups
--------------
Lexical      token length, type-token ratio
Readability  Flesch reading ease score
Overlap      ROUGE-L F1 vs reference (chosen) response
Semantic     BERTScore F1, sentence embedding vector (all-MiniLM-L6-v2)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import textstat
from tqdm import tqdm

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FEATURES_CACHE = Path("data/processed/features.parquet")
EMBEDDINGS_CACHE = Path("data/processed/embeddings.npy")


# ---------------------------------------------------------------------------
# Lexical features (fast, CPU-only)
# ---------------------------------------------------------------------------

def token_length(text: str) -> int:
    """Number of whitespace-separated tokens."""
    return len(text.split())


def type_token_ratio(text: str) -> float:
    """Lexical diversity: unique tokens / total tokens."""
    tokens = text.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def flesch_score(text: str) -> float:
    """Flesch Reading Ease score. Higher = easier to read."""
    return textstat.flesch_reading_ease(text)


def compute_lexical_features(texts: list[str]) -> pd.DataFrame:
    """Compute token_length, type_token_ratio, flesch_score for each text."""
    records = []
    for text in tqdm(texts, desc="Lexical features"):
        records.append({
            "token_length": token_length(text),
            "type_token_ratio": type_token_ratio(text),
            "flesch_score": flesch_score(text),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Semantic features (slower, model-dependent)
# ---------------------------------------------------------------------------

def compute_embeddings(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts with a sentence transformer model.

    Returns a float32 ndarray of shape (len(texts), embedding_dim).
    Embeddings are L2-normalized so cosine similarity = dot product.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    logger.info("Encoding %d texts with %s (batch_size=%d)", len(texts), model_name, batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def compute_rouge_l(
    predictions: list[str],
    references: list[str],
) -> list[float]:
    """ROUGE-L F1 between each prediction and its reference."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in tqdm(
        zip(predictions, references),
        desc="ROUGE-L",
        total=len(predictions),
    ):
        result = scorer.score(ref, pred)
        scores.append(result["rougeL"].fmeasure)
    return scores


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    batch_size: int = 32,
    lang: str = "en",
) -> list[float]:
    """BERTScore F1 between each prediction and its reference.

    This is the slowest feature to compute. It is skipped by default
    in extract_all_features() and must be opted into explicitly.
    """
    from bert_score import score as bert_score_fn

    logger.info("Computing BERTScore for %d pairs", len(predictions))
    _, _, f1 = bert_score_fn(
        predictions,
        references,
        lang=lang,
        batch_size=batch_size,
        verbose=True,
    )
    return f1.tolist()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_all_features(
    df: pd.DataFrame,
    text_col: str = "chosen",
    reference_col: str = "chosen",
    include_bertscore: bool = False,
    embedding_model: str = EMBEDDING_MODEL,
    batch_size: int = 64,
    features_cache: Path = FEATURES_CACHE,
    embeddings_cache: Path = EMBEDDINGS_CACHE,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute all features and cache results to disk.

    Returns a tuple of:
      features_df   DataFrame of scalar features (one row per input row)
      embeddings    ndarray of shape (n, embedding_dim)

    On subsequent calls, cached files are loaded directly unless
    force_recompute=True.
    """
    features_cache = Path(features_cache)
    embeddings_cache = Path(embeddings_cache)

    if features_cache.exists() and embeddings_cache.exists() and not force_recompute:
        logger.info("Loading cached features from %s", features_cache)
        features_df = pd.read_parquet(features_cache)
        embeddings = np.load(embeddings_cache)
        return features_df, embeddings

    logger.info("Computing features for %d responses", len(df))
    texts = df[text_col].tolist()
    references = df[reference_col].tolist()

    features_df = compute_lexical_features(texts)
    features_df["rouge_l"] = compute_rouge_l(texts, references)

    if include_bertscore:
        features_df["bertscore_f1"] = compute_bertscore(texts, references)

    embeddings = compute_embeddings(texts, model_name=embedding_model, batch_size=batch_size)

    features_cache.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(features_cache, index=False)
    np.save(embeddings_cache, embeddings)
    logger.info("Cached features to %s and embeddings to %s", features_cache, embeddings_cache)

    return features_df, embeddings
