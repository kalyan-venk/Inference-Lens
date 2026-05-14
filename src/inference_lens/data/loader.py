"""
Data loading for Inference-Lens.

Two sources live here:
  HH-RLHF  -- 170K+ human preference pairs from Anthropic (HuggingFace)
  LLM-Bar  -- 419 adversarial evaluator stress-test pairs (EMNLP 2023)

HH-RLHF is used for training, EDA, and clustering.
LLM-Bar is held out completely and used only for the stress-test phase.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

HH_RLHF_SUBSETS = ("helpful-base", "harmless-base")
LLM_BAR_CATEGORIES = ("neighbor", "natural", "gpt4_generated", "manual")


def extract_assistant_response(conversation: str) -> str:
    """Pull out the final assistant turn from a full HH-RLHF conversation string.

    HH-RLHF stores the entire conversation in each field, formatted like:
        \\n\\nHuman: <question>\\n\\nAssistant: <response>

    Multi-turn conversations follow the same pattern, just repeated. We want
    the last assistant response, since that's what the human was actually judging.
    Returns the full conversation string unchanged if no assistant turn is found.
    """
    marker = "\n\nAssistant:"
    idx = conversation.rfind(marker)
    if idx == -1:
        logger.warning("No assistant turn found in conversation, returning full text.")
        return conversation
    return conversation[idx + len(marker):].strip()


def load_hh_rlhf(
    cache_dir: str | Path = "data/raw",
    subsets: tuple[str, ...] = HH_RLHF_SUBSETS,
) -> dict:
    """Stream HH-RLHF from HuggingFace and cache locally.

    Returns a dict keyed by subset name. Each value is a HuggingFace DatasetDict
    with train and test splits as provided by the source.
    Call flatten_hh_rlhf() to convert to a usable DataFrame.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading HH-RLHF subsets: %s", subsets)
    datasets = {}
    for subset in subsets:
        datasets[subset] = load_dataset(
            "anthropic/hh-rlhf",
            data_dir=subset,
            cache_dir=str(cache_dir / "hh_rlhf"),
        )
        logger.info("Loaded subset '%s': %s", subset, datasets[subset])
    return datasets


def flatten_hh_rlhf(datasets: dict) -> pd.DataFrame:
    """Flatten HH-RLHF DatasetDicts into a single DataFrame.

    Each row is one preference comparison. Columns:
      chosen_response    the final assistant turn from the preferred conversation
      rejected_response  the final assistant turn from the rejected conversation
      chosen_full        full conversation text (kept for context if needed)
      rejected_full      full conversation text (kept for context if needed)
      subset             helpful-base or harmless-base
      original_split     train or test (as labeled by the source dataset)
    """
    rows = []
    for subset, ds in datasets.items():
        for split in ("train", "test"):
            for example in ds[split]:
                rows.append({
                    "chosen_response": extract_assistant_response(example["chosen"]),
                    "rejected_response": extract_assistant_response(example["rejected"]),
                    "chosen_full": example["chosen"],
                    "rejected_full": example["rejected"],
                    "subset": subset,
                    "original_split": split,
                })
    df = pd.DataFrame(rows)
    logger.info("Flattened HH-RLHF: %d preference pairs total", len(df))
    return df


def split_dataframe(
    df: pd.DataFrame,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Deterministic train/val/test split.

    Shuffles once with a fixed seed so every experiment sees the same data.
    Stratification is skipped here and verified separately in EDA.
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Splits must sum to 1.0"

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train)
    n_val = int(n * val)

    splits = {
        "train": df.iloc[:n_train],
        "val": df.iloc[n_train : n_train + n_val],
        "test": df.iloc[n_train + n_val :],
    }
    for name, split in splits.items():
        logger.info("Split '%s': %d rows", name, len(split))
    return splits


def load_llm_bar(path: str | Path = "data/raw/llm_bar") -> pd.DataFrame:
    """Load LLM-Bar adversarial pairs from local path.

    LLM-Bar must be downloaded manually from https://github.com/llm-bar/LLMBar
    and placed at data/raw/llm_bar/. Expected files:
      neighbor.json, natural.json, gpt4_generated.json, manual.json

    Each file has a list of dicts with keys: input, output1, output2, label.
    Returns a single DataFrame with a 'category' column added.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"LLM-Bar not found at {path}. "
            "Download from https://github.com/llm-bar/LLMBar and place files at data/raw/llm_bar/"
        )

    rows = []
    for category in LLM_BAR_CATEGORIES:
        fp = path / f"{category}.json"
        if not fp.exists():
            logger.warning("LLM-Bar category file not found, skipping: %s", fp)
            continue
        with open(fp) as f:
            data = json.load(f)
        for item in data:
            rows.append({**item, "category": category})

    df = pd.DataFrame(rows)
    logger.info("Loaded LLM-Bar: %d pairs across %d categories", len(df), df["category"].nunique())
    return df
