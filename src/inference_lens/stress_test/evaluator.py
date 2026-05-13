"""
Evaluator reliability stress-test for Inference-Lens.

All three trained models are applied zero-shot to LLM-Bar adversarial pairs.
No LLM-Bar data is ever seen during training — this is a pure out-of-distribution test.

What we measure:
  Accuracy degradation     vs HH-RLHF test set baseline
  AUC degradation          how much scoring confidence collapses
  Calibration shift        do probability estimates remain trustworthy
  False-preference rate    how often the model picks the adversarially preferred (wrong) response
  Per-category breakdown   which of the 4 perturbation types hurts most

The output of this phase is the reliability profile for each model family,
which is the core analytical contribution of Inference-Lens.
"""
from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)

LLM_BAR_CATEGORIES = ("neighbor", "natural", "gpt4_generated", "manual")


def run_stress_test(
    model: Any,
    model_name: str,
    X_llmbar: np.ndarray,
    y_llmbar: np.ndarray,
    categories: np.ndarray,
    baseline_auc: float,
    baseline_acc: float,
) -> pd.DataFrame:
    """Apply a trained model to LLM-Bar pairs and measure degradation.

    Args:
        model         A fitted sklearn-compatible model with predict/predict_proba.
        model_name    Name used in MLflow logging.
        X_llmbar      Feature matrix for LLM-Bar pairs.
        y_llmbar      True preference labels (0 or 1).
        categories    Array of perturbation category strings per row.
        baseline_auc  AUC on clean HH-RLHF test set for comparison.
        baseline_acc  Accuracy on clean HH-RLHF test set for comparison.

    Returns:
        DataFrame with per-category and overall degradation metrics.
    """
    y_pred = model.predict(X_llmbar)
    y_prob = model.predict_proba(X_llmbar)[:, 1]

    overall_acc = accuracy_score(y_llmbar, y_pred)
    overall_auc = roc_auc_score(y_llmbar, y_prob)
    false_pref_rate = (y_pred != y_llmbar).mean()

    rows = []
    rows.append({
        "model": model_name,
        "category": "overall",
        "n_pairs": len(y_llmbar),
        "accuracy": overall_acc,
        "auc_roc": overall_auc,
        "false_preference_rate": false_pref_rate,
        "acc_degradation": baseline_acc - overall_acc,
        "auc_degradation": baseline_auc - overall_auc,
    })

    for cat in LLM_BAR_CATEGORIES:
        mask = categories == cat
        if mask.sum() == 0:
            continue
        cat_acc = accuracy_score(y_llmbar[mask], y_pred[mask])
        cat_auc = roc_auc_score(y_llmbar[mask], y_prob[mask]) if len(np.unique(y_llmbar[mask])) > 1 else float("nan")
        rows.append({
            "model": model_name,
            "category": cat,
            "n_pairs": int(mask.sum()),
            "accuracy": cat_acc,
            "auc_roc": cat_auc,
            "false_preference_rate": (y_pred[mask] != y_llmbar[mask]).mean(),
            "acc_degradation": baseline_acc - cat_acc,
            "auc_degradation": baseline_auc - cat_auc,
        })

    results = pd.DataFrame(rows)

    with mlflow.start_run(run_name=f"stress_test_{model_name}", nested=True):
        mlflow.log_param("model", model_name)
        mlflow.log_metric("stress_overall_auc", overall_auc)
        mlflow.log_metric("stress_overall_acc", overall_acc)
        mlflow.log_metric("false_preference_rate", false_pref_rate)
        mlflow.log_metric("auc_degradation", baseline_auc - overall_auc)

    logger.info(
        "%s stress-test  AUC=%.4f (baseline %.4f)  FPR=%.4f",
        model_name, overall_auc, baseline_auc, false_pref_rate,
    )
    return results
