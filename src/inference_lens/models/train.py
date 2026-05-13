"""
Supervised model training and evaluation for Inference-Lens.

Three model families are trained on engineered features to predict
human preference (chosen=1, rejected=0):

  Logistic Regression   interpretable L2-regularized baseline
  XGBoost               gradient-boosted ensemble with grid search
  DeBERTa-v3-small      fine-tuned transformer (trained on Colab, loaded here)

All runs are tracked in MLflow. Each model is evaluated on:
  AUC-ROC
  F1 macro
  Precision-recall curve
  Calibration plot (reliability diagram)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict[str, float]:
    """Compute standard evaluation metrics and log to active MLflow run."""
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_binary": f1_score(y_true, y_pred, average="binary"),
    }
    mlflow.log_metrics(metrics)
    logger.info(
        "%s  AUC=%.4f  F1_macro=%.4f",
        model_name, metrics["auc_roc"], metrics["f1_macro"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    c_values: list[float] = [0.01, 0.1, 1.0, 10.0],
    cv_folds: int = 5,
) -> Any:
    """Train L2 logistic regression with cross-validated C selection."""
    from sklearn.linear_model import LogisticRegressionCV

    with mlflow.start_run(run_name="logreg", nested=True):
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("Cs", c_values)
        mlflow.log_param("cv_folds", cv_folds)

        model = LogisticRegressionCV(
            Cs=c_values,
            cv=cv_folds,
            scoring="roc_auc",
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        mlflow.log_param("best_C", model.C_[0])

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        evaluate_model(y_val, y_pred, y_prob, "LogReg")

        mlflow.sklearn.log_model(model, "logreg_model")
    return model


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: dict | None = None,
    cv_folds: int = 5,
) -> Any:
    """Train XGBoost with 5-fold cross-validated grid search."""
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier

    if param_grid is None:
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 300],
            "subsample": [0.8, 1.0],
        }

    with mlflow.start_run(run_name="xgboost", nested=True):
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("param_grid", str(param_grid))
        mlflow.log_param("cv_folds", cv_folds)

        base = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        search = GridSearchCV(
            base,
            param_grid,
            cv=cv_folds,
            scoring="roc_auc",
            verbose=1,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        mlflow.log_params(search.best_params_)
        y_pred = best.predict(X_val)
        y_prob = best.predict_proba(X_val)[:, 1]
        evaluate_model(y_val, y_pred, y_prob, "XGBoost")

        mlflow.xgboost.log_model(best, "xgboost_model")
    return best


# ---------------------------------------------------------------------------
# DeBERTa (inference only — training happens on Colab)
# ---------------------------------------------------------------------------

def load_deberta_for_inference(checkpoint_path: str | Path) -> Any:
    """Load a fine-tuned DeBERTa checkpoint saved from Colab training.

    Training is done separately on Colab (see notebooks/03_models/deberta_colab.ipynb).
    This function loads the saved checkpoint for local inference and evaluation.
    """
    from transformers import pipeline

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"DeBERTa checkpoint not found at {checkpoint_path}. "
            "Run notebooks/03_models/deberta_colab.ipynb on Colab first."
        )

    logger.info("Loading DeBERTa checkpoint from %s", checkpoint_path)
    classifier = pipeline(
        "text-classification",
        model=str(checkpoint_path),
        tokenizer=str(checkpoint_path),
        device=-1,  # CPU inference
    )
    return classifier
