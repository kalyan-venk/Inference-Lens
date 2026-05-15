"""
MLflow experiment setup for Inference-Lens.

Run this once before starting any training notebooks to make sure the
experiment exists and the tracking URI is configured correctly.

Usage:
    python scripts/setup_mlflow.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow

TRACKING_URI = str(Path(__file__).resolve().parents[1] / "mlruns")
EXPERIMENT_NAME = "inference-lens"


def setup():
    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        print(f"Created experiment '{EXPERIMENT_NAME}' (id: {experiment_id})")
    else:
        print(f"Experiment '{EXPERIMENT_NAME}' already exists (id: {experiment.experiment_id})")

    print(f"Tracking URI: {TRACKING_URI}")
    print()
    print("To view the MLflow UI, run:")
    print(f"    mlflow ui --backend-store-uri {TRACKING_URI}")
    print()
    print("Then open http://localhost:5000 in your browser.")


if __name__ == "__main__":
    setup()
