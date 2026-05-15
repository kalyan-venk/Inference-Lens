"""
Inference-Lens Streamlit App

Run with:
    streamlit run src/inference_lens/app/main.py

Scores an LLM response pair using trained models and shows which one
the models prefer, how confident they are, and why.
"""
import sys
from pathlib import Path

# make sure src/ is on the path when running via streamlit
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import joblib
import numpy as np
import streamlit as st

from inference_lens.features.extractor import (
    token_length,
    type_token_ratio,
    flesch_score,
    compute_rouge_l,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Inference-Lens",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Model loading (cached so we don't reload on every interaction)
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


@st.cache_resource
def load_models():
    models = {}
    errors = []

    scaler_path = MODELS_DIR / "scaler.joblib"
    if scaler_path.exists():
        models["scaler"] = joblib.load(scaler_path)
    else:
        errors.append("scaler.joblib not found -- run notebook 01_logreg_xgboost first.")

    logreg_path = MODELS_DIR / "logreg.joblib"
    if logreg_path.exists():
        models["logreg"] = joblib.load(logreg_path)
    else:
        errors.append("logreg.joblib not found -- run notebook 01_logreg_xgboost first.")

    xgb_path = MODELS_DIR / "xgboost.joblib"
    if xgb_path.exists():
        models["xgboost"] = joblib.load(xgb_path)
    else:
        errors.append("xgboost.joblib not found -- run notebook 01_logreg_xgboost first.")

    deberta_path = MODELS_DIR / "deberta_checkpoint"
    if deberta_path.exists():
        try:
            from transformers import pipeline as hf_pipeline
            models["deberta"] = hf_pipeline(
                "text-classification",
                model=str(deberta_path),
                tokenizer=str(deberta_path),
                device=-1,
            )
        except Exception as e:
            errors.append(f"DeBERTa failed to load: {e}")
    else:
        errors.append("DeBERTa checkpoint not found -- download from Colab and place at models/deberta_checkpoint/.")

    return models, errors


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

def extract_features(text: str, reference: str) -> np.ndarray:
    """Extract the same feature vector used during training."""
    rouge = compute_rouge_l([text], [reference])[0]
    return np.array([[
        token_length(text),
        type_token_ratio(text),
        flesch_score(text),
        rouge,
    ]])


def score_response(text: str, reference: str, models: dict) -> dict:
    """Score a single response with all available models. Returns dict of model -> prob."""
    features = extract_features(text, reference)
    scores = {}

    if "logreg" in models and "scaler" in models:
        scaled = models["scaler"].transform(features)
        scores["Logistic Regression"] = float(models["logreg"].predict_proba(scaled)[0, 1])

    if "xgboost" in models:
        scores["XGBoost"] = float(models["xgboost"].predict_proba(features)[0, 1])

    if "deberta" in models:
        result = models["deberta"](text[:512], truncation=True)[0]
        prob = result["score"] if result["label"] == "LABEL_1" else 1 - result["score"]
        scores["DeBERTa-v3-small"] = float(prob)

    return scores


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Inference-Lens")
st.caption("LLM output quality scoring with evaluator reliability analysis")

st.markdown(
    "Paste two LLM responses below. The scoring models will tell you which one "
    "they think is higher quality and how confident they are. "
    "This is the same pipeline used in the evaluator reliability stress-test."
)

# load models once
models, load_errors = load_models()

if load_errors:
    with st.expander("Model loading warnings", expanded=len(models) == 0):
        for err in load_errors:
            st.warning(err)

if not models or "scaler" not in models:
    st.error("No models loaded. Run the training notebooks first, then restart the app.")
    st.stop()

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

st.divider()
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Response A")
    response_a = st.text_area(
        "Paste response A here",
        height=250,
        placeholder="Paste the first LLM response here...",
        label_visibility="collapsed",
    )

with col_b:
    st.subheader("Response B")
    response_b = st.text_area(
        "Paste response B here",
        height=250,
        placeholder="Paste the second LLM response here...",
        label_visibility="collapsed",
    )

score_button = st.button("Score responses", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Scoring and results
# ---------------------------------------------------------------------------

if score_button:
    if not response_a.strip() or not response_b.strip():
        st.error("Both response fields need to be filled in.")
        st.stop()

    with st.spinner("Scoring..."):
        scores_a = score_response(response_a, response_b, models)
        scores_b = score_response(response_b, response_a, models)

    st.divider()
    st.subheader("Results")

    model_names = list(scores_a.keys())
    verdict_cols = st.columns(len(model_names))

    for col, model_name in zip(verdict_cols, model_names):
        prob_a = scores_a[model_name]
        prob_b = scores_b[model_name]
        winner = "A" if prob_a > prob_b else "B"
        margin = abs(prob_a - prob_b)
        confidence = "High" if margin > 0.3 else "Medium" if margin > 0.1 else "Low"
        conf_color = "green" if confidence == "High" else "orange" if confidence == "Medium" else "red"

        with col:
            st.metric(
                label=model_name,
                value=f"Response {winner} wins",
                delta=f"{confidence} confidence ({margin:.2f} margin)",
            )
            st.progress(prob_a, text=f"A: {prob_a:.3f}")
            st.progress(prob_b, text=f"B: {prob_b:.3f}")

    # ---------------------------------------------------------------------------
    # Feature breakdown
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("Feature breakdown")
    st.caption("The scalar signals each model uses to form its score.")

    feature_names = ["Token length", "Type-token ratio", "Flesch score", "ROUGE-L (vs other)"]
    feat_a = extract_features(response_a, response_b)[0]
    feat_b = extract_features(response_b, response_a)[0]

    feat_df_data = {
        "Feature": feature_names,
        "Response A": [round(float(v), 3) for v in feat_a],
        "Response B": [round(float(v), 3) for v in feat_b],
    }

    import pandas as pd
    feat_df = pd.DataFrame(feat_df_data).set_index("Feature")
    st.dataframe(feat_df, use_container_width=True)

    # ---------------------------------------------------------------------------
    # Aggregate verdict
    # ---------------------------------------------------------------------------
    st.divider()
    votes_a = sum(1 for model in model_names if scores_a[model] > scores_b[model])
    votes_b = len(model_names) - votes_a

    if votes_a > votes_b:
        st.success(f"Aggregate verdict: **Response A** ({votes_a}/{len(model_names)} models agree)")
    elif votes_b > votes_a:
        st.success(f"Aggregate verdict: **Response B** ({votes_b}/{len(model_names)} models agree)")
    else:
        st.info("Models are split. No clear winner.")

    st.caption(
        "Disagreement between models is informative -- it often signals the kind of "
        "ambiguous quality that makes automated evaluation unreliable."
    )
