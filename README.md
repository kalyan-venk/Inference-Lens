# InferenceLens

**End-to-end LLM output quality scoring system with evaluator reliability stress-testing under adversarial conditions.**

> *Can we trust the systems we use to evaluate LLMs? InferenceLens finds out.*

---

## Overview

As LLM-powered applications proliferate, engineering teams depend on automated evaluation pipelines to judge whether model outputs are helpful, accurate, and safe. But most evaluation frameworks assume the evaluator itself is reliable — a gap that InferenceLens directly investigates.

InferenceLens is a research-grade ML system that:
1. **Scores LLM output quality** using human preference annotations and a suite of trained ML models
2. **Stress-tests evaluator reliability** by exposing scoring models to adversarially constructed inputs designed to fool automated judges
3. **Discovers latent response quality archetypes** via unsupervised clustering of 170K+ human preference pairs
4. **Deploys a real-time scoring interface** via Streamlit for interactive LLM output evaluation

This project directly extends prior work on [Multi-Agent Inference Reliability](https://kalyan-venk.github.io/agentic-llmops/), where stronger critic models were found to amplify damage in pipelines with unreliable downstream agents. InferenceLens applies the same principle to evaluation: if the judge model is miscalibrated or adversarially deceived, how much does scoring degrade — and which ML architectures are most robust?

---

## Research Question

> **How does evaluator unreliability propagate into automated LLM quality scoring pipelines, and can it be detected or corrected?**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      InferenceLens                       │
├──────────────┬──────────────────┬───────────────────────┤
│   Data Layer │  Modeling Layer  │    Evaluation Layer   │
│              │                  │                        │
│  HH-RLHF     │  Logistic Reg.   │  Standard Benchmark   │
│  (170K+ pairs│  XGBoost         │  (HH-RLHF test set)   │
│              │  DeBERTa-v3      │                        │
│  LLM-Bar     │                  │  Adversarial Stress    │
│  (419 adv.   │  5-fold CV +     │  Test (LLM-Bar)        │
│   inputs)    │  Grid Search     │                        │
└──────┬───────┴────────┬─────────┴──────────┬────────────┘
       │                │                    │
       ▼                ▼                    ▼
  Feature Eng.    MLflow Tracking      Degradation
  + Clustering    + Artifact Store     Quantification
                       │
                       ▼
                 Streamlit App
                 (Live Scoring)
```

---

## Datasets

| Dataset | Source | Size | Role |
|---|---|---|---|
| Anthropic HH-RLHF | Hugging Face | 170K+ preference pairs | Training + EDA + Clustering |
| LLM-Bar | GitHub (EMNLP 2023) | 419 adversarial pairs | Evaluator stress-test only |

---

## Methods

### Unsupervised Learning — Response Quality Archetype Discovery
- K-Means (K swept 2-12, elbow + silhouette selection)
- DBSCAN (density-based, noise-point identification)
- Hierarchical clustering (Ward linkage, dendrogram analysis)
- Evaluation: Davies-Bouldin index, cluster-label cross-tabulation

### Supervised Learning — Quality Score Prediction
- Logistic Regression (L2, interpretable baseline)
- XGBoost (gradient-boosted ensemble, cross-validated grid search)
- DeBERTa-v3-small (fine-tuned transformer, end-to-end text classification)
- Metrics: AUC-ROC, F1 (macro), precision-recall, calibration plots

### Evaluator Reliability Stress-Test (Novel Contribution)
- All trained models applied zero-shot to LLM-Bar adversarial pairs
- Degradation measured across 4 perturbation categories
- False-preference rate quantified per model family
- Cluster membership correlated with adversarial vulnerability

---

## MLflow Experiment Tracking

All experiments are tracked with MLflow (SQLite backend):
- Model parameters, hyperparameters, and metrics logged per run
- Artifacts versioned per phase
- Reproducible from config files

---

## Project Status

| Phase | Status |
|---|---|
| Data loading + feature engineering | In Progress |
| EDA + visualization | Planned |
| Unsupervised clustering | Planned |
| Supervised model training + evaluation | Planned |
| Adversarial stress-test | Planned |
| Streamlit app deployment | Planned |
| Final report + writeup | Planned |

---

## Repository Structure

```
InferenceLens/
├── data/               # Dataset loading and preprocessing scripts
├── features/           # Feature extraction (BERTScore, ROUGE, embeddings)
├── clustering/         # Unsupervised learning experiments
├── models/             # Supervised model training and evaluation
├── stress_test/        # Adversarial evaluator reliability experiments
├── app/                # Streamlit scoring interface
├── mlruns/             # MLflow experiment logs
├── notebooks/          # EDA and visualization notebooks
├── configs/            # Experiment configuration files
└── reports/            # Final findings and writeup
```

---

## Related Work

- **Multi-Agent Inference Reliability Framework** — Prior work investigating critic reliability in agentic LLM pipelines. Results published at [kalyan-venk.github.io/agentic-llmops](https://kalyan-venk.github.io/agentic-llmops/)
- **LLM-Bar** — Zeng et al., EMNLP 2023. Evaluating LLMs as Judges with LLM-Bar.
- **HH-RLHF** — Bai et al., Anthropic 2022. Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.

---

## Academic Context

Developed as part of **DSC 478 — Programming Machine Learning Applications**
Advisor: **Prof. Bamshad Mobasher**, DePaul University
Timeline: May 2026 — Present

---

## Author

**Kalyan Venkatesh**
[LinkedIn](https://www.linkedin.com/in/kalyan-venk/) | [GitHub](https://github.com/kalyan-venk) | [Prior Research](https://kalyan-venk.github.io/agentic-llmops/)

---

*InferenceLens is part of a growing body of work on inference-time reliability and evaluation robustness in LLM systems.*
