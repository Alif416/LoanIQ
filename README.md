<div align="center">

# LoanIQ — Credit Risk Intelligence

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-189AB4?style=for-the-badge)
![Tests](https://img.shields.io/badge/Tests-48%20passing-16A34A?style=for-the-badge)
![CI](https://github.com/Alif416/Loan-Prediction-App/actions/workflows/test.yml/badge.svg)

**XGBoost credit-risk model trained on 1.3 million LendingClub loans.**  
Real-time default risk prediction with TreeSHAP explanations, repayment analysis, and interactive analytics.

[Live Demo](https://loan-prediction-ml-k4xffncvhbyh7h9qy72mjk.streamlit.app/) · [Notebook](notebooks/loan-prediction.ipynb)

</div>

---

## Overview

LoanIQ predicts whether a loan applicant is likely to fully repay or default, trained on LendingClub's complete 2007–2018 loan book (1.3M loans, 145 features). It demonstrates the full ML lifecycle: EDA, feature selection, temporal validation, model comparison, threshold tuning, explainability, and deployment.

The app is built around three principles a production lending model requires:

- **No data leakage** — temporal train/test split (train pre-2017, test 2017–2018)
- **Honest evaluation** — macro F1 threshold tuning, not accuracy optimisation
- **Explainability** — per-prediction TreeSHAP values via XGBoost native API

---

## Features

| Tab | What it does |
|---|---|
| **Loan Predictor** | 22-field input form → default risk probability → TreeSHAP explanation → repayment schedule |
| **Data Analytics** | 8 interactive charts from a 50k-row sample: default by grade, purpose, home ownership, DTI, time |
| **Model Insights** | Live metrics, ROC curve, confusion matrix, class-level breakdown, global SHAP importance |

---

## Quick Start

```bash
git clone https://github.com/Alif416/Loan-Prediction-App.git
cd Loan-Prediction-App

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

For the Data Analytics and global SHAP tabs, place the LendingClub dataset at `data/loan.csv`  
(available on Kaggle: search "LendingClub Loan Data by kakashi").

---

## Project Structure

```
Loan Approval App/
├── app.py                        # Streamlit application (3 tabs)
├── config.py                     # Paths, constants, dropdown options
├── requirements.txt              # Pinned dependencies
│
├── model/
│   ├── loan_model_pipeline.pkl   # Trained sklearn Pipeline (XGBoost)
│   └── model_metadata.json       # Metrics, threshold, feature lists
│
├── notebooks/
│   └── loan-prediction.ipynb     # EDA, model comparison, fairness analysis
│
├── tests/
│   ├── test_calculations.py      # Unit tests: emi(), risk_label(), validate_inputs()
│   ├── test_model.py             # Integration tests: model load, inference, edge cases
│   └── test_pipeline.py          # Contract tests: config, metadata schema, requirements
│
└── .github/workflows/
    └── test.yml                  # CI: lint (ruff) + pytest on every push
```

---

## ML Pipeline

```
22 LendingClub Features (at origination time only)
              │
              ▼
   ┌──────────────────────────┐
   │     ColumnTransformer    │
   │                          │
   │  Numerical (14)          │  Categorical (8)
   │  SimpleImputer(median)   │  SimpleImputer(most_frequent)
   │  StandardScaler          │  OneHotEncoder(ignore_unknown)
   └────────────┬─────────────┘
                │
                ▼
         XGBClassifier
         n_estimators=200
         class_weight=balanced
                │
                ▼
        Probability Score
        Threshold = 0.37
                │
                ▼
      TreeSHAP Explanation
      (XGBoost native API)
```

---

## Dataset

| Property | Value |
|---|---|
| Source | LendingClub via Kaggle |
| Raw size | 2.26M rows, 145 columns |
| After filtering | 1.3M rows (Fully Paid + Charged Off only) |
| Training period | 2007 – 2016 |
| Test period | 2017 – 2018 (temporal holdout) |
| Features used | 22 (origination-time only — no post-loan leakage) |
| Class balance | ~80% Fully Paid / ~20% Charged Off |
| Imbalance handling | `class_weight='balanced'` |

**Why temporal split?** Random splits let future loan data leak into training, inflating scores. Training on pre-2017 and testing on 2017–2018 mimics real deployment: the model only ever sees past loans, and is evaluated on loans it couldn't have learned from.

---

## Model Performance

Metrics on the 2017–2018 temporal holdout (206,082 loans):

| Metric | Score |
|---|---|
| **ROC-AUC** | 0.7155 |
| **Accuracy** | 74.34% |
| **F1 Score** | 83.56% |
| **Precision** | 84.56% |
| **Recall** | 82.58% |
| Charged Off Recall | 43% |
| Charged Off Precision | 40% |

**Why is accuracy 74%?** The decision threshold is 0.37, not 0.50. At 0.50 the model catches only 1% of defaulting loans — useless in practice. Tuning to 0.37 via macro F1 raises default detection to 43%, at the cost of some accuracy. In a real lending system the threshold would be set by the cost of approving a bad loan vs. rejecting a good one.

**Why not higher AUC?** 0.71 on a proper temporal split is realistic for credit risk — most production models sit in the 0.65–0.80 range. A random 80/20 split on the same data gives ~0.85, but that is inflated by future data leakage.

---

## Design Decisions

**Why XGBoost over Logistic Regression?**  
XGBoost achieved the highest AUC (0.7155) across four models (LR, Random Forest, XGBoost, LightGBM) in head-to-head comparison on the same temporal split. LR would be preferred in a regulated lending context for its interpretability and ECOA compliance, but for a portfolio demonstration XGBoost is the better technical choice.

**Why 22 features, not all 145?**  
The remaining 123 columns are post-loan data — payment history, recovery amounts, hardship flags. Using them would be a data leak: at prediction time (loan application) none of those values exist yet. Only features available at origination are included.

**Why XGBoost's native SHAP instead of the SHAP library?**  
XGBoost 2.x changed its internal model format in a way that breaks SHAP 0.46's model loader. `predict(pred_contribs=True)` produces mathematically identical TreeSHAP values via XGBoost's own implementation, with no version dependency.

---

## Known Limitations

- **No protected attributes used** (no race, gender, religion, national origin). However, `home_ownership`, `addr_state`, and `purpose` can act as demographic proxies. Disparate impact testing would be required before production deployment under ECOA.
- **Temporal drift** — the model was trained on 2007–2016 data. Economic conditions post-2018 (COVID, rate environment) may reduce accuracy on current loans.
- **No model monitoring** — a production system would track score distribution drift and trigger retraining when input distributions shift.
- **Single model** — a production system would run challenger models in parallel and route traffic via A/B testing.

---

## Testing

```bash
pytest tests/ -v
```

48 tests across three files:

| File | Coverage |
|---|---|
| `test_calculations.py` | `emi()`, `risk_label()`, `validate_inputs()` — boundary cases, math correctness |
| `test_model.py` | Pipeline loads, predict_proba in [0,1], safe > risky, unknown category doesn't crash |
| `test_pipeline.py` | Config paths, metadata schema, all requirements pinned |

CI runs on every push via GitHub Actions (`.github/workflows/test.yml`).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.56 |
| ML | scikit-learn 1.6.1, XGBoost 2.1.3 |
| Explainability | XGBoost native TreeSHAP |
| Data | pandas 2.3.3, numpy 1.26.4 |
| Charts | Plotly 5.24.1 |
| Model I/O | joblib 1.4.2 |
| Testing | pytest 9.0.3 |
| Linting | ruff |
| CI | GitHub Actions |

---

## License

MIT
