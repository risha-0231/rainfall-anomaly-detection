# Rainfall Anomaly Detection — SARIMA Model

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Model](https://img.shields.io/badge/model-SARIMA-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

##  Project Overview

This project detects **rainfall anomalies** (unexpected wet or dry spells) across Indian meteorological subdivisions using **SARIMA** (Seasonal AutoRegressive Integrated Moving Average) — a classical statistical time-series model. Anomalies are labeled using the **Z-score method** and the model is evaluated across four train-test splits and validated on five real test cases.

The system identifies:
- **Wet Anomalies**: Unusually heavy rainfall months (Z-score > +2.0)
- **Dry Anomalies**: Unusually deficient rainfall months (Z-score < −2.0)
- **Normal Months**: Rainfall within the expected seasonal range

---

##  Pipeline Architecture
┌──────────────────────────────────────────────────────┐
│ DATA LOADING LAYER │
│ • Kaggle: Rainfall in India (1901–2015) │
│ • 36 meteorological subdivisions, monthly records │
│ • Source: India Meteorological Department (IMD) │
└────────────────────┬─────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────┐
│ PREPROCESSING LAYER │
│ • Melt wide CSV → long format (one row per month) │
│ • Missing value imputation (median per subdivision) │
│ • Date column synthesis from Year + Month │
│ • Chronological sorting per subdivision │
└────────────────────┬─────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────┐
│ ANOMALY LABELING (Z-SCORE) │
│ • Group by Subdivision + Month │
│ • Z = (Rainfall − μ) / σ for each group │
│ • Z > +2.0 → Wet Anomaly (label = 1) │
│ • Z < −2.0 → Dry Anomaly (label = 1) │
│ • |Z| ≤ 2.0 → Normal (label = 0) │
└────────────────────┬─────────────────────────────────┘
│
▼
┌──────────────────────────────────────────────────────┐
│ SARIMA MODEL LAYER │
│ • Order : (1, 1, 1) │
│ • Seasonal : (1, 1, 1, 12) — yearly cycle │
│ • Fit on train split, forecast test period │
│ • Residual Z-score → anomaly flag │
└────────────────────┬─────────────────────────────────┘
│
┌──────────┴──────────┐
▼ ▼
┌──────────────────┐ ┌────────────────────┐
│ 4 Train-Test │ │ 5 Test Case │
│ Split Evaluation│ │ Validation │
│ 60:40 70:30 │ │ 5 subdivisions │
│ 80:20 90:10 │ │ peak monsoon │
└──────────────────┘ └────────────────────┘
│ │
└──────────┬──────────┘
▼
┌──────────────────────────────────────────────────────┐
│ RESULTS & METRICS │
│ Accuracy · Precision · Recall · F1 · AUC · Error │
│ Confusion Matrix · ROC Curve · Result Plots │
└──────────────────────────────────────────────────────┘
---

##  ML Pipeline Components

### 1. Dataset

| Property | Value |
|---|---|
| Name | Rainfall in India (1901–2015) |
| Source | Kaggle / India Meteorological Department |
| Link | kaggle.com/datasets/rajanand/rainfall-in-india |
| Raw size | 4,116 rows × 19 columns |
| After melting | 49,322 monthly records |
| Subdivisions | 36 across India |
| Time span | 115 years (1901–2015) |

**Columns used:** `SUBDIVISION`, `YEAR`, `JAN` through `DEC`

---

### 2. Preprocessing

- **Melt**: Wide format (one row per year) → long format (one row per month per subdivision)
- **Type conversion**: All monthly columns cast to `float`, non-numeric → `NaN`
- **Imputation**: Missing values filled with the **median** rainfall for that subdivision-month pair
- **Date synthesis**: `DATE` column created from `YEAR` + `MONTH` (day = 1)
- **Sorting**: Data sorted chronologically per subdivision before any modeling

---

### 3. Anomaly Labeling — Z-Score Method
Z = (Rainfall − μ) / σ
Where μ and σ are computed per (Subdivision, Month) group across all years.

**Thresholds:**

| Z-Score Range | Label | Type |
|---|---|---|
| Z > +2.0 | 1 (Anomaly) | Wet — excess rainfall |
| −2.0 ≤ Z ≤ +2.0 | 0 (Normal) | Within expected range |
| Z < −2.0 | 1 (Anomaly) | Dry — deficient rainfall |

**Dataset anomaly rate:** ~4.5% of all records flagged as anomalies.  
**Why Z-score:** No manual labeling required, statistically grounded, directly interpretable, consistent across all subdivisions.

---

### 4. SARIMA Model

**Full notation:** `SARIMA(1, 1, 1)(1, 1, 1, 12)`

| Parameter | Value | Meaning |
|---|---|---|
| p = 1 | Non-seasonal AR | Uses 1 past rainfall value |
| d = 1 | Differencing | Removes trend (makes series stationary) |
| q = 1 | Non-seasonal MA | Uses 1 past forecast error |
| P = 1 | Seasonal AR | Seasonal autoregressive term |
| D = 1 | Seasonal differencing | Removes yearly seasonal trend |
| Q = 1 | Seasonal MA | Seasonal moving average term |
| s = 12 | Seasonal period | 12 months = 1 full monsoon cycle |

**Library:** `statsmodels.tsa.statespace.sarimax.SARIMAX`  
**Fitting method:** Maximum Likelihood Estimation (MLE)

**How anomaly detection works:**
1. Fit SARIMA on the training portion of a subdivision's rainfall series
2. Forecast each month in the test period (`fit.forecast(steps=n_test)`)
3. Compute residual = Actual − Forecast for every test month
4. Standardize residuals into a Z-score
5. Flag month as anomaly if |residual Z-score| > 2.0

**Why SARIMA suits this data:** India's rainfall has a strong, consistent 12-month monsoon cycle. SARIMA's seasonal differencing (D=1, s=12) explicitly models and removes this cycle — so only deviations that cannot be explained by the known season are flagged.

---

### 5. Experimental Evaluation

**Four train-test splits** (temporal order preserved — data is NOT shuffled):

| Split | Train | Test | Purpose |
|---|---|---|---|
| 60:40 | 60% | 40% | Least training data |
| 70:30 | 70% | 30% | |
| 80:20 | 80% | 20% | Best balance ✓ |
| 90:10 | 90% | 10% | Most training data |

A separate SARIMA model is fitted for each split from scratch.

**Five test case validation (80:20 split):**
- One real test case per subdivision (5 different subdivisions)
- Selects the peak rainfall month from each subdivision's test period
- Per-case accuracy = `(1 − |Actual − Predicted| / Actual) × 100%`

---

### 6. Metrics

| Metric | Formula | What it tells you |
|---|---|---|
| Accuracy | (TP+TN) / Total | Overall correct predictions |
| Precision | TP / (TP+FP) | Of predicted anomalies, how many were real |
| Recall / Sensitivity | TP / (TP+FN) | Of real anomalies, how many were caught |
| F1-Score | 2 × P × R / (P+R) | Balance of precision and recall |
| AUC | Area under ROC curve | Separation ability across all thresholds |
| Error Rate | 1 − Accuracy | Fraction of wrong predictions |

> **Note:** F1-Score is the most important metric here. With only ~4.5% anomalies, a model predicting "normal" every time would score 95.5% accuracy while detecting nothing. F1 penalizes this.

---

##  How to Run

### Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels tabulate
```

### Run the Python script

```bash
# Put rainfall_india.csv in the same folder, then:
python sarima_rainfall.py
```

### Run in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `SARIMA_Rainfall_Anomaly_Detection.ipynb`
3. Run Cell 1 (installs libraries)
4. Run Cell 2 → upload `rainfall_india.csv` when prompted
5. Run all remaining cells top to bottom (`Shift+Enter`)

---

##  Project Structure

The Z-score measures how far a monthly rainfall reading is from its long-run average for that subdivision and calendar month.
sarima_rainfall_anomaly/
├── sarima_rainfall.py # Main Python script (local run)
├── SARIMA_Rainfall_Anomaly_Detection.ipynb # Google Colab notebook
├── README.md # This file
│
└── outputs/ (generated on run)
├── 01_confusion_matrix.png # Confusion matrix — all 4 splits
├── 02_f1_precision_recall.png # F1, Precision, Recall, Sensitivity
├── 03_accuracy_errorrate.png # Accuracy and Error Rate bars
├── 04_roc_auc_curve.png # ROC Curve and AUC — all splits
├── 05_summary_dashboard.png # All metrics in one view
└── 06_forecast_vs_actual.png # SARIMA forecast vs actual (80:20)
>  Dataset not included — download `rainfall_india.csv` from [Kaggle](https://www.kaggle.com/datasets/rajanand/rainfall-in-india) and place it in the root folder before running.

---

##  Result Screenshots

After running the notebook, 6 PNG files are automatically saved and downloaded:

| File | Contents |
|---|---|
| `01_confusion_matrix.png` | TP, TN, FP, FN for all 4 splits |
| `02_f1_precision_recall.png` | F1-Score, Precision, Recall, Sensitivity bars |
| `03_accuracy_errorrate.png` | Accuracy % and Error Rate % per split |
| `04_roc_auc_curve.png` | ROC curves with AUC values |
| `05_summary_dashboard.png` | All 6 metrics + confusion matrix + ROC in one figure |
| `06_forecast_vs_actual.png` | Time-series: actual vs SARIMA forecast + residuals |

---

##  How It Works — A Simple Example

**Scenario:** Is August 1998 rainfall in Kerala anomalous?

1. **Load data** — Kerala August historical average (μ): 380 mm, std (σ): 85 mm
2. **Z-score labeling** — Actual: 620 mm → Z = (620 − 380) / 85 = **+2.82** → Wet Anomaly
3. **SARIMA training** — Fitted on Kerala 1901–1983 monthly rainfall
4. **Forecast** — SARIMA predicts 395 mm → Residual = 225 mm → Residual Z = **+2.64** → exceeds 2.0
5. **Result: ANOMALY detected ✓**

---

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `KeyError: 'SUBDIVISION'` | Fixed in current version — uses `transform()` instead of `groupby().apply()` |
| Negative or >100% case accuracy | Fixed by filtering `actual > 0` before case selection |
| SARIMA takes too long | Normal — fitting 4 SARIMA models takes 2–5 mins |
| `ModuleNotFoundError: tabulate` | Run `pip install tabulate` |
| CSV not found | Rename downloaded Kaggle file to exactly `rainfall_india.csv` |

---

##  Dependencies
pandas >= 2.0.0
numpy >= 1.24.0

---

##  Key References

1. Box, G.E.P. & Jenkins, G.M. (1976). *Time Series Analysis: Forecasting and Control.* Holden-Day.
2. Subbaiah Naidu K.Ch.V. (2016). SARIMA Modeling and Forecasting of Seasonal Rainfall Patterns in India. *IJMTT*, 38, 15–22.
3. Seabold & Perktold (2010). Statsmodels: Econometric and Statistical Modeling with Python. *Proc. SciPy 2010.*
4. Chandola et al. (2009). Anomaly Detection: A Survey. *ACM Computing Surveys*, 41(3).
5. IMD / Kaggle Dataset — kaggle.com/datasets/rajanand/rainfall-in-india

---

##  Academic Context

**Project:** Rainfall Anomaly Detection for Extreme Event Prediction  
**Model:** SARIMA  
**Dataset:** Rainfall in India, 1901–2015 (IMD via Kaggle)  
**Evaluation:** 60:40 · 70:30 · 80:20 · 90:10 train-test splits + 5 test case validation  
**Submission:** Google Colab Notebook + PDF Export + Result Screenshots

---

**Last Updated:** May 2026  
**Version:** 1.0.0
matplotlib >= 3.7.0
scikit-learn >= 1.3.0
statsmodels >= 0.14.0
tabulate >= 0.9.0
