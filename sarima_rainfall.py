"""
SARIMA — Rainfall Anomaly Detection (fixed version)
Dataset: rainfall_india.csv from data.gov.in
Install: pip install pandas numpy matplotlib scikit-learn statsmodels tabulate
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from tabulate import tabulate
 
# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  SARIMA — Rainfall Anomaly Detection")
print("=" * 60)
 
df_raw = pd.read_csv("rainfall_india.csv")
df_raw.columns = df_raw.columns.str.strip().str.upper()
print(f"Loaded: {df_raw.shape[0]} rows | Columns: {list(df_raw.columns)}")
 
# ─────────────────────────────────────────────────────────────
# STEP 2 — MELT TO MONTHLY ROWS
# ─────────────────────────────────────────────────────────────
print("\n[STEP 2] Reshaping to monthly time series ...")
 
MONTH_COLS = ["JAN","FEB","MAR","APR","MAY","JUN",
              "JUL","AUG","SEP","OCT","NOV","DEC"]
MONTH_NUM  = {m: i+1 for i, m in enumerate(MONTH_COLS)}
 
df = (df_raw[["SUBDIVISION","YEAR"] + MONTH_COLS]
      .melt(id_vars=["SUBDIVISION","YEAR"],
            value_vars=MONTH_COLS,
            var_name="MONTH_NAME",
            value_name="RAINFALL"))
 
df["MONTH"]    = df["MONTH_NAME"].map(MONTH_NUM)
df["RAINFALL"] = pd.to_numeric(df["RAINFALL"], errors="coerce")
df = df.dropna(subset=["RAINFALL"]).reset_index(drop=True)
 
df["DATE"] = pd.to_datetime({
    "year":  df["YEAR"],
    "month": df["MONTH"],
    "day":   1
})
 
df = df.sort_values(["SUBDIVISION","DATE"]).reset_index(drop=True)
print(f"Monthly rows: {len(df):,} | Subdivisions: {df['SUBDIVISION'].nunique()}")
print(f"Columns: {list(df.columns)}")
 
# ─────────────────────────────────────────────────────────────
# STEP 3 — LABEL ANOMALIES (Z-score, |z| > 2)
# ─────────────────────────────────────────────────────────────
print("\n[STEP 3] Labeling anomalies ...")
 
# Use transform() instead of groupby().apply() — avoids the
# KeyError/column-drop bug in pandas 2.x and Python 3.14
df["MEAN"] = df.groupby(["SUBDIVISION","MONTH"])["RAINFALL"].transform("mean")
df["STD"]  = df.groupby(["SUBDIVISION","MONTH"])["RAINFALL"].transform("std").fillna(1)
df["Z_SCORE"] = (df["RAINFALL"] - df["MEAN"]) / (df["STD"] + 1e-9)
 
df["ANOMALY"] = (df["Z_SCORE"].abs() > 2.0).astype(int)
df["TYPE"]    = "Normal"
df.loc[df["Z_SCORE"] >  2.0, "TYPE"] = "Wet Anomaly"
df.loc[df["Z_SCORE"] < -2.0, "TYPE"] = "Dry Anomaly"
 
print("Anomaly breakdown:")
print(df["TYPE"].value_counts().to_string())
print(f"Overall anomaly rate: {df['ANOMALY'].mean()*100:.1f}%")
 
# ─────────────────────────────────────────────────────────────
# STEP 4 — PICK ONE SUBDIVISION FOR SARIMA
# ─────────────────────────────────────────────────────────────
TARGET_SUB = df["SUBDIVISION"].value_counts().index[0]
print(f"\n[STEP 4] Using subdivision: '{TARGET_SUB}'")
 
ts_df  = df[df["SUBDIVISION"] == TARGET_SUB].sort_values("DATE").reset_index(drop=True)
ts     = ts_df.set_index("DATE")["RAINFALL"]
labels = ts_df.set_index("DATE")["ANOMALY"]
 
print(f"Time series: {len(ts)} months  ({ts.index.min().year}–{ts.index.max().year})")
 
# ─────────────────────────────────────────────────────────────
# STEP 5 — SARIMA FUNCTION
# ─────────────────────────────────────────────────────────────
ANOMALY_THRESHOLD = 2.0
 
def run_sarima(ts, labels, train_ratio, split_label):
    n        = len(ts)
    n_train  = int(n * train_ratio)
    n_test   = n - n_train
    ts_train = ts.iloc[:n_train]
    ts_test  = ts.iloc[n_train:]
    y_true   = labels.iloc[n_train:].values
 
    print(f"  [{split_label}] train={n_train}, test={n_test} ... ", end="", flush=True)
 
    model = SARIMAX(ts_train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fit      = model.fit(disp=False)
    forecast = fit.forecast(steps=n_test)
 
    residuals = ts_test.values - forecast.values
    res_z     = (residuals - residuals.mean()) / (residuals.std() + 1e-9)
    y_pred    = (np.abs(res_z) > ANOMALY_THRESHOLD).astype(int)
 
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
 
    print(f"Accuracy={acc*100:.2f}%")
 
    return {
        "split":     split_label,
        "n_train":   n_train,
        "n_test":    n_test,
        "accuracy":  round(acc  * 100, 2),
        "precision": round(prec * 100, 2),
        "recall":    round(rec  * 100, 2),
        "f1":        round(f1   * 100, 2),
        "cm":        confusion_matrix(y_true, y_pred),
        "ts_test":   ts_test,
        "forecast":  forecast,
        "y_true":    y_true,
        "y_pred":    y_pred,
    }
 
# ─────────────────────────────────────────────────────────────
# STEP 6 — RUN 4 SPLITS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[STEP 6] Running SARIMA with 4 train/test splits ...")
print("="*60)
 
SPLITS  = [(0.60,"60:40"), (0.70,"70:30"), (0.80,"80:20"), (0.90,"90:10")]
results = [run_sarima(ts, labels, r, l) for r, l in SPLITS]
 
# ─────────────────────────────────────────────────────────────
# STEP 7 — RESULTS TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS TABLE — copy this into your report")
print("="*60)
table = [[r["split"], r["n_train"], r["n_test"],
          f"{r['accuracy']}%", f"{r['precision']}%",
          f"{r['recall']}%",   f"{r['f1']}%"]
         for r in results]
print(tabulate(table,
               headers=["Split","Train","Test","Accuracy","Precision","Recall","F1"],
               tablefmt="grid"))
 
# ─────────────────────────────────────────────────────────────
# STEP 8 — 5 TEST CASE VALIDATION (80:20 split)
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5 TEST CASE VALIDATION  (80:20 split)")
print("="*60)
 
r80      = next(r for r in results if r["split"] == "80:20")
ts_test  = r80["ts_test"]
forecast = r80["forecast"]
y_true80 = r80["y_true"]
y_pred80 = r80["y_pred"]
n_test   = len(ts_test)
 
indices = [0, n_test//4, n_test//2, 3*n_test//4, n_test-1]
 
val_rows = []
for i, idx in enumerate(indices, 1):
    date     = ts_test.index[idx].strftime("%b %Y")
    actual   = round(ts_test.iloc[idx], 2)
    pred_val = round(forecast.iloc[idx], 2)
    residual = round(actual - pred_val, 2)
    t_lbl    = "ANOMALY" if y_true80[idx] else "normal"
    p_lbl    = "ANOMALY" if y_pred80[idx] else "normal"
    correct  = "YES" if y_true80[idx] == y_pred80[idx] else "NO"
    val_rows.append([i, date, actual, pred_val, residual, t_lbl, p_lbl, correct])
 
print(tabulate(val_rows,
               headers=["Case","Date","Actual RF","Forecast","Residual",
                         "True Label","Predicted","Correct?"],
               tablefmt="grid"))
 
correct_n = sum(1 for r in val_rows if r[7] == "YES")
print(f"\nValidation: {correct_n}/5 correct  ({correct_n*20}%)")
 
# ─────────────────────────────────────────────────────────────
# STEP 9 — PLOTS
# ─────────────────────────────────────────────────────────────
print("\n[STEP 9] Saving plots ...")
 
# Plot 1: accuracy bar chart
fig, ax = plt.subplots(figsize=(8, 4))
split_lbls = [r["split"] for r in results]
accs       = [r["accuracy"] for r in results]
bars = ax.bar(split_lbls, accs,
              color=["#4C9ED9","#6DB36D","#FF8C42","#D96C6C"],
              width=0.5, edgecolor="white")
for b, v in zip(bars, accs):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4,
            f"{v}%", ha="center", fontsize=11, fontweight="bold")
ax.set_title("SARIMA — Test Accuracy per Split", fontsize=13, pad=12)
ax.set_xlabel("Train : Test Split")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 110)
ax.axhline(sum(accs)/len(accs), color="gray", linestyle="--", lw=1, label="Average")
ax.legend()
plt.tight_layout()
plt.savefig("sarima_accuracy_splits.png", dpi=150, bbox_inches="tight")
plt.show()
 
# Plot 2: forecast vs actual
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig.suptitle(f"SARIMA Forecast vs Actual — {TARGET_SUB}  (80:20 split)", fontsize=13)
axes[0].plot(r80["ts_test"].values, label="Actual",    color="steelblue", lw=1.2)
axes[0].plot(r80["forecast"].values, label="Forecast", color="tomato",
             linestyle="--", lw=1.2)
axes[0].set_ylabel("Rainfall (mm)")
axes[0].legend()
axes[1].step(range(len(y_true80)), y_true80, where="mid",
             label="True anomaly",      color="steelblue", lw=1.5)
axes[1].step(range(len(y_pred80)), y_pred80, where="mid",
             label="Predicted anomaly", color="tomato", linestyle="--", lw=1.5)
for idx in indices:
    axes[1].axvline(idx, color="green", linestyle=":", lw=1, alpha=0.8)
axes[1].set_ylabel("Anomaly (1=yes)")
axes[1].set_xlabel("Test months")
axes[1].set_yticks([0, 1])
axes[1].legend()
plt.tight_layout()
plt.savefig("sarima_forecast_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()
 
# Plot 3: all metrics grouped bar
fig, ax = plt.subplots(figsize=(10, 5))
metric_names = ["accuracy","precision","recall","f1"]
colors       = ["#4C9ED9","#FF8C42","#6DB36D","#D96C6C"]
x = np.arange(len(results))
w = 0.2
for i, (m, c) in enumerate(zip(metric_names, colors)):
    vals = [r[m] for r in results]
    ax.bar(x + i*w, vals, w, label=m.capitalize(), color=c, alpha=0.85)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels([r["split"] for r in results])
ax.set_xlabel("Train : Test Split")
ax.set_ylabel("Score (%)")
ax.set_title("SARIMA — All Metrics Across All Splits", fontsize=13)
ax.set_ylim(0, 115)
ax.legend()
plt.tight_layout()
plt.savefig("sarima_all_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
 
print("\n" + "="*60)
print("Done! Files saved:")
print("  sarima_accuracy_splits.png")
print("  sarima_forecast_vs_actual.png")
print("  sarima_all_metrics.png")
print("="*60)
