
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "patchcore_single_switch_v1_predictions.csv"

df = pd.read_csv(CSV_PATH)

# Ground-truth labels: defective = 1, good = 0
df["actual_binary"] = (df["actual"].str.lower() == "defective").astype(int)

print(f"Total images: {len(df)}")
print(f"Normal images: {(df['actual_binary'] == 0).sum()}")
print(f"Defective images: {(df['actual_binary'] == 1).sum()}")
print(f"Missing scores: {df['anomaly_score'].isna().sum()}")

if df["anomaly_score"].isna().any():
    raise ValueError("Some anomaly scores are missing.")

y_true = df["actual_binary"]
scores = df["anomaly_score"]

print(f"\nIndependent image AUROC: {roc_auc_score(y_true, scores):.4f}")

# Diagnostic threshold only; not a production threshold.
threshold = float(scores.median())
y_pred = (scores >= threshold).astype(int)

print(f"Diagnostic median threshold: {threshold:.6f}")
print(f"Diagnostic F1: {f1_score(y_true, y_pred, zero_division=0):.4f}")
print("\nConfusion matrix [ [TN, FP], [FN, TP] ]:")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["good", "defective"],
    zero_division=0,
))

df["diagnostic_prediction"] = [
    "defective" if p else "good" for p in y_pred
]
df["diagnostic_correct"] = y_true == y_pred

print("\nImages misclassified at this diagnostic threshold:")
errors = df.loc[
    ~df["diagnostic_correct"],
    ["image", "actual", "anomaly_score", "diagnostic_prediction"],
]
print(errors.to_string(index=False) if not errors.empty else "None")

output = ROOT / "data" / "patchcore_single_switch_v1_diagnostic.csv"
df.to_csv(output, index=False)
print(f"\nDiagnostic report saved to: {output}")
