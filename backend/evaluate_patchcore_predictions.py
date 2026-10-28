
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "data" / "patchcore_predictions.csv"

df = pd.read_csv(CSV_PATH)

# 1 = defective, 0 = good
y_true = (df["ground_truth"] == "defective").astype(int)
scores = df["anomaly_score"]

print(f"Total images: {len(df)}")
print(f"Defective: {sum(y_true == 1)}")
print(f"Good: {sum(y_true == 0)}")
print(f"Unique anomaly scores: {scores.nunique()}")

# AUROC evaluates ranking across all possible thresholds.
print(f"\nAUROC: {roc_auc_score(y_true, scores):.4f}")

# Exploratory threshold only; not a production threshold.
threshold = 0.90
y_pred = (scores >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)

print(f"\nExploratory threshold: {threshold:.2f}")
print(f"True positives:  {tp}")
print(f"False positives: {fp}")
print(f"True negatives:  {tn}")
print(f"False negatives: {fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Good", "Defective"],
        zero_division=0,
    )
)

print("\nScores by image:")
print(
    df.sort_values("anomaly_score", ascending=False).to_string(index=False)
)
