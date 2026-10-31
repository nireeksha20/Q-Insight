
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "patchcore_single_switch_v1_predictions.csv"

df = pd.read_csv(CSV)
good = df.loc[df["actual"] == "good", "anomaly_score"]
defective = df.loc[df["actual"] == "defective", "anomaly_score"]

print("NORMAL (GOOD) SCORES")
print(f"Count:  {len(good)}")
print(f"Min:    {good.min():.6f}")
print(f"Max:    {good.max():.6f}")
print(f"Mean:   {good.mean():.6f}")
print(f"Median: {good.median():.6f}")

print("\nDEFECTIVE SCORES")
print(f"Count:  {len(defective)}")
print(f"Min:    {defective.min():.6f}")
print(f"Max:    {defective.max():.6f}")
print(f"Mean:   {defective.mean():.6f}")
print(f"Median: {defective.median():.6f}")

print("\nSCORE SEPARATION")
if good.max() < defective.min():
    print("All normal scores are below all defective scores.")
    print(f"Observed gap: {defective.min() - good.max():.6f}")
    print(f"Possible separating threshold: {(good.max() + defective.min()) / 2:.6f}")
else:
    print("The score ranges overlap; inspect individual predictions.")

print("\nLOWEST DEFECT SCORES")
print(
    df[df["actual"] == "defective"]
    .nsmallest(5, "anomaly_score")[["image", "anomaly_score"]]
    .to_string(index=False)
)

print("\nHIGHEST NORMAL SCORES")
print(
    df[df["actual"] == "good"]
    .nlargest(5, "anomaly_score")[["image", "anomaly_score"]]
    .to_string(index=False)
)
