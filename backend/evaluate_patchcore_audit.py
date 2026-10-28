
from pathlib import Path
import csv

import torch
from anomalib.data import Folder
from anomalib.models import Patchcore
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "anomalib_dataset_audit_v1"
CHECKPOINT = (
    ROOT / "data" / "patchcore_audit_results"
    / "Patchcore" / "q_insight_audit_v1"
    / "v0" / "weights" / "lightning" / "model.ckpt"
)
OUTPUT_CSV = ROOT / "data" / "patchcore_audit_predictions.csv"


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    datamodule = Folder(
        name="q_insight_audit_v1",
        root=DATASET,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/defective",
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        seed=42,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
    )
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    model = Patchcore.load_from_checkpoint(
        str(CHECKPOINT),
        map_location="cpu",
    )
    model.eval()

    rows = []

    with torch.inference_mode():
        for batch in test_loader:
            output = model(batch.image)
            scores = output.pred_score.detach().cpu().reshape(-1).tolist()
            labels = batch.gt_label.detach().cpu().reshape(-1).tolist()
            paths = batch.image_path

            for path, label, score in zip(paths, labels, scores):
                rows.append({
                    "image": Path(str(path)).name,
                    "path": str(path),
                    "actual": int(label),
                    "score": float(score),
                })

    if len(rows) != 12:
        raise RuntimeError(
            f"Expected 12 test predictions, got {len(rows)}. "
            "Check the configured test split before interpreting metrics."
        )

    rows.sort(key=lambda row: (row["actual"], row["image"]))

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "path", "actual", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)

    y_true = [row["actual"] for row in rows]
    scores = [row["score"] for row in rows]

    # Anomaly scores are continuous scores, not probabilities.
    auc = roc_auc_score(y_true, scores)

    # Exploratory threshold only; not a validated production threshold.
    threshold = 0.5
    y_pred = [int(score >= threshold) for score in scores]

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    print("\nINDIVIDUAL PATCHCORE PREDICTIONS")
    print("-" * 78)
    print(f"{'Image':<20} {'Actual':<12} {'Anomaly score':>16}")
    print("-" * 78)

    for row in rows:
        actual = "DEFECTIVE" if row["actual"] == 1 else "GOOD"
        print(
            f"{row['image']:<20} {actual:<12} "
            f"{row['score']:>16.6f}"
        )

    print("\nINDEPENDENT METRICS")
    print(f"Images evaluated: {len(rows)}")
    print(f"Good images: {y_true.count(0)}")
    print(f"Defective images: {y_true.count(1)}")
    print(f"AUROC: {auc:.4f}")
    print(f"Exploratory threshold: {threshold}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"\nPredictions saved to: {OUTPUT_CSV}")
    print("Existing model and API were not modified.")


if __name__ == "__main__":
    main()
