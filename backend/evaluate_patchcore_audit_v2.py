
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
DATASET = ROOT / "data" / "anomalib_dataset_audit_v2"
CHECKPOINT = (
    ROOT / "data" / "patchcore_audit_v2_results"
    / "Patchcore" / "q_insight_audit_v2"
    / "v0" / "weights" / "lightning" / "model.ckpt"
)
OUTPUT_CSV = ROOT / "data" / "patchcore_audit_v2_predictions.csv"


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    datamodule = Folder(
        name="q_insight_audit_v2",
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

    model = Patchcore.load_from_checkpoint(
        str(CHECKPOINT), map_location="cpu"
    )
    model.eval()

    rows = []
    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            output = model(batch.image)
            scores = output.pred_score.detach().cpu().reshape(-1).tolist()
            labels = batch.gt_label.detach().cpu().reshape(-1).tolist()

            for path, label, score in zip(batch.image_path, labels, scores):
                rows.append({
                    "image": Path(str(path)).name,
                    "path": str(path),
                    "actual": int(label),
                    "score": float(score),
                })

    if len(rows) != 12:
        raise RuntimeError(
            f"Expected 12 test predictions, received {len(rows)}."
        )

    rows.sort(key=lambda row: (row["actual"], row["image"]))

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image", "path", "actual", "score"]
        )
        writer.writeheader()
        writer.writerows(rows)

    y_true = [row["actual"] for row in rows]
    scores = [row["score"] for row in rows]
    auc = roc_auc_score(y_true, scores)

    # Exploratory threshold only; not validated for production.
    threshold = 0.5
    y_pred = [int(score >= threshold) for score in scores]
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    print("\nPATCHCORE V2 PREDICTIONS")
    print("-" * 65)
    print(f"{'Image':<16} {'Actual':<12} {'Score':>12}")
    print("-" * 65)

    for row in rows:
        label = "DEFECTIVE" if row["actual"] == 1 else "GOOD"
        print(f"{row['image']:<16} {label:<12} {row['score']:>12.6f}")

    print("\nINDEPENDENT METRICS")
    print(f"Images: {len(rows)}")
    print(f"AUROC: {auc:.4f}")
    print(f"Threshold (exploratory): {threshold}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"\nCSV saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
