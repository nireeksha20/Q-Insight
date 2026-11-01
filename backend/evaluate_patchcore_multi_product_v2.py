from pathlib import Path
import csv
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "anomalib_multi_product_v1"
VALIDATION_ROOT = DATA_ROOT / "test"
CHECKPOINT = (
    PROJECT_ROOT / "data" / "patchcore_multi_product_v2_results"
    / "Patchcore" / "q_insight_multi_product_v2" / "v0"
    / "weights" / "lightning" / "model.ckpt"
)
OUTPUT_CSV = PROJECT_ROOT / "data" / "multi_product_v2_evaluation.csv"


def main():
    for folder in ("good", "defective"):
        path = VALIDATION_ROOT / folder
        if not path.is_dir() or not any(path.iterdir()):
            raise FileNotFoundError(f"Missing or empty folder: {path}")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    dm = Folder(
        name="q_insight_multi_product_v2_eval",
        root=DATA_ROOT,
        normal_dir=DATA_ROOT / "train" / "good",
        normal_test_dir=VALIDATION_ROOT / "good",
        abnormal_dir=VALIDATION_ROOT / "defective",
        test_split_mode="from_dir",
        val_split_mode="from_train",
        val_split_ratio=0.2,
        train_batch_size=8,
        eval_batch_size=1,
        num_workers=0,
        seed=42,
    )
    dm.setup()

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )
    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    model.load_state_dict(
        checkpoint.get("state_dict", checkpoint), strict=True
    )
    model.eval()

    rows = []
    with torch.inference_mode():
        for batch in dm.test_dataloader():
            if batch.image is not None:
                batch.image = batch.image.cpu()

            predictions = model.predict_step(batch, 0)
            for prediction in predictions:
                path = Path(str(prediction.image_path))
                actual = (
                    "defective"
                    if "defective" in str(path).lower()
                    else "good"
                )
                score = float(
                    prediction.pred_score.detach().cpu().reshape(-1)[0]
                )
                rows.append({
                    "filename": path.name,
                    "actual": actual,
                    "score": score,
                })

    if not rows:
        raise RuntimeError("No predictions were produced.")

    # Report score separation and threshold-independent ranking.
    good = [r["score"] for r in rows if r["actual"] == "good"]
    defective = [r["score"] for r in rows if r["actual"] == "defective"]

    # A threshold sweep is diagnostic only; it is not a final test-set
    # threshold selection.
    candidates = sorted({r["score"] for r in rows})
    best = None
    for threshold in candidates:
        tn = sum(s < threshold for s in good)
        fp = len(good) - tn
        fn = sum(s < threshold for s in defective)
        tp = len(defective) - fn
        accuracy = (tn + tp) / len(rows)
        balanced = 0.5 * (
            tn / len(good) + tp / len(defective)
        )
        candidate = (balanced, accuracy, threshold, tn, fp, fn, tp)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "actual", "score"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nMULTI-PRODUCT PATCHCORE V2 — HELD-OUT EVALUATION")
    print("-" * 72)
    for r in sorted(rows, key=lambda x: (x["actual"], x["filename"])):
        print(
            f'{r["filename"]:24} actual={r["actual"]:10} '
            f'score={r["score"]:.6f}'
        )

    for label, scores in (("GOOD", good), ("DEFECTIVE", defective)):
        print(f"\n{label}: {len(scores)} images")
        print(f"  Score range: {min(scores):.6f} - {max(scores):.6f}")
        print(f"  Mean score:  {sum(scores) / len(scores):.6f}")

    _, _, threshold, tn, fp, fn, tp = best
    print("\nBEST THRESHOLD ON THIS EVALUATION SET (diagnostic only)")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Good correctly accepted (TN): {tn}/{len(good)}")
    print(f"  Good incorrectly flagged (FP): {fp}/{len(good)}")
    print(f"  Defects missed (FN): {fn}/{len(defective)}")
    print(f"  Defects detected (TP): {tp}/{len(defective)}")
    print(f"  Accuracy: {(tn + tp) / len(rows):.1%}")
    print(f"  Balanced accuracy: {best[0]:.1%}")
    print(f"\nCSV saved to: {OUTPUT_CSV}")
    print("The existing API, checkpoint, and dataset were not modified.")


if __name__ == "__main__":
    main()
