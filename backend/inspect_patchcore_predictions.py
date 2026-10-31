
from pathlib import Path
import csv
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "single_switch_v1" / "prepared_v1"

CHECKPOINT = (
    PROJECT_ROOT
    / "data"
    / "patchcore_single_switch_v1_results"
    / "Patchcore"
    / "q_insight_single_switch_v1"
    / "v1"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

OUTPUT_CSV = PROJECT_ROOT / "data" / "patchcore_single_switch_v1_predictions.csv"


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    datamodule = Folder(
        name="q_insight_single_switch_v1",
        root=DATA_ROOT,
        normal_dir=DATA_ROOT / "train" / "good",
        normal_test_dir=DATA_ROOT / "test" / "good",
        abnormal_dir=DATA_ROOT / "test" / "defective",
        test_split_mode="from_dir",
        val_split_mode="from_train",
        val_split_ratio=0.2,
        train_batch_size=8,
        eval_batch_size=1,
        num_workers=0,
        seed=42,
    )
    datamodule.setup()

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Use the same Anomalib test dataloader.
    dataloader = datamodule.test_dataloader()
    device = torch.device("cpu")
    model.to(device)

    rows = []

    with torch.inference_mode():
        for batch in dataloader:
            # ImageBatch is a data container, not a tensor.
            # Move its tensor fields to the CPU explicitly.
            if hasattr(batch, "image") and batch.image is not None:
                batch.image = batch.image.to(device)

            if hasattr(batch, "gt_label") and batch.gt_label is not None:
                batch.gt_label = batch.gt_label.to(device)

            if hasattr(batch, "gt_mask") and batch.gt_mask is not None:
                batch.gt_mask = batch.gt_mask.to(device)

            predictions = model.predict_step(batch, 0)

            for prediction in predictions:
                image_path = str(
                    getattr(prediction, "image_path", "")
                )

                score = getattr(prediction, "pred_score", None)
                label = getattr(prediction, "pred_label", None)

                if score is None:
                    raise RuntimeError(
                        "Prediction has no pred_score attribute."
                    )

                score_value = float(score.detach().cpu().reshape(-1)[0])
                label_value = (
                    int(label.detach().cpu().reshape(-1)[0])
                    if label is not None
                    else -1
                )

                path = Path(image_path)
                actual = (
                    "defective"
                    if "defective" in str(path).lower()
                    else "good"
                )

                rows.append({
                    "image": path.name,
                    "path": image_path,
                    "actual": actual,
                    "anomaly_score": score_value,
                    "predicted_label": label_value,
                })

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "path",
                "actual",
                "anomaly_score",
                "predicted_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} predictions to:")
    print(OUTPUT_CSV)
    print("\nFirst 10 predictions:")

    for row in rows[:10]:
        print(
            f"{row['image']:<30} "
            f"actual={row['actual']:<10} "
            f"score={row['anomaly_score']:.6f} "
            f"predicted_label={row['predicted_label']}"
        )


if __name__ == "__main__":
    main()
