
from pathlib import Path
import csv
import torch

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "data" / "anomalib_dataset"
RESULTS = BASE / "data" / "patchcore_results"
OUTPUT = BASE / "data" / "patchcore_predictions.csv"


def main():
    datamodule = Folder(
        name="q_insight",
        root=DATASET,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/defective",
        train_batch_size=4,
        eval_batch_size=1,
        num_workers=0,
        seed=42,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",  
    )

    checkpoint = (
        RESULTS / "Patchcore" / "q_insight" / "latest"
        / "weights" / "lightning" / "model.ckpt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    model = Patchcore.load_from_checkpoint(
        str(checkpoint), map_location="cpu"
    )
    engine = Engine(
        default_root_dir=RESULTS / "prediction_run",
        accelerator="cpu",
        devices=1,
    )

    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    rows = []
    model.eval()

    with torch.inference_mode():
        for batch in loader:
            output = model(batch.image)

            scores = output.pred_score
            labels = batch.gt_label

            paths = getattr(batch, "image_path", None)
            if paths is None:
                paths = getattr(batch, "image_paths", None)

            for index in range(len(scores)):
                score = float(scores[index].detach().cpu().reshape(-1)[0])
                label = int(labels[index].detach().cpu().reshape(-1)[0])

                path = str(paths[index]) if paths is not None else f"unknown_{len(rows)}"

                rows.append({
                    "image_path": path,
                    "ground_truth": "defective" if label == 1 else "good",
                    "anomaly_score": score,
                })

    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image_path", "ground_truth", "anomaly_score"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Predictions exported: {len(rows)}")
    print(f"Saved to: {OUTPUT}")
    for row in sorted(rows, key=lambda item: item["anomaly_score"], reverse=True):
        print(
            f"{row['ground_truth']:10s} | "
            f"{row['anomaly_score']:.6f} | {row['image_path']}"
        )


if __name__ == "__main__":
    main()
