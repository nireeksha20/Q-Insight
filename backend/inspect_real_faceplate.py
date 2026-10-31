
from pathlib import Path
import csv
import torch

from anomalib.data import Folder
from anomalib.models import Patchcore

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYNTHETIC_DATA = (
    PROJECT_ROOT / "data" / "single_switch_v1" / "prepared_v1"
)

REAL_PHOTOS = (
    PROJECT_ROOT / "data" / "real_switch_pilot" / "damaged_faceplate"
)

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

OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "real_faceplate_patchcore_scores.csv"
)


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not REAL_PHOTOS.is_dir():
        raise FileNotFoundError(f"Photo folder not found: {REAL_PHOTOS}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    real_images = sorted(
        p for p in REAL_PHOTOS.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    if not real_images:
        raise RuntimeError(f"No images found in {REAL_PHOTOS}")

    print(f"Real photographs found: {len(real_images)}")
    print("Loading the existing PatchCore checkpoint...")

    datamodule = Folder(
        name="q_insight_real_faceplate_pilot",
        root=SYNTHETIC_DATA,
        normal_dir=SYNTHETIC_DATA / "train" / "good",
        normal_test_dir=SYNTHETIC_DATA / "test" / "good",
        abnormal_dir=REAL_PHOTOS,
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
    model.to(torch.device("cpu"))

    rows = []
    real_folder = str(REAL_PHOTOS.resolve()).lower()

    print("Scoring images...")

    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            # ImageBatch is a container, so move its tensor fields.
            if getattr(batch, "image", None) is not None:
                batch.image = batch.image.to("cpu")

            if getattr(batch, "gt_label", None) is not None:
                batch.gt_label = batch.gt_label.to("cpu")

            if getattr(batch, "gt_mask", None) is not None:
                batch.gt_mask = batch.gt_mask.to("cpu")

            predictions = model.predict_step(batch, 0)

            for prediction in predictions:
                image_path = str(
                    getattr(prediction, "image_path", "")
                )
                if not image_path:
                    continue

                if real_folder not in str(Path(image_path).resolve()).lower():
                    continue

                score = getattr(prediction, "pred_score", None)
                if score is None:
                    print(f"Warning: no score returned for {image_path}")
                    continue

                score_value = float(
                    score.detach().cpu().reshape(-1)[0]
                )

                rows.append({
                    "image": Path(image_path).name,
                    "path": image_path,
                    "anomaly_score": score_value,
                    "reference_synthetic_normal_min": "",
                    "reference_synthetic_normal_max": "",
                    "interpretation": (
                        "Uncalibrated anomaly score; "
                        "not a confirmed defect classification"
                    ),
                })

    if len(rows) != len(real_images):
        print(
            f"Warning: found {len(real_images)} source images, "
            f"but obtained scores for {len(rows)} images."
        )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nREAL FACEPLATE RESULTS")
    for row in rows:
        print(
            f"{row['image']:<25} "
            f"anomaly_score={row['anomaly_score']:.6f}"
        )

    print(f"\nSaved report: {OUTPUT_CSV}")
    print(
        "\nReference only: synthetic normal scores previously ranged "
        "from 51.668976 to 57.221947. Do not use this range as a "
        "validated threshold for real faceplates."
    )


if __name__ == "__main__":
    main()
