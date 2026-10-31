from pathlib import Path
import csv
import json
import torch

from anomalib.data import Folder
from anomalib.models import Patchcore

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SYNTHETIC_DATA = PROJECT_ROOT / "data" / "single_switch_v1" / "prepared_v1"
REAL_PHOTOS = PROJECT_ROOT / "data" / "real_switch_pilot" / "damaged_faceplate"

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

ANNOTATIONS_JSON = (
    PROJECT_ROOT / "data" / "faceplate_annotations" / "annotations.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "real_faceplate_reports"
OUTPUT_CSV = OUTPUT_DIR / "real_faceplate_pilot_evaluation.csv"
OUTPUT_JSON = OUTPUT_DIR / "real_faceplate_pilot_evaluation.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    # 1. Verify required inputs
    for path in (SYNTHETIC_DATA, REAL_PHOTOS, CHECKPOINT, ANNOTATIONS_JSON):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    image_paths = sorted(
        p for p in REAL_PHOTOS.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {REAL_PHOTOS}")

    # 2. Read the actual annotation JSON structure:
    #    {"project": ..., "images": {"photo.jpg": {...}}}
    with ANNOTATIONS_JSON.open("r", encoding="utf-8") as file:
        annotation_data = json.load(file)

    image_annotations = annotation_data.get("images")

    if not isinstance(image_annotations, dict):
        raise ValueError(
            "Expected annotations.json to contain an 'images' dictionary."
        )

    missing = [
        p.name for p in image_paths
        if p.name not in image_annotations
    ]
    if missing:
        raise ValueError(
            "Missing annotation records for: " + ", ".join(missing)
        )

    print(f"Real photographs found: {len(image_paths)}")
    print(f"Annotation records found: {len(image_annotations)}")
    print("Loading existing PatchCore checkpoint...")

    # 3. Use the same model configuration as the working script
    datamodule = Folder(
        name="q_insight_real_faceplate_pilot_evaluation",
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

    # 4. Score the real photographs
    scores = {}
    real_folder = str(REAL_PHOTOS.resolve()).lower()

    print("Scoring images...")

    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            for field in ("image", "gt_label", "gt_mask"):
                value = getattr(batch, field, None)
                if value is not None:
                    setattr(batch, field, value.to("cpu"))

            predictions = model.predict_step(batch, 0)

            for prediction in predictions:
                image_path = str(
                    getattr(prediction, "image_path", "") or ""
                )

                if not image_path:
                    continue

                if real_folder not in str(
                    Path(image_path).resolve()
                ).lower():
                    continue

                score = getattr(prediction, "pred_score", None)

                if score is None:
                    print(f"Warning: score unavailable for {image_path}")
                    continue

                scores[Path(image_path).name] = float(
                    score.detach().cpu().reshape(-1)[0]
                )

    if not scores:
        raise RuntimeError(
            "No image scores were returned. Compare checkpoint loading "
            "and dataloader setup with inspect_real_faceplate.py."
        )

    # 5. Combine model scores with manual annotations
    rows = []

    for image_path in image_paths:
        name = image_path.name
        annotation = image_annotations[name]

        if name not in scores:
            print(f"Warning: no score returned for {name}")
            continue

        status = str(
            annotation.get("review_status", "uncertain")
        ).lower()

        defects = annotation.get("defects", [])
        if not isinstance(defects, list):
            defects = []

        defect_labels = sorted({
            str(item.get("type", "unlabelled"))
            for item in defects
            if isinstance(item, dict)
        })

        rows.append({
            "image": name,
            "anomaly_score": round(scores[name], 6),
            "manual_review_status": status,
            "included_in_confirmed_comparison": (
                status in {"good", "defective"}
            ),
            "manual_box_count": len(defects),
            "manual_defect_labels": "; ".join(defect_labels),
            "interpretation": (
                "Uncalibrated anomaly score; not a defect probability "
                "or confirmed defect diagnosis"
            ),
        })

    if not rows:
        raise RuntimeError("No scores could be matched to annotation records.")

    # 6. Save reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    confirmed = [
        row for row in rows
        if row["included_in_confirmed_comparison"]
    ]
    good = [
        row for row in confirmed
        if row["manual_review_status"] == "good"
    ]
    defective = [
        row for row in confirmed
        if row["manual_review_status"] == "defective"
    ]
    uncertain = [
        row for row in rows
        if row["manual_review_status"] not in {"good", "defective"}
    ]

    summary = {
        "project": "Q-Insight",
        "purpose": "Descriptive real-faceplate pilot evaluation",
        "checkpoint": str(CHECKPOINT),
        "annotation_source": str(ANNOTATIONS_JSON),
        "total_scored_images": len(rows),
        "confirmed_good_count": len(good),
        "confirmed_defective_count": len(defective),
        "uncertain_or_unconfirmed_count": len(uncertain),
        "good_score_range": (
            {
                "min": min(r["anomaly_score"] for r in good),
                "max": max(r["anomaly_score"] for r in good),
            } if good else None
        ),
        "defective_score_range": (
            {
                "min": min(r["anomaly_score"] for r in defective),
                "max": max(r["anomaly_score"] for r in defective),
            } if defective else None
        ),
        "limitations": [
            "Anomaly scores are not calibrated probabilities.",
            "The model was trained on synthetic normal images.",
            "Real photographs may differ from synthetic training images.",
            "Manual review labels are not independently verified ground truth.",
            "The small pilot cannot establish reliable real-world accuracy.",
            "Do not derive a production decision threshold from these images.",
        ],
        "results": rows,
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    # 7. Display results
    print("\nREAL FACEPLATE PILOT RESULTS")
    print("-" * 95)

    for row in rows:
        print(
            f"{row['image']:<24} "
            f"score={row['anomaly_score']:>10.6f}  "
            f"review={row['manual_review_status']:<10} "
            f"boxes={row['manual_box_count']}"
        )

    print("\nREVIEW SUMMARY")
    print(f"Confirmed good images:      {len(good)}")
    print(f"Confirmed defective images: {len(defective)}")
    print(f"Uncertain/unconfirmed:      {len(uncertain)}")

    if good:
        print(
            "Good score range: "
            f"{min(r['anomaly_score'] for r in good):.6f} - "
            f"{max(r['anomaly_score'] for r in good):.6f}"
        )

    if defective:
        print(
            "Defective score range: "
            f"{min(r['anomaly_score'] for r in defective):.6f} - "
            f"{max(r['anomaly_score'] for r in defective):.6f}"
        )

    print(f"\nCSV report:  {OUTPUT_CSV}")
    print(f"JSON report: {OUTPUT_JSON}")
    print("\nEvaluation complete. No model training or annotation changes made.")
    print("These results are preliminary and are not validated accuracy.")


if __name__ == "__main__":
    main()
