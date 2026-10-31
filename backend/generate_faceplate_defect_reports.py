
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "real_switch_pilot" / "damaged_faceplate"
SCORE_FILE = PROJECT_ROOT / "data" / "real_faceplate_patchcore_scores.csv"
VISUAL_DIR = PROJECT_ROOT / "data" / "real_faceplate_heatmaps"
OUTPUT_DIR = PROJECT_ROOT / "data" / "real_faceplate_reports"

# These are preliminary observations from visual review.
# They are NOT confirmed defects and must be reviewed by a human.
CANDIDATE_OBSERVATIONS = {
    "photo_01.jpg": [
        {
            "type": "Possible surface scratches or scuffing",
            "location": "Near the large rectangular cutout",
        },
        {
            "type": "Possible surface wear",
            "location": "Lower-left inner edge of the cutout",
        },
    ],
    "photo_02.jpg": [
        {
            "type": "Possible surface marks or discoloration",
            "location": "Around the cutout and central faceplate",
        },
    ],
    "photo_03.jpg": [
        {
            "type": "Possible surface wear",
            "location": "Central surface and around the openings",
        },
    ],
    "photo_04.jpg": [
        {
            "type": "Possible scratches or surface marks",
            "location": "Near the upper cutout",
        },
    ],
    "photo_05.jpg": [
        {
            "type": "Possible surface scuffing",
            "location": "Around the cutout and adjacent surface",
        },
    ],
}


def normalized_name(value):
    """Normalize filenames, including accidental duplicated extensions."""
    name = Path(str(value).strip()).name

    while name.lower().endswith((".jpg.jpg", ".jpeg.jpeg", ".png.png")):
        name = name.rsplit(".", 1)[0]

    return name.lower()


def load_scores():
    """Read PatchCore scores without assuming a particular column name."""
    scores = {}

    if not SCORE_FILE.exists():
        print(f"Warning: score CSV not found: {SCORE_FILE}")
        return scores

    with SCORE_FILE.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            filename = (
                row.get("image")
                or row.get("filename")
                or row.get("file")
                or row.get("path")
                or ""
            )

            score_value = (
                row.get("anomaly_score")
                or row.get("score")
                or row.get("pred_score")
            )

            if not filename or score_value in (None, ""):
                continue

            try:
                scores[normalized_name(filename)] = float(score_value)
            except (TypeError, ValueError):
                continue

    return scores


def find_visual_file(image_path, suffix):
    """Find a corresponding heatmap or overlay despite filename variations."""
    candidates = [
        VISUAL_DIR / f"{image_path.name}_{suffix}.png",
        VISUAL_DIR / f"{image_path.stem}_{suffix}.png",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.relative_to(PROJECT_ROOT))

    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image folder not found: {IMAGE_DIR}")

    images = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not images:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    scores = load_scores()
    report_items = []

    for image_path in images:
        key = normalized_name(image_path.name)
        score = scores.get(key)

        # Do not apply an unvalidated numeric threshold.
        item = {
            "image": image_path.name,
            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            "patchcore_anomaly_score": score,
            "classification": "Needs Review",
            "classification_status": "Not yet confirmed",
            "candidate_observations": CANDIDATE_OBSERVATIONS.get(
                normalized_name(image_path.name).replace(".jpg.jpg", ".jpg"), []
            ),
            "confirmed_defects": [],
            "heatmap_path": find_visual_file(image_path, "heatmap"),
            "overlay_path": find_visual_file(image_path, "overlay"),
            "reviewer_decision": None,
            "reviewer_notes": "",
        }

        report_items.append(item)

    report = {
        "project": "Q-Insight",
        "report_type": "Real Faceplate Defect Inspection",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": len(report_items),
        "model": "PatchCore",
        "threshold_status": "Not calibrated for real faceplates",
        "important_note": (
            "Anomaly scores indicate visual unusualness, not defect probability. "
            "Candidate observations are unconfirmed and require human review."
        ),
        "images": report_items,
    }

    json_path = OUTPUT_DIR / "faceplate_inspection_report.json"
    csv_path = OUTPUT_DIR / "faceplate_inspection_summary.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    columns = [
        "image",
        "patchcore_anomaly_score",
        "classification",
        "classification_status",
        "candidate_observations",
        "confirmed_defects",
        "heatmap_path",
        "overlay_path",
        "reviewer_decision",
        "reviewer_notes",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for item in report_items:
            row = item.copy()
            row["candidate_observations"] = "; ".join(
                f'{entry["type"]} — {entry["location"]}'
                for entry in item["candidate_observations"]
            )
            row["confirmed_defects"] = "; ".join(item["confirmed_defects"])
            writer.writerow({column: row.get(column) for column in columns})

    print("\nQ-INSIGHT FACEPLATE INSPECTION REPORT")
    print("=" * 42)

    for item in report_items:
        score = item["patchcore_anomaly_score"]
        score_text = f"{score:.4f}" if score is not None else "Unavailable"

        print(f'\nImage: {item["image"]}')
        print(f"Anomaly score: {score_text}")
        print(f'Status: {item["classification"]}')

        for observation in item["candidate_observations"]:
            print(
                f'  Candidate: {observation["type"]} '
                f'| Location: {observation["location"]}'
            )

        if not item["candidate_observations"]:
            print("  Candidate: No preliminary observation available")

    print(f"\nJSON report: {json_path}")
    print(f"CSV summary: {csv_path}")
    print("\nAll classifications remain Needs Review until confirmed.")


if __name__ == "__main__":
    main()
