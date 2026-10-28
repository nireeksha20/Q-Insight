
from pathlib import Path
import csv
import cv2

from main import inspect_image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GOOD_DIR = DATA_DIR / "validation" / "good"
BAD_DIR = DATA_DIR / "validation" / "defective"
OUTPUT_CSV = DATA_DIR / "inspection_predictions.csv"

# Exploratory threshold only. Do not treat this as a validated setting.
CANDIDATE_THRESHOLD = 0.03


def main():
    rows = []

    datasets = [
        (GOOD_DIR, "good"),
        (BAD_DIR, "defective"),
    ]

    for folder, label in datasets:
        for image_path in sorted(folder.glob("*")):
            if image_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }:
                continue

            try:
                # Uses the existing application's inspection function.
                result = inspect_image(image_path)

                if not isinstance(result, dict):
                    print(
                        f"{image_path.name}: inspect_image returned "
                        "an unexpected result; check main.py"
                    )
                    continue

                score = result.get("anomaly_score")
                if score is None:
                    score = result.get("difference_score")

                if score is None:
                    print(
                        f"{image_path.name}: no numeric anomaly score "
                        "found in inspect_image result"
                    )
                    continue

                score = float(score)
                prediction = (
                    "defective"
                    if score >= CANDIDATE_THRESHOLD
                    else "good"
                )

                rows.append({
                    "image": image_path.name,
                    "dataset": label,
                    "ground_truth": label,
                    "score": round(score, 6),
                    "threshold": CANDIDATE_THRESHOLD,
                    "prediction": prediction,
                    "correct": prediction == label,
                })

                print(
                    f"{label}/{image_path.name}: score={score:.5f}, "
                    f"prediction={prediction}"
                )

            except Exception as exc:
                print(f"{image_path}: ERROR - {exc}")

    fields = [
        "image", "dataset", "ground_truth", "score",
        "threshold", "prediction", "correct"
    ]

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPredictions saved to: {OUTPUT_CSV}")
    print(f"Images scored: {len(rows)}")
    print(
        "\nIMPORTANT: This uses a candidate threshold and the existing "
        "reference-based inspection function. Verify its score definition "
        "before interpreting these predictions as accuracy."
    )


if __name__ == "__main__":
    main()
