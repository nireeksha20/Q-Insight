
import csv
import sys
from pathlib import Path

from main import inspect_image

BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATION_DIR = BASE_DIR / "data" / "validation"
OUTPUT_FILE = BASE_DIR / "data" / "validation_results.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    rows = []

    for actual_label in ("good", "defective"):
        folder = VALIDATION_DIR / actual_label

        if not folder.is_dir():
            print(f"Missing folder: {folder}")
            sys.exit(1)

        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        print(f"{actual_label}: found {len(images)} images")

        for image_path in images:
            try:
                result = inspect_image(image_path)

                rows.append({
                    "filename": image_path.name,
                    "actual_label": actual_label,
                    "inspection_status": result.get(
                        "inspection_status"
                    ),
                    "defect_label": result.get("defect_label"),
                    "anomaly_score": result.get("anomaly_score"),
                    "similarity_score": result.get(
                        "similarity_score"
                    ),
                    "explanation": result.get("explanation"),
                })

                print(
                    f"{image_path.name}: "
                    f"{result.get('inspection_status')} | "
                    f"anomaly={result.get('anomaly_score')}"
                )

            except Exception as exc:
                print(f"ERROR processing {image_path.name}: {exc}")
                rows.append({
                    "filename": image_path.name,
                    "actual_label": actual_label,
                    "inspection_status": "Processing error",
                    "defect_label": "",
                    "anomaly_score": "",
                    "similarity_score": "",
                    "explanation": str(exc),
                })

    if not rows:
        print("No images found to evaluate.")
        return

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nEvaluation report saved to: {OUTPUT_FILE}")
    print(f"Images processed: {len(rows)}")


if __name__ == "__main__":
    main()
