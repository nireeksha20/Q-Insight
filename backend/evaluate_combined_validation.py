
from pathlib import Path
import csv

import cv2
from main import inspect_image, DATA_DIR

ROOT = DATA_DIR / "validation"
OUTPUT = DATA_DIR / "combined_validation_results.csv"

rows = []

for label in ("good", "defective"):
    folder = ROOT / label

    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue

        result = inspect_image(path)

        row = {
            "filename": path.name,
            "actual_label": label,
            "patchcore_score": result["anomaly_score"],
            "discoloration_score": result["discoloration_score"],
            "predicted_label": (
                "defective"
                if result["inspection_status"] == "Review required"
                else "good"
            ),
        }
        rows.append(row)

        print(
            f'{path.name:24} actual={label:10} '
            f'PatchCore={row["patchcore_score"]:.2f} '
            f'discoloration={row["discoloration_score"]:.2f} '
            f'predicted={row["predicted_label"]}'
        )

if not rows:
    raise RuntimeError("No validation images were evaluated.")

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

for label in ("good", "defective"):
    subset = [r for r in rows if r["actual_label"] == label]
    correct = sum(r["actual_label"] == r["predicted_label"] for r in subset)
    print(f"\n{label.upper()}: {correct}/{len(subset)} classified correctly")

print(f"\nTotal evaluated: {len(rows)}")
print(f"Results saved to: {OUTPUT}")
