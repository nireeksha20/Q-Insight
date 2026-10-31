
import csv
from collections import defaultdict
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

DATASETS = {
    "v2_train_good": ROOT / "data/anomalib_dataset_audit_v2/train/good",
    "v2_test_good": ROOT / "data/anomalib_dataset_audit_v2/test/good",
    "v2_test_defective": ROOT / "data/anomalib_dataset_audit_v2/test/defective",
}

OUTPUT = ROOT / "data" / "resolution_comparison.csv"


def inspect_folder(name, folder):
    records = []

    for path in sorted(folder.glob("*")):
        if not path.is_file():
            continue

        try:
            with Image.open(path) as image:
                width, height = image.size
                records.append({
                    "dataset": name,
                    "image": path.name,
                    "width": width,
                    "height": height,
                    "pixels": width * height,
                    "aspect_ratio": round(width / height, 3),
                })
        except Exception as exc:
            print(f"Cannot read {path}: {exc}")

    return records


all_records = []

for name, folder in DATASETS.items():
    if not folder.exists():
        print(f"Missing folder: {folder}")
        continue

    records = inspect_folder(name, folder)
    all_records.extend(records)

    pixels = sorted(row["pixels"] for row in records)
    print(f"\n{name}: {len(records)} images")

    if pixels:
        print(f"  Smallest: {pixels[0]:,} pixels")
        print(f"  Largest:  {pixels[-1]:,} pixels")
        print(f"  Median:   {pixels[len(pixels) // 2]:,} pixels")

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "dataset", "image", "width", "height",
            "pixels", "aspect_ratio",
        ],
    )
    writer.writeheader()
    writer.writerows(all_records)

print(f"\nSaved report: {OUTPUT}")
print("No images or checkpoints were modified.")
