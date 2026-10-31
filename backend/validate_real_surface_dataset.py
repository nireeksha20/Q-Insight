
from pathlib import Path
import csv
import hashlib
from collections import defaultdict

import cv2


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "real_surface_pilot" / "good_online"
REPORT_DIR = PROJECT_ROOT / "data" / "real_surface_pilot" / "validation"
CSV_PATH = REPORT_DIR / "good_online_validation.csv"

# Minimum dimensions for a useful initial inspection image.
# This is a screening rule, not a guarantee of defect visibility.
MIN_WIDTH = 300
MIN_HEIGHT = 300

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    print("=" * 72)
    print("Q-INSIGHT: READ-ONLY IMAGE DATASET VALIDATION")
    print("=" * 72)
    print(f"Image folder: {IMAGE_DIR}")

    if not IMAGE_DIR.exists():
        print("\nERROR: Image folder does not exist.")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print("\nNo supported image files found.")
        return

    hash_to_names = defaultdict(list)
    rows = []

    for path in files:
        row = {
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "width": "",
            "height": "",
            "channels": "",
            "sha256": "",
            "readable": False,
            "exact_duplicate": False,
            "duplicate_of": "",
            "dimension_status": "",
            "recommendation": "",
        }

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if image is None:
            row["dimension_status"] = "UNREADABLE"
            row["recommendation"] = "Check file format or file integrity"
            rows.append(row)
            print(f"[ERROR] {path.name}: image cannot be opened")
            continue

        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]

        file_hash = sha256_file(path)
        hash_to_names[file_hash].append(path.name)

        row.update({
            "width": width,
            "height": height,
            "channels": channels,
            "sha256": file_hash,
            "readable": True,
        })

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            row["dimension_status"] = "LOW_RESOLUTION"
            row["recommendation"] = (
                "Keep for pipeline testing; prefer a larger image "
                "for defect evaluation"
            )
        else:
            row["dimension_status"] = "DIMENSIONS_OK"
            row["recommendation"] = (
                "Review visually; dimensions alone do not prove suitability"
            )

        rows.append(row)

    # Mark exact duplicate files by content hash.
    for row in rows:
        file_hash = row["sha256"]

        if file_hash and len(hash_to_names[file_hash]) > 1:
            duplicate_names = hash_to_names[file_hash]
            first_name = duplicate_names[0]

            if row["filename"] != first_name:
                row["exact_duplicate"] = True
                row["duplicate_of"] = first_name
                row["recommendation"] = (
                    "Exact duplicate; keep unchanged until manual review"
                )

    columns = [
        "filename",
        "extension",
        "size_bytes",
        "width",
        "height",
        "channels",
        "sha256",
        "readable",
        "exact_duplicate",
        "duplicate_of",
        "dimension_status",
        "recommendation",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    readable_count = sum(row["readable"] for row in rows)
    unreadable_count = len(rows) - readable_count
    low_resolution_count = sum(
        row["dimension_status"] == "LOW_RESOLUTION" for row in rows
    )
    duplicate_count = sum(row["exact_duplicate"] for row in rows)

    print("\nVALIDATION SUMMARY")
    print("-" * 72)
    print(f"Files found:          {len(rows)}")
    print(f"Readable images:      {readable_count}")
    print(f"Unreadable images:    {unreadable_count}")
    print(f"Low-resolution files: {low_resolution_count}")
    print(f"Exact duplicate files:{duplicate_count:>3}")

    print("\nIMAGE DETAILS")
    print("-" * 72)

    for row in rows:
        if row["readable"]:
            dimensions = f'{row["width"]}x{row["height"]}'
        else:
            dimensions = "unreadable"

        tags = []
        if row["dimension_status"] == "LOW_RESOLUTION":
            tags.append("LOW RESOLUTION")
        if row["exact_duplicate"]:
            tags.append(f'DUPLICATE OF {row["duplicate_of"]}')
        if not row["readable"]:
            tags.append("UNREADABLE")

        suffix = f" | {', '.join(tags)}" if tags else ""
        print(f'{row["filename"]:<28} {dimensions:<14}{suffix}')

    print(f"\nCSV report saved to:\n{CSV_PATH}")
    print("\nNo source images, annotations, or model files were modified.")
    print("Note: perceptually similar images are not detected by exact hashes.")


if __name__ == "__main__":
    main()
