from pathlib import Path
import csv
import cv2
import numpy as np

from main import (
    read_image,
    crop_product_region,
    align_reference_to_image,
)

BASE_DIR = Path(__file__).resolve().parent.parent
GOOD_DIR = BASE_DIR / "data" / "validation" / "good"
DEFECTIVE_DIR = BASE_DIR / "data" / "validation" / "defective"
OUTPUT_FILE = BASE_DIR / "data" / "paired_inspection_diagnostic.csv"


def compare_images(reference_path, test_path):
    reference = read_image(reference_path)
    image = read_image(test_path)

    if reference is None or image is None:
        return None, "Unreadable image"

    reference = crop_product_region(reference)
    image = crop_product_region(image)

    aligned_reference = align_reference_to_image(reference, image)

    # Resize only if the alignment function returns different dimensions.
    if aligned_reference.shape[:2] != image.shape[:2]:
        aligned_reference = cv2.resize(
            aligned_reference,
            (image.shape[1], image.shape[0]),
        )

    ref_gray = cv2.cvtColor(aligned_reference, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    score = float(np.mean(cv2.absdiff(img_gray, ref_gray)) / 255.0)

    return round(score, 5), "Compared"


def main():
    rows = []

    for index in range(1, 17):
        good_path = GOOD_DIR / f"image{index}.jpg"
        defective_path = DEFECTIVE_DIR / f"image{index}.jpg"

        if not good_path.exists() or not defective_path.exists():
            print(f"Pair {index}: missing image; skipped")
            continue

        score, status = compare_images(good_path, defective_path)

        rows.append({
            "pair": index,
            "reference": good_path.name,
            "test_image": defective_path.name,
            "reference_label": "good",
            "test_label": "defective",
            "difference_score": score,
            "status": status,
        })

        if score is None:
            print(f"Pair {index}: {status}")
        else:
            print(
                f"Pair {index}: difference score={score:.5f} "
                f"({status})"
            )

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pair",
                "reference",
                "test_image",
                "reference_label",
                "test_label",
                "difference_score",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    valid_scores = [
        row["difference_score"]
        for row in rows
        if row["difference_score"] is not None
    ]

    print(f"\nPairs compared: {len(valid_scores)}")
    if valid_scores:
        print(f"Mean paired difference: {np.mean(valid_scores):.5f}")
        print(f"Median paired difference: {np.median(valid_scores):.5f}")

    print(f"CSV saved to: {OUTPUT_FILE}")
    print(
        "\nIMPORTANT: These are paired visual-difference scores, "
        "not classification accuracy or defect probabilities."
    )


if __name__ == "__main__":
    main()