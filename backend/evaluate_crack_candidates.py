
from pathlib import Path
import cv2
import numpy as np

from main import read_image

BASE = Path(__file__).resolve().parent.parent
DEFECTIVE = BASE / "data" / "validation" / "defective"
GOOD = BASE / "data" / "validation" / "good"
OUT = BASE / "data" / "crack_candidates"
OUT.mkdir(parents=True, exist_ok=True)


def detect_line_candidates(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve local contrast before looking for dark line-like features.
    gray = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(gray)

    responses = []

    # Detect narrow dark features in horizontal, vertical and diagonal directions.
    for length, width in [(11, 3), (17, 3), (25, 3), (3, 11), (3, 17)]:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (length, width)
        )
        response = cv2.morphologyEx(
            gray, cv2.MORPH_BLACKHAT, kernel
        )
        responses.append(response)

    combined = np.maximum.reduce(responses)

    # Keep unusually strong responses; this is a visualization threshold,
    # not a calibrated defect threshold.
    threshold = max(
        18, int(np.percentile(combined, 92))
    )
    mask = (combined >= threshold).astype(np.uint8) * 255

    # Ignore the outer image border, where background edges dominate.
    h, w = mask.shape
    border_y = max(2, int(h * 0.025))
    border_x = max(2, int(w * 0.025))
    mask[:border_y, :] = 0
    mask[-border_y:, :] = 0
    mask[:, :border_x] = 0
    mask[:, -border_x:] = 0

    # Remove tiny responses, retaining elongated connected regions.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    filtered = np.zeros_like(mask)

    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        major = max(bw, bh)
        minor = max(1, min(bw, bh))

        if major >= 12 and area >= 10 and major / minor >= 2.0:
            filtered[labels == i] = 255

    return filtered


def save_overlay(image, mask, path):
    overlay = image.copy()

    # Mark candidate pixels in red for easy inspection.
    overlay[mask > 0] = (
        0.35 * overlay[mask > 0]
        + 0.65 * np.array([0, 0, 255])
    ).astype(np.uint8)

    cv2.imwrite(str(path), overlay)


def main():
    processed = 0

    for i in range(1, 17):
        bad_path = DEFECTIVE / f"image{i}.jpg"
        good_path = GOOD / f"image{i}.jpg"

        for label, path in [("DEFECTIVE", bad_path), ("GOOD", good_path)]:
            if not path.exists():
                print(f"Missing: {path}")
                continue

            image = read_image(path)
            if image is None:
                print(f"Unreadable: {path}")
                continue

            mask = detect_line_candidates(image)
            output = OUT / f"image{i}_{label.lower()}.jpg"
            save_overlay(image, mask, output)

            fraction = float(np.count_nonzero(mask)) / mask.size
            print(
                f"Pair {i} {label}: "
                f"candidate_pixels={fraction:.2%}, saved={output.name}"
            )
            processed += 1

    print(f"\nImages processed: {processed}")
    print(f"Overlays saved in: {OUT}")
    print(
        "Warning: socket openings, screw holes, shadows and printed "
        "details may also be highlighted. Review overlays visually."
    )


if __name__ == "__main__":
    main()
