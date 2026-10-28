
from pathlib import Path
import csv
import cv2
import numpy as np

from main import read_image, crop_product_region

BASE_DIR = Path(__file__).resolve().parent.parent
GOOD_DIR = BASE_DIR / "data" / "validation" / "good"
BAD_DIR = BASE_DIR / "data" / "validation" / "defective"
OUT_FILE = BASE_DIR / "data" / "alignment_diagnostic.csv"


def orb_alignment_metrics(reference, image):
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1500)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None:
        return len(kp1), len(kp2), 0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = [
        m for pair in matches if len(pair) == 2
        for m, n in [pair] if m.distance < 0.75 * n.distance
    ]

    if len(good) < 4:
        return len(kp1), len(kp2), len(good), 0.0

    src = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)
    dst = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    _, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    inlier_ratio = inlier_count / len(good) if good else 0.0

    return len(kp1), len(kp2), inlier_count, round(inlier_ratio, 4)


def main():
    rows = []

    for i in range(1, 17):
        good_path = GOOD_DIR / f"image{i}.jpg"
        bad_path = BAD_DIR / f"image{i}.jpg"

        if not good_path.exists() or not bad_path.exists():
            print(f"Pair {i}: missing image")
            continue

        good = read_image(good_path)
        bad = read_image(bad_path)

        if good is None or bad is None:
            print(f"Pair {i}: unreadable image")
            continue

        good_crop = crop_product_region(good)
        bad_crop = crop_product_region(bad)

        metrics = orb_alignment_metrics(good_crop, bad_crop)
        kp_ref, kp_test, inliers, ratio = metrics

        row = {
            "pair": i,
            "good_original_size": f"{good.shape[1]}x{good.shape[0]}",
            "defective_original_size": f"{bad.shape[1]}x{bad.shape[0]}",
            "good_crop_size": f"{good_crop.shape[1]}x{good_crop.shape[0]}",
            "defective_crop_size": f"{bad_crop.shape[1]}x{bad_crop.shape[0]}",
            "good_keypoints": kp_ref,
            "defective_keypoints": kp_test,
            "homography_inliers": inliers,
            "inlier_ratio": ratio,
        }
        rows.append(row)

        print(
            f"Pair {i}: keypoints={kp_ref}/{kp_test}, "
            f"inliers={inliers}, inlier_ratio={ratio:.2f}, "
            f"crop_sizes={row['good_crop_size']}/{row['defective_crop_size']}"
        )

    fields = [
        "pair", "good_original_size", "defective_original_size",
        "good_crop_size", "defective_crop_size", "good_keypoints",
        "defective_keypoints", "homography_inliers", "inlier_ratio",
    ]

    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPairs diagnosed: {len(rows)}")
    print(f"CSV saved to: {OUT_FILE}")
    print(
        "Note: feature-match metrics are alignment diagnostics, "
        "not defect-detection accuracy."
    )


if __name__ == "__main__":
    main()
