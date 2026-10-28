
from pathlib import Path
import csv
import cv2
import numpy as np

from main import read_image

BASE = Path(__file__).resolve().parent.parent
GOOD = BASE / "data" / "validation" / "good"
BAD = BASE / "data" / "validation" / "defective"
OUT = BASE / "data" / "approved_pair_evaluation.csv"
HEATMAP_DIR = BASE / "data" / "approved_pair_heatmaps"

# Initial visual-review candidates; not yet ground-truth-approved.
CANDIDATE_PAIRS = [1, 2, 5, 7, 8, 9, 10, 11, 13]


def align_pair(reference, image):
    """Align only when a plausible homography has enough inliers."""
    h, w = image.shape[:2]
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)

    if des1 is None or des2 is None:
        return None, 0, 0.0

    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        des1, des2, k=2
    )
    good = [
        m for pair in matches if len(pair) == 2
        for m, n in [pair] if m.distance < 0.75 * n.distance
    ]

    if len(good) < 12:
        return None, len(good), 0.0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if matrix is None or inliers is None:
        return None, len(good), 0.0

    inlier_count = int(inliers.sum())
    ratio = inlier_count / len(good)

    # Reject weak estimates; never silently resize and call it aligned.
    if inlier_count < 10 or ratio < 0.35:
        return None, inlier_count, ratio

    # Reject transformations with non-finite values.
    if not np.isfinite(matrix).all():
        return None, inlier_count, ratio

    corners = np.float32([
        [0, 0], [reference.shape[1] - 1, 0],
        [reference.shape[1] - 1, reference.shape[0] - 1],
        [0, reference.shape[0] - 1]
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(corners, matrix)
    if not np.isfinite(projected).all():
        return None, inlier_count, ratio

    aligned = cv2.warpPerspective(reference, matrix, (w, h))
    return aligned, inlier_count, ratio


def compare_images(good, defective):
    """Compare aligned product images using illumination-normalized edges."""
    target_size = (512, 512)
    good = cv2.resize(good, target_size)
    defective = cv2.resize(defective, target_size)

    aligned, inliers, ratio = align_pair(good, defective)
    if aligned is None:
        return None, inliers, ratio, None

    good_gray = cv2.cvtColor(good, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    # Local contrast normalization reduces some illumination variation.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    good_norm = clahe.apply(good_gray)
    bad_norm = clahe.apply(aligned_gray)

    difference = cv2.absdiff(good_norm, bad_norm)
    difference = cv2.GaussianBlur(difference, (5, 5), 0)

    score = float(np.mean(difference) / 255.0)

    heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(defective, 0.65, heatmap, 0.35, 0)

    return score, inliers, ratio, overlay


def main():
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for pair_id in CANDIDATE_PAIRS:
        good_path = GOOD / f"image{pair_id}.jpg"
        bad_path = BAD / f"image{pair_id}.jpg"

        good = read_image(good_path)
        bad = read_image(bad_path)

        if good is None or bad is None:
            print(f"Pair {pair_id}: missing or unreadable image")
            continue

        score, inliers, ratio, overlay = compare_images(good, bad)

        if score is None:
            status = "MANUAL_REVIEW_ALIGNMENT_FAILED"
            score_text = ""
            print(
                f"Pair {pair_id}: MANUAL REVIEW "
                f"(inliers={inliers}, ratio={ratio:.2f})"
            )
        else:
            status = "VISUAL_DIFFERENCE_MEASURED"
            score_text = round(score, 5)
            output = HEATMAP_DIR / f"pair{pair_id}_heatmap.jpg"
            cv2.imwrite(str(output), overlay)
            print(
                f"Pair {pair_id}: difference={score:.5f}, "
                f"inliers={inliers}, ratio={ratio:.2f}"
            )

        rows.append({
            "pair": pair_id,
            "difference_score": score_text,
            "homography_inliers": inliers,
            "inlier_ratio": round(ratio, 4),
            "status": status,
        })

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair", "difference_score", "homography_inliers",
                "inlier_ratio", "status"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    measured = [r for r in rows if r["difference_score"] != ""]
    print(f"\nCandidate pairs: {len(rows)}")
    print(f"Pairs with measurable differences: {len(measured)}")
    print(f"Pairs requiring manual review: {len(rows) - len(measured)}")
    print(f"CSV: {OUT}")
    print(f"Heatmaps: {HEATMAP_DIR}")
    print("Scores measure visual differences, not defect probabilities or accuracy.")


if __name__ == "__main__":
    main()
