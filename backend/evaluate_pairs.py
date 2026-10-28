
from pathlib import Path
import csv
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GOOD_DIR = ROOT / "data" / "validation" / "good"
BAD_DIR = ROOT / "data" / "validation" / "defective"
OUT_DIR = ROOT / "data" / "pair_heatmaps"
CSV_PATH = ROOT / "data" / "paired_evaluation.csv"

SIZE = (512, 512)


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def shared_center_crop(good, bad):
    def crop(img):
        h, w = img.shape[:2]
        mx, my = int(w * 0.04), int(h * 0.04)
        return img[my:h-my or h, mx:w-mx or w]

    return crop(good), crop(bad)


def resize_pair(good, bad):
    return (
        cv2.resize(good, SIZE, interpolation=cv2.INTER_AREA),
        cv2.resize(bad, SIZE, interpolation=cv2.INTER_AREA),
    )


def alignment_gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(gray)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def overlap_fraction(mask):
    return np.count_nonzero(mask) / mask.size


def align_pair(good, bad):
    """Align the defective image to the clean reference."""
    h, w = good.shape[:2]
    good_gray = alignment_gray(good)
    bad_gray = alignment_gray(bad)

    # Method 1: ORB feature matching and homography
    orb = cv2.ORB_create(nfeatures=3500)
    kp_bad, des_bad = orb.detectAndCompute(bad_gray, None)
    kp_good, des_good = orb.detectAndCompute(good_gray, None)

    if des_bad is not None and des_good is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(des_bad, des_good, k=2)
        matches = [
            m for m, n in pairs
            if m.distance < 0.72 * n.distance
        ]

        if len(matches) >= 12:
            src = np.float32([
                kp_bad[m.queryIdx].pt for m in matches
            ]).reshape(-1, 1, 2)

            dst = np.float32([
                kp_good[m.trainIdx].pt for m in matches
            ]).reshape(-1, 1, 2)

            H, inlier_mask = cv2.findHomography(
                src, dst, cv2.RANSAC, 4.0
            )

            if H is not None and inlier_mask is not None:
                inliers = int(inlier_mask.sum())
                ratio = inliers / max(len(matches), 1)
                area_ratio = abs(np.linalg.det(H[:2, :2]))

                warped = cv2.warpPerspective(
                    bad, H, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )

                source_mask = np.full(
                    (h, w), 255, dtype=np.uint8
                )
                valid = cv2.warpPerspective(
                    source_mask, H, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT
                )
                valid = cv2.erode(
                    valid, np.ones((9, 9), np.uint8)
                )

                overlap = overlap_fraction(valid)

                if (
                    inliers >= 12
                    and ratio >= 0.25
                    and 0.35 <= area_ratio <= 2.8
                    and overlap >= 0.50
                ):
                    return (
                        warped, valid,
                        f"ORB homography ({inliers} inliers)",
                        "aligned"
                    )

    # Method 2: ECC affine alignment fallback
    warp = np.eye(2, 3, dtype=np.float32)

    try:
        score, warp = cv2.findTransformECC(
            good_gray,
            bad_gray,
            warp,
            cv2.MOTION_AFFINE,
            (
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_COUNT,
                150,
                1e-6
            ),
            None,
            5
        )

        aligned = cv2.warpAffine(
            bad, warp, (w, h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT
        )

        source_mask = np.full(
            (h, w), 255, dtype=np.uint8
        )
        valid = cv2.warpAffine(
            source_mask, warp, (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT
        )
        valid = cv2.erode(
            valid, np.ones((9, 9), np.uint8)
        )

        determinant = abs(float(np.linalg.det(warp[:, :2])))
        overlap = overlap_fraction(valid)

        if (
            score >= 0.65
            and overlap >= 0.50
            and 0.55 <= determinant <= 1.8
        ):
            return (
                aligned, valid,
                f"ECC affine ({score:.3f})",
                "aligned"
            )

    except cv2.error:
        pass

    return (
        bad,
        np.zeros((h, w), dtype=np.uint8),
        "Alignment rejected",
        "manual_review"
    )


def normalize_for_comparison(image):
    """Reduce some lighting and contrast variation."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)

    lightness = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    ).apply(lightness)

    lab = cv2.merge((lightness, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)



def compare_pair(good, aligned_bad, valid_mask):
    """
    Compare local image details while reducing broad lighting changes.
    Returns difference image, heatmap, difference score, changed fraction.
    """
    good_norm = normalize_for_comparison(good)
    bad_norm = normalize_for_comparison(aligned_bad)

    good_gray = cv2.cvtColor(good_norm, cv2.COLOR_BGR2GRAY)
    bad_gray = cv2.cvtColor(bad_norm, cv2.COLOR_BGR2GRAY)

    # Enhance local contrast before comparing details.
    clahe = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)
    )
    good_gray = clahe.apply(good_gray)
    bad_gray = clahe.apply(bad_gray)

    # Remove broad illumination variation while retaining local details.
    good_background = cv2.GaussianBlur(
        good_gray, (0, 0), sigmaX=9
    )
    bad_background = cv2.GaussianBlur(
        bad_gray, (0, 0), sigmaX=9
    )

    good_detail = cv2.subtract(good_gray, good_background)
    bad_detail = cv2.subtract(bad_gray, bad_background)

    # Compare local structures such as cracks, scratches and edges.
    difference = cv2.absdiff(good_detail, bad_detail)
    valid = valid_mask > 0

    if not np.any(valid):
        return difference, np.zeros_like(difference), 0.0, 0.0

    values = difference[valid].astype(np.float32)

    # Robust threshold: median plus three scaled median deviations.
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(10.0, median + 3.0 * 1.4826 * mad)

    changed = (difference >= threshold) & valid
    changed_fraction = (
        np.count_nonzero(changed) / np.count_nonzero(valid)
    )

    difference_score = float(np.mean(values)) / 255.0

    # Display only valid comparison areas.
    visible_diff = np.zeros_like(difference)
    visible_diff[valid] = difference[valid]

    # Normalize the heatmap for visualization only.
    heatmap = cv2.applyColorMap(
        visible_diff, cv2.COLORMAP_JET
    )
    heatmap[~valid] = 0

    # Outline regions exceeding the robust threshold.
    contours, _ = cv2.findContours(
        changed.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Ignore tiny isolated regions likely caused by noise.
    significant = [
        contour for contour in contours
        if cv2.contourArea(contour) >= 8
    ]
    cv2.drawContours(heatmap, significant, -1, (0, 255, 255), 1)

    return difference, heatmap, difference_score, changed_fraction
    

def make_panel(good, bad, difference, heatmap, valid, status):
    # Force every panel to the same dimensions and 3-channel format.
    h, w = good.shape[:2]

    def prepare(image):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h))

        return image.copy()

    good = prepare(good)
    bad = prepare(bad)
    difference = prepare(difference)
    heatmap = prepare(heatmap)

    valid = cv2.resize(
        valid, (w, h), interpolation=cv2.INTER_NEAREST
    )

    difference[valid == 0] = 0
    heatmap[valid == 0] = 0

    panels = [
        ("CLEAN REFERENCE", good),
        ("DEFECTIVE / ALIGNED", bad),
        ("NORMALIZED DIFFERENCE", difference),
        ("DIFFERENCE HEATMAP", heatmap),
    ]

    labeled = []

    for label, image in panels:
        cv2.rectangle(
            image, (0, 0), (w - 1, 34), (35, 35, 35), -1
        )
        cv2.putText(
            image, label, (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA
        )
        labeled.append(image)

    if status != "aligned":
        cv2.putText(
            labeled[-1], "MANUAL REVIEW - ALIGNMENT FAILED",
            (10, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 0, 255), 2, cv2.LINE_AA
        )

    return cv2.hconcat(labeled)



def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(1, 17):
        good_path = GOOD_DIR / f"image{i}.jpg"
        bad_path = BAD_DIR / f"image{i}.jpg"

        if not good_path.exists() or not bad_path.exists():
            print(f"image{i}.jpg: missing pair; skipped")
            continue

        try:
            good = read_image(good_path)
            bad = read_image(bad_path)

            good, bad = shared_center_crop(good, bad)
            good, bad = resize_pair(good, bad)

            aligned_bad, valid, method, status = align_pair(
                good, bad
            )

            difference, heatmap, score, changed = compare_pair(
                good, aligned_bad, valid
            )

            panel = make_panel(
                good, aligned_bad, difference,
                heatmap, valid, status
            )

            cv2.imwrite(
                str(OUT_DIR / f"pair{i}_heatmap.jpg"),
                panel
            )

            rows.append({
                "image": f"image{i}.jpg",
                "alignment_method": method,
                "difference_score": (
                    round(score, 4) if status == "aligned" else ""
                ),
                "changed_pixel_fraction": (
                    round(changed, 4) if status == "aligned" else ""
                ),
                "status": status,
            })

            if status == "aligned":
                print(
                    f"image{i}.jpg: {method}, "
                    f"difference={score:.4f}, "
                    f"changed={changed:.1%}"
                )
            else:
                print(
                    f"image{i}.jpg: Alignment rejected; "
                    "manual review"
                )

        except Exception as exc:
            print(f"image{i}.jpg: ERROR - {exc}")
            rows.append({
                "image": f"image{i}.jpg",
                "alignment_method": "error",
                "difference_score": "",
                "changed_pixel_fraction": "",
                "status": f"error: {exc}",
            })

    fields = [
        "image", "alignment_method", "difference_score",
        "changed_pixel_fraction", "status"
    ]

    with CSV_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aligned_count = sum(
        row["status"] == "aligned" for row in rows
    )

    print(f"\nPairs evaluated: {len(rows)}")
    print(
        f"Aligned: {aligned_count}; "
        f"manual review/errors: {len(rows) - aligned_count}"
    )
    print(f"CSV: {CSV_PATH}")
    print(f"Heatmaps: {OUT_DIR}")


if __name__ == "__main__":
    main()
