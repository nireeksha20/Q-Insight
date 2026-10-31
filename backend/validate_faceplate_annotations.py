"""
Q-Insight faceplate annotation validator and artifact synchronizer.

Run from the backend directory:
    python validate_faceplate_annotations.py
    python validate_faceplate_annotations.py --sync

Default mode is read-only. --sync regenerates the CSV and annotated previews
from annotations.json, but never changes annotations.json or model files.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "real_switch_pilot" / "damaged_faceplate"
ANNOTATION_DIR = DATA_DIR / "faceplate_annotations"
JSON_PATH = ANNOTATION_DIR / "annotations.json"
CSV_PATH = ANNOTATION_DIR / "annotations.csv"
PREVIEW_DIR = ANNOTATION_DIR / "annotated_previews"

ALLOWED_STATUSES = {"good", "defective", "uncertain"}
ALLOWED_LABELS = {
    "scratch", "crack", "chip", "stain_discoloration",
    "deformation", "damaged_screw_hole", "other",
}


def load_data():
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {JSON_PATH}")
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def find_image(filename):
    candidate = IMAGE_DIR / filename
    if candidate.is_file():
        return candidate
    matches = [p for p in IMAGE_DIR.iterdir()
               if p.is_file() and p.name.lower() == filename.lower()]
    return matches[0] if len(matches) == 1 else None


def box_xyxy(box):
    x, y, w, h = box["bbox_xywh"]
    return x, y, x + w, y + h


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_xyxy(box_a)
    bx1, by1, bx2, by2 = box_xyxy(box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def near_duplicate(box_a, box_b, iou_threshold=0.30):
    a, b = box_a.get("bbox_xywh", []), box_b.get("bbox_xywh", [])
    if len(a) != 4 or len(b) != 4:
        return False
    if iou(box_a, box_b) >= iou_threshold:
        return True
    max_dim = max(a[2], a[3], b[2], b[3], 1)
    deltas = [abs(float(x) - float(y)) for x, y in zip(a, b)]
    return max(deltas) <= max(5, 0.02 * max_dim)


def validate(data):
    errors, warnings = [], []
    images = data.get("images")
    if not isinstance(images, dict):
        return ["JSON must contain an 'images' object."], warnings

    print("Q-INSIGHT ANNOTATION VALIDATION")
    print("=" * 68)
    print(f"Annotation records: {len(images)}")
    print(f"Source image directory: {IMAGE_DIR}")

    for filename, record in images.items():
        print(f"\n{filename}")
        if not isinstance(record, dict):
            errors.append(f"{filename}: image record must be an object.")
            continue

        image_path = find_image(filename)
        if image_path is None:
            errors.append(f"{filename}: source image missing or ambiguous.")
            print("  ERROR: source image missing or ambiguous")
            continue

        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except Exception as exc:
            errors.append(f"{filename}: cannot open source image: {exc}")
            continue

        status = record.get("review_status")
        defects = record.get("defects", [])
        if status not in ALLOWED_STATUSES:
            errors.append(f"{filename}: invalid review_status {status!r}.")
        if not isinstance(defects, list):
            errors.append(f"{filename}: 'defects' must be a list.")
            continue

        print(f"  dimensions={width}x{height} | status={status} | boxes={len(defects)}")
        if status == "good" and defects:
            warnings.append(f"{filename}: status 'good' but {len(defects)} defect box(es) exist.")
        if status == "defective" and not defects:
            warnings.append(f"{filename}: status 'defective' but no defect boxes exist.")
        if status == "uncertain":
            warnings.append(f"{filename}: uncertain image; exclude from confirmed-label evaluation.")

        for index, box in enumerate(defects):
            prefix = f"{filename}, box {index + 1}"
            if not isinstance(box, dict):
                errors.append(f"{prefix}: annotation must be an object.")
                continue
            if box.get("type") not in ALLOWED_LABELS:
                errors.append(f"{prefix}: invalid or missing defect type {box.get('type')!r}.")
            coords = box.get("bbox_xywh")
            if not isinstance(coords, list) or len(coords) != 4:
                errors.append(f"{prefix}: bbox_xywh must contain [x, y, width, height].")
                continue
            try:
                x, y, w, h = [float(v) for v in coords]
            except (TypeError, ValueError):
                errors.append(f"{prefix}: bbox coordinates must be numeric.")
                continue
            if not all(v == v and abs(v) != float("inf") for v in (x, y, w, h)):
                errors.append(f"{prefix}: bbox contains non-finite values.")
            elif x < 0 or y < 0 or w <= 0 or h <= 0:
                errors.append(f"{prefix}: negative origin or non-positive size.")
            elif x + w > width or y + h > height:
                errors.append(f"{prefix}: bbox exceeds image bounds ({width}x{height}): {coords}.")

        for i in range(len(defects)):
            for j in range(i + 1, len(defects)):
                a, b = defects[i], defects[j]
                if not isinstance(a, dict) or not isinstance(b, dict):
                    continue
                try:
                    if near_duplicate(a, b):
                        warnings.append(
                            f"{filename}: boxes {i+1} and {j+1} may be duplicates "
                            f"(IoU={iou(a,b):.2f}; labels={a.get('type')}/{b.get('type')})."
                        )
                except (TypeError, ValueError, KeyError, IndexError):
                    pass

    print("\n" + "=" * 68)
    print(f"Errors: {len(errors)} | Warnings: {len(warnings)}")
    for item in errors:
        print("ERROR:", item)
    for item in warnings:
        print("REVIEW:", item)
    return errors, warnings

def synchronize_artifacts(data):
    ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for filename, record in data.get("images", {}).items():
        defects = record.get("defects", [])
        if not defects:
            rows.append({
                "image": filename, "review_status": record.get("review_status", ""),
                "defect_index": "", "defect_type": "", "x": "", "y": "",
                "width": "", "height": "", "notes": "",
            })
        for idx, defect in enumerate(defects, 1):
            bbox = defect.get("bbox_xywh", ["", "", "", ""])
            if not isinstance(bbox, list) or len(bbox) != 4:
                bbox = ["", "", "", ""]
            rows.append({
                "image": filename, "review_status": record.get("review_status", ""),
                "defect_index": idx, "defect_type": defect.get("type", ""),
                "x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3],
                "notes": defect.get("notes", ""),
            })

    fields = ["image", "review_status", "defect_index", "defect_type",
              "x", "y", "width", "height", "notes"]
    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for filename, record in data.get("images", {}).items():
        image_path = find_image(filename)
        if image_path is None:
            print(f"Skipping preview for missing image: {filename}")
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {filename}")
            continue

        for idx, defect in enumerate(record.get("defects", []), 1):
            bbox = defect.get("bbox_xywh", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x, y, w, h = [int(round(float(v))) for v in bbox]
            except (TypeError, ValueError):
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(image.shape[1] - 1, x + w), min(image.shape[0] - 1, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            label = str(defect.get("type", "unknown"))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, f"{idx}. {label}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        output = PREVIEW_DIR / (Path(filename).stem + "_annotated.jpg")
        if not cv2.imwrite(str(output), image):
            print(f"Could not write preview: {output}")
        else:
            print(f"Preview updated: {output.name}")

    print(f"\nCSV regenerated: {CSV_PATH}")
    print(f"Previews regenerated in: {PREVIEW_DIR}")
    print("annotations.json was not modified.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true",
                        help="Regenerate CSV and previews from JSON after validation.")
    args = parser.parse_args()
    try:
        data = load_data()
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot load annotations: {exc}")

    errors, _warnings = validate(data)
    if args.sync:
        if errors:
            print("\nNot synchronizing because validation errors exist.")
            raise SystemExit(1)
        synchronize_artifacts(data)
    else:
        print("\nRead-only validation finished; no files were modified.")
        print("After reviewing warnings, run: python validate_faceplate_annotations.py --sync")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
