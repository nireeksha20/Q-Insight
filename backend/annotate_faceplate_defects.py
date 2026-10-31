
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "real_switch_pilot" / "damaged_faceplate"
OUTPUT_DIR = PROJECT_ROOT / "data" / "faceplate_annotations"

DEFECT_TYPES = [
    "scratch",
    "crack",
    "chip",
    "stain_discoloration",
    "deformation",
    "damaged_screw_hole",
    "other",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR = OUTPUT_DIR / "annotated_previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

ANNOTATION_FILE = OUTPUT_DIR / "annotations.json"
CSV_FILE = OUTPUT_DIR / "annotations.csv"


def load_existing_annotations():
    if ANNOTATION_FILE.exists():
        with ANNOTATION_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "project": "Q-Insight",
        "annotation_type": "human-reviewed bounding boxes",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "images": {},
    }


def save_annotations(data):
    data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    with ANNOTATION_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    rows = []
    for filename, record in data["images"].items():
        for defect in record.get("defects", []):
            x, y, w, h = defect["bbox_xywh"]
            rows.append({
                "image": filename,
                "image_width": record["image_width"],
                "image_height": record["image_height"],
                "defect_type": defect["type"],
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "review_status": record.get("review_status", "pending"),
                "notes": defect.get("notes", ""),
            })

    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [
            "image", "image_width", "image_height", "defect_type",
            "x", "y", "width", "height", "review_status", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_preview(image, defects):
    preview = image.copy()

    for index, defect in enumerate(defects, start=1):
        x, y, w, h = defect["bbox_xywh"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f'{index}. {defect["type"]}'
        cv2.putText(
            preview, label, (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2
        )

    return preview


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {IMAGE_DIR}")

    data = load_existing_annotations()

    print("\nQ-INSIGHT MANUAL DEFECT ANNOTATOR")
    print("=" * 38)
    print("Draw a rectangle around each visible defect.")
    print("Use the original image, not the heatmap, to select defects.")
    print("Only annotate defects you can actually see.")
    print("\nDefect labels:")
    for i, label in enumerate(DEFECT_TYPES, start=1):
        print(f"  {i}. {label}")
    print("\nIn the selection window:")
    print("  ENTER or SPACE = accept rectangle")
    print("  C = cancel current rectangle")
    print("In the terminal, enter 0 when finished with an image.")
    print("Enter 'skip' to leave an image unannotated.")
    print("Enter 'quit' to save and exit.\n")

    for image_path in image_paths:
        filename = image_path.name
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read {filename}; skipping.")
            continue

        height, width = image.shape[:2]
        existing = data["images"].get(filename, {})
        defects = list(existing.get("defects", []))

        print(f"\nImage: {filename} ({width} x {height})")
        print(f"Existing boxes: {len(defects)}")

        while True:
            preview = draw_preview(image, defects)
            display = preview.copy()

            instruction = "Select a defect region | close window to finish"
            cv2.putText(
                display, instruction, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )

            cv2.imshow("Q-Insight - Faceplate Annotation", display)
            print("\nDraw a rectangle in the image window.")
            print("Close the window without selecting to finish this image.")

            roi = cv2.selectROI(
                "Q-Insight - Faceplate Annotation",
                display,
                showCrosshair=True,
                fromCenter=False,
            )
            cv2.destroyAllWindows()

            x, y, w, h = [int(v) for v in roi]

            if w <= 0 or h <= 0:
                break

            print("\nChoose the best matching defect type:")
            for i, label in enumerate(DEFECT_TYPES, start=1):
                print(f"  {i}. {label}")

            choice = input("Defect number (1-7), or 0 to cancel this box: ").strip()

            if choice == "0":
                continue

            if not choice.isdigit() or not 1 <= int(choice) <= len(DEFECT_TYPES):
                print("Invalid choice; box discarded.")
                continue

            defect_type = DEFECT_TYPES[int(choice) - 1]
            notes = input("Optional note (press Enter to skip): ").strip()

            defects.append({
                "type": defect_type,
                "bbox_xywh": [x, y, w, h],
                "notes": notes,
            })

            print(f"Added {defect_type} annotation.")

        cv2.destroyAllWindows()

        decision = input(
            "Review decision: [d] defective, [g] good, [u] uncertain, "
            "[s] skip image, [q] save and quit: "
        ).strip().lower()

        if decision == "q":
            save_annotations(data)
            print(f"Saved annotations to {ANNOTATION_FILE}")
            break

        if decision == "s":
            continue

        status_map = {
            "d": "defective",
            "g": "good",
            "u": "uncertain",
        }

        if decision not in status_map:
            print("Unrecognized decision; marking image uncertain.")
            decision = "u"

        if decision == "g" and defects:
            print(
                "Warning: this image has defect boxes but was marked good. "
                "Please resolve this inconsistency before model training."
            )

        data["images"][filename] = {
            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            "image_width": width,
            "image_height": height,
            "review_status": status_map[decision],
            "defects": defects,
        }

        save_annotations(data)

        preview = draw_preview(image, defects)
        preview_path = PREVIEW_DIR / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(preview_path), preview)

        print(f"Saved annotations and preview for {filename}.")

    save_annotations(data)
    print("\nAnnotation session complete.")
    print(f"JSON: {ANNOTATION_FILE}")
    print(f"CSV: {CSV_FILE}")
    print(f"Previews: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
