
import csv
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "anomalib_dataset_audit_v2_manifest.csv"
OUTPUT = ROOT / "data" / "training_image_review.csv"

rows = []

with MANIFEST.open("r", newline="", encoding="utf-8-sig") as file:
    for row in csv.DictReader(file):
        if row["label"] != "good":
            continue

        image_path = ROOT / row["destination"]
        exists = image_path.is_file()
        width = height = 0
        readable = False

        if exists:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image.verify()
                readable = True
            except Exception:
                readable = False

        # This is a review flag, not an automatic defect label.
        if not readable:
            review = "CHECK_FILE"
        elif min(width, height) < 150:
            review = "REVIEW_LOW_RESOLUTION"
        else:
            review = "MANUAL_VISUAL_REVIEW"

        rows.append({
            "pair_number": row["pair_number"],
            "source": row["source"],
            "destination": row["destination"],
            "width": width,
            "height": height,
            "file_readable": "Yes" if readable else "No",
            "review_status": review,
            "confirmed_normal": "No",
        })

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=[
        "pair_number", "source", "destination",
        "width", "height", "file_readable",
        "review_status", "confirmed_normal",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Training images audited: {len(rows)}")
print(f"Report saved: {OUTPUT}")
print("No images or model checkpoints were modified.")
