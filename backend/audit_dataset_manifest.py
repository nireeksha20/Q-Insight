
import csv
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GOOD_DIR = ROOT / "data" / "validation" / "good"
DEFECTIVE_DIR = ROOT / "data" / "validation" / "defective"
OUTPUT = ROOT / "data" / "dataset_audit_manifest.csv"

# Decisions from our visual review of the comparison sheets.
PAIR_DECISIONS = {
    1: ("Keep candidate", "Cracked plate and damaged screw areas; close visual match."),
    2: ("Keep candidate", "Visible cracking, wear, and damage to the mechanism."),
    3: ("Keep candidate", "Prominent cracks, dark marks, and damaged socket components."),
    4: ("Review", "Aged plates and exposed wiring; construction differences need review."),
    5: ("Review", "Similar outlet layout, but low resolution and appearance differences."),
    6: ("Exclude from paired evaluation", "Different switch plate designs."),
    7: ("Exclude from paired evaluation", "Plate color and lighting differ; defect is uncertain."),
    8: ("Exclude from paired evaluation", "Possible damage, but product and viewpoint are uncertain."),
    9: ("Review", "Visible plate damage, but framing and plate proportions differ."),
    10: ("Keep candidate", "Similar horizontal outlet layout and background; confirm defect label."),
    11: ("Keep candidate", "Cover is missing and the outlet mechanism is exposed."),
    12: ("Review", "Similar switch layout, but plate color and appearance differ."),
    13: ("Exclude from paired evaluation", "Very small images; defect cannot be assessed confidently."),
    14: ("Exclude from paired evaluation", "Plain plate versus plate with a switch; configuration differs."),
    15: ("Exclude from paired evaluation", "Different plate appearance; defect is uncertain."),
    16: ("Keep candidate", "Similar double-switch design; damaged or separated edge is visible."),
}


def inspect_image(path):
    """Return image dimensions, or an error if the file cannot be read."""
    if not path.exists():
        return "", "", "Missing file"

    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        return width, height, ""
    except Exception as exc:
        return "", "", f"Unreadable image: {exc}"


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for pair_number in range(1, 17):
        decision, notes = PAIR_DECISIONS[pair_number]

        for label, directory in [
            ("good", GOOD_DIR),
            ("defective", DEFECTIVE_DIR),
        ]:
            path = directory / f"image{pair_number}.jpg"
            width, height, error = inspect_image(path)

            rows.append({
                "pair_number": pair_number,
                "label_from_folder": label,
                "filename": path.name,
                "relative_path": str(path.relative_to(ROOT)),
                "width": width,
                "height": height,
                "pair_review_decision": decision,
                "review_notes": notes,
                "label_independently_verified": "No",
                "file_status": error or "OK",
                "original_file_modified": "No",
            })

    fieldnames = [
        "pair_number",
        "label_from_folder",
        "filename",
        "relative_path",
        "width",
        "height",
        "pair_review_decision",
        "review_notes",
        "label_independently_verified",
        "file_status",
        "original_file_modified",
    ]

    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest saved: {OUTPUT}")
    print(f"Image records: {len(rows)}")

    counts = {}
    for decision, _ in PAIR_DECISIONS.values():
        counts[decision] = counts.get(decision, 0) + 1

    print("\nPair review summary:")
    for decision, count in counts.items():
        print(f"  {decision}: {count}")

    problems = [row for row in rows if row["file_status"] != "OK"]
    if problems:
        print(f"\nWARNING: {len(problems)} image files are missing or unreadable.")
        for row in problems:
            print(f"  Pair {row['pair_number']} ({row['label_from_folder']}): "
                  f"{row['file_status']}")
    else:
        print("All 32 image files were found and readable.")

    print("\nOriginal images were not modified.")


if __name__ == "__main__":
    main()
