
from pathlib import Path
import csv
import shutil

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "validation"
DEST = ROOT / "data" / "anomalib_dataset_audit_v2"
MANIFEST = ROOT / "data" / "anomalib_dataset_audit_v2_manifest.csv"

# Reserve these pairs exclusively for evaluation.
EVALUATION_PAIRS = [1, 2, 3, 10, 11, 16]

# Use the remaining good images for training.
TRAIN_GOOD_PAIRS = [
    pair for pair in range(1, 17)
    if pair not in EVALUATION_PAIRS
]


def main():
    if DEST.exists():
        raise FileExistsError(
            f"{DEST} already exists. Nothing was overwritten. "
            "Choose a new destination folder before proceeding."
        )

    train_dir = DEST / "train" / "good"
    test_good_dir = DEST / "test" / "good"
    test_defective_dir = DEST / "test" / "defective"

    for folder in [train_dir, test_good_dir, test_defective_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    records = []

    def copy_image(pair, label, destination):
        source = SOURCE / label / f"image{pair}.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"Missing image: {source}")

        target = destination / source.name
        shutil.copy2(source, target)

        records.append({
            "pair_number": pair,
            "label": label,
            "source": str(source.relative_to(ROOT)),
            "destination": str(target.relative_to(ROOT)),
        })

    for pair in TRAIN_GOOD_PAIRS:
        copy_image(pair, "good", train_dir)

    for pair in EVALUATION_PAIRS:
        copy_image(pair, "good", test_good_dir)
        copy_image(pair, "defective", test_defective_dir)

    with MANIFEST.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "pair_number", "label", "source", "destination"
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Dataset created: {DEST}")
    print(f"Normal training images: {len(TRAIN_GOOD_PAIRS)}")
    print(f"Good evaluation images: {len(EVALUATION_PAIRS)}")
    print(f"Defective evaluation images: {len(EVALUATION_PAIRS)}")
    print(f"Manifest: {MANIFEST}")
    print("Original validation images were not modified.")


if __name__ == "__main__":
    main()
