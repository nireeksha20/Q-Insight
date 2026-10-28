
from pathlib import Path
from collections import Counter
import hashlib
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SPLITS = {
    "train/good": DATA / "anomalib_dataset" / "train" / "good",
    "test/good": DATA / "anomalib_dataset" / "test" / "good",
    "test/defective": DATA / "anomalib_dataset" / "test" / "defective",
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    records = []
    hashes_by_split = {}
    errors = []

    print("Q-Insight PatchCore dataset audit")
    print("=" * 40)

    for split_name, folder in SPLITS.items():
        if not folder.is_dir():
            print(f"\nMISSING DIRECTORY: {folder}")
            errors.append(f"Missing directory: {split_name}")
            continue

        images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        )

        print(f"\n{split_name}: {len(images)} images")
        split_hashes = {}

        for path in images:
            try:
                with Image.open(path) as image:
                    dimensions = image.size
                    image.verify()

                file_hash = sha256(path)
                split_hashes.setdefault(file_hash, []).append(path.name)

                records.append({
                    "split": split_name,
                    "filename": path.name,
                    "width": dimensions[0],
                    "height": dimensions[1],
                    "sha256": file_hash,
                })

                print(f"  {path.name}: {dimensions[0]}x{dimensions[1]}")
            except Exception as exc:
                errors.append(f"{split_name}/{path.name}: {exc}")
                print(f"  ERROR: {path.name}: {exc}")

        hashes_by_split[split_name] = split_hashes

    print("\n" + "=" * 40)
    print("EXPECTED COUNTS")
    expected = {
        "train/good": 12,
        "test/good": 4,
        "test/defective": 16,
    }

    for split_name, expected_count in expected.items():
        actual = len(hashes_by_split.get(split_name, {}))
        # Count files rather than unique hashes for the actual count.
        folder = SPLITS[split_name]
        actual_files = (
            sum(
                p.is_file() and p.suffix.lower() in
                {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                for p in folder.iterdir()
            )
            if folder.is_dir() else 0
        )
        status = "OK" if actual_files == expected_count else "CHECK"
        print(f"{split_name}: {actual_files} / {expected_count} [{status}]")

    print("\nEXACT DUPLICATES ACROSS SPLITS")
    found_overlap = False
    split_names = list(hashes_by_split)

    for i, first in enumerate(split_names):
        for second in split_names[i + 1:]:
            common = (
                set(hashes_by_split[first])
                & set(hashes_by_split[second])
            )
            if common:
                found_overlap = True
                print(f"{first} <-> {second}: {len(common)} matching file(s)")
                for digest in common:
                    print("  ", hashes_by_split[first][digest][0],
                          "<->", hashes_by_split[second][digest][0])

    if not found_overlap:
        print("No exact image duplicates found across the checked splits.")

    print("\nIMAGE READ ERRORS")
    if errors:
        for error in errors:
            print("  ", error)
    else:
        print("None")

    print("\nAudit complete. No files were changed.")
    print("Note: matching hashes detect exact duplicates, not near-duplicates.")

if __name__ == "__main__":
    main()
