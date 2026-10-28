from pathlib import Path
import shutil

base = Path(__file__).resolve().parent.parent
source = base / "data" / "validation"
root = base / "data" / "anomalib_dataset"

# Reset only the generated experiment dataset.
if root.exists():
    shutil.rmtree(root)

for folder in [
    root / "train" / "good",
    root / "test" / "good",
    root / "test" / "defective",
]:
    folder.mkdir(parents=True, exist_ok=True)

# Hold out four normal images for a basic false-positive check.
for i in range(1, 17):
    src = source / "good" / f"image{i}.jpg"
    if not src.exists():
        raise FileNotFoundError(src)

    destination = (
        root / "train" / "good" / src.name
        if i <= 12
        else root / "test" / "good" / src.name
    )
    shutil.copy2(src, destination)

    defective = source / "defective" / f"image{i}.jpg"
    if not defective.exists():
        raise FileNotFoundError(defective)

    shutil.copy2(defective, root / "test" / "defective" / defective.name)

print("Dataset prepared:", root)
print("Training normal images: 12")
print("Held-out normal images: 4")
print("Defective test images: 16")
