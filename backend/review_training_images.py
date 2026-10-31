
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "data" / "anomalib_dataset_audit_v2" / "train" / "good"
OUTPUT = ROOT / "data" / "patchcore_training_review.jpg"

THUMB_W = 280
THUMB_H = 230
LABEL_H = 42
COLS = 2
GAP = 16

files = sorted(
    TRAIN_DIR.glob("*"),
    key=lambda p: int(p.stem.replace("image", ""))
    if p.stem.replace("image", "").isdigit() else 9999
)

if not files:
    raise FileNotFoundError(
        f"No training images found in: {TRAIN_DIR}"
    )

rows = (len(files) + COLS - 1) // COLS
cell_h = THUMB_H + LABEL_H
sheet = Image.new(
    "RGB",
    (GAP + COLS * (THUMB_W + GAP), GAP + rows * (cell_h + GAP)),
    "white",
)
draw = ImageDraw.Draw(sheet)

for index, path in enumerate(files):
    x = GAP + (index % COLS) * (THUMB_W + GAP)
    y = GAP + (index // COLS) * (cell_h + GAP)

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        original_size = image.size
        image.thumbnail((THUMB_W, THUMB_H))
        canvas = Image.new("RGB", (THUMB_W, THUMB_H), "#eeeeee")
        canvas.paste(
            image,
            ((THUMB_W - image.width) // 2,
             (THUMB_H - image.height) // 2),
        )
        sheet.paste(canvas, (x, y))

    draw.text(
        (x, y + THUMB_H + 4),
        f"{path.name} | {original_size[0]}x{original_size[1]}",
        fill="black",
    )

sheet.save(OUTPUT, quality=95)

print(f"Training review saved: {OUTPUT}")
print(f"Training images: {len(files)}")
print("Original images and model checkpoints were not modified.")
