
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
GOOD_DIR = ROOT / "data" / "validation" / "good"
DEFECTIVE_DIR = ROOT / "data" / "validation" / "defective"
OUTPUT = ROOT / "data" / "audit_pairs_1_5.jpg"

PAIRS = range(16, 17)
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 440
HEADER_HEIGHT = 45
GAP = 16
BACKGROUND = "white"

font = ImageFont.truetype("arial.ttf", 22)
small_font = ImageFont.truetype("arial.ttf", 16)

rows = []

for number in PAIRS:
    good_path = GOOD_DIR / f"image{number}.jpg"
    defective_path = DEFECTIVE_DIR / f"image{number}.jpg"

    if not good_path.exists() or not defective_path.exists():
        print(f"Skipping pair {number}: one or both files are missing.")
        continue

    good = Image.open(good_path).convert("RGB")
    defective = Image.open(defective_path).convert("RGB")

    print(
        f"Pair {number}: "
        f"GOOD {good.size}, DEFECTIVE {defective.size}"
    )

    row = Image.new(
        "RGB",
        (IMAGE_WIDTH * 2 + GAP, IMAGE_HEIGHT + HEADER_HEIGHT),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(row)

    draw.text((10, 8), f"PAIR {number} — GOOD", fill="black", font=font)
    draw.text(
        (IMAGE_WIDTH + GAP + 10, 8),
        f"PAIR {number} — DEFECTIVE",
        fill="black",
        font=font,
    )

    good.thumbnail((IMAGE_WIDTH, IMAGE_HEIGHT))
    defective.thumbnail((IMAGE_WIDTH, IMAGE_HEIGHT))

    good_x = (IMAGE_WIDTH - good.width) // 2
    defect_x = IMAGE_WIDTH + GAP + (
        IMAGE_WIDTH - defective.width
    ) // 2

    good_y = HEADER_HEIGHT + (IMAGE_HEIGHT - good.height) // 2
    defect_y = HEADER_HEIGHT + (
        IMAGE_HEIGHT - defective.height
    ) // 2

    row.paste(good, (good_x, good_y))
    row.paste(defective, (defect_x, defect_y))

    rows.append(row)

if not rows:
    raise SystemExit("No image pairs found. Check the dataset paths.")

sheet = Image.new(
    "RGB",
    (
        IMAGE_WIDTH * 2 + GAP,
        len(rows) * (IMAGE_HEIGHT + HEADER_HEIGHT + GAP),
    ),
    BACKGROUND,
)

for index, row in enumerate(rows):
    sheet.paste(
        row,
        (0, index * (IMAGE_HEIGHT + HEADER_HEIGHT + GAP)),
    )

sheet.save(OUTPUT, quality=95)
print(f"\nSaved comparison sheet: {OUTPUT}")
