
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "validation"
OUTPUT = ROOT / "data" / "candidate_pairs_review.jpg"

PAIR_NUMBERS = [1, 2, 3, 10, 11, 16]
THUMBNAIL_SIZE = (300, 230)
PADDING = 18
HEADER_HEIGHT = 58
ROW_HEIGHT = 300

try:
    font = ImageFont.truetype("arial.ttf", 17)
    small_font = ImageFont.truetype("arial.ttf", 13)
except OSError:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()


def load_image(folder, number):
    path = SOURCE / folder / f"image{number}.jpg"
    if not path.exists():
        return None, path

    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy(), path
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
        return None, path


def fit_image(image, size):
    if image is None:
        return Image.new("RGB", size, "#eeeeee")

    return ImageOps.pad(
        image,
        size,
        method=Image.Resampling.LANCZOS,
        color="#eeeeee",
        centering=(0.5, 0.5),
    )


canvas_width = PADDING * 4 + THUMBNAIL_SIZE[0] * 2
canvas_height = HEADER_HEIGHT + len(PAIR_NUMBERS) * ROW_HEIGHT + PADDING

sheet = Image.new("RGB", (canvas_width, canvas_height), "white")
draw = ImageDraw.Draw(sheet)

draw.text(
    (PADDING, 15),
    "Q-Insight — Candidate Pair Review",
    fill="black",
    font=font,
)

for row, number in enumerate(PAIR_NUMBERS):
    y = HEADER_HEIGHT + row * ROW_HEIGHT
    good, good_path = load_image("good", number)
    defective, defective_path = load_image("defective", number)

    draw.text(
        (PADDING, y + 3),
        f"PAIR {number}",
        fill="black",
        font=font,
    )

    image_y = y + 32
    good_x = PADDING
    defective_x = PADDING * 3 + THUMBNAIL_SIZE[0]

    draw.text(
        (good_x, y + 31),
        f"GOOD folder | {good_path.name}",
        fill="#176b37",
        font=small_font,
    )
    draw.text(
        (defective_x, y + 31),
        f"DEFECTIVE folder | {defective_path.name}",
        fill="#a12626",
        font=small_font,
    )

    image_y += 22
    sheet.paste(fit_image(good, THUMBNAIL_SIZE), (good_x, image_y))
    sheet.paste(
        fit_image(defective, THUMBNAIL_SIZE),
        (defective_x, image_y),
    )

    for image, path in [(good, good_path), (defective, defective_path)]:
        if image is None:
            detail = "MISSING / UNREADABLE"
        else:
            detail = f"{image.width} x {image.height}"

        x = good_x if path.parent.name == "good" else defective_x
        draw.text(
            (x, image_y + THUMBNAIL_SIZE[1] + 5),
            detail,
            fill="#444444",
            font=small_font,
        )

    draw.line(
        (PADDING, y + ROW_HEIGHT - 4,
         canvas_width - PADDING, y + ROW_HEIGHT - 4),
        fill="#cccccc",
        width=1,
    )

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT, quality=92)

print(f"Review sheet saved: {OUTPUT}")
print("Original dataset images were not modified.")
