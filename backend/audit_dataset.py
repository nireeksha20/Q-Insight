
from pathlib import Path
import cv2
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "validation"
OUTPUT = BASE / "data" / "dataset_audit.jpg"

THUMB_W, THUMB_H = 240, 190
LABEL_H, GAP = 32, 12
COLS = 4


def load_thumbnail(path):
    image = cv2.imread(str(path))
    if image is None:
        return None

    h, w = image.shape[:2]
    scale = min(THUMB_W / w, THUMB_H / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    image = cv2.resize(image, new_size)

    canvas = np.full((THUMB_H, THUMB_W, 3), 245, dtype=np.uint8)
    y = (THUMB_H - image.shape[0]) // 2
    x = (THUMB_W - image.shape[1]) // 2
    canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
    return canvas


def main():
    groups = [
        ("GOOD", DATA / "good"),
        ("DEFECTIVE", DATA / "defective"),
    ]

    cards = []
    for label, folder in groups:
        files = sorted(
            folder.glob("*"),
            key=lambda p: int(p.stem.replace("image", ""))
            if p.stem.replace("image", "").isdigit()
            else p.name.lower(),
        )
        for path in files:
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            thumb = load_thumbnail(path)
            if thumb is None:
                print(f"Could not read: {path}")
                continue

            card = np.full(
                (THUMB_H + LABEL_H, THUMB_W, 3), 255, dtype=np.uint8
            )
            card[:THUMB_H] = thumb
            text = f"{label} - {path.name}"
            cv2.putText(
                card, text, (5, THUMB_H + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (20, 20, 20), 1, cv2.LINE_AA,
            )
            cards.append((label, path.name, card))

    if not cards:
        raise RuntimeError(f"No images found under {DATA}")

    rows = []
    for start in range(0, len(cards), COLS):
        row_cards = [item[2] for item in cards[start:start + COLS]]
        row = np.full(
            (THUMB_H + LABEL_H, COLS * THUMB_W + (COLS - 1) * GAP, 3),
            235, dtype=np.uint8,
        )
        for i, card in enumerate(row_cards):
            x = i * (THUMB_W + GAP)
            row[:, x:x + THUMB_W] = card
        rows.append(row)

    sheet = np.full(
        (len(rows) * (THUMB_H + LABEL_H + GAP) - GAP,
         COLS * THUMB_W + (COLS - 1) * GAP, 3),
        235, dtype=np.uint8,
    )

    y = 0
    for row in rows:
        sheet[y:y + row.shape[0]] = row
        y += row.shape[0] + GAP

    if not cv2.imwrite(str(OUTPUT), sheet):
        raise RuntimeError("Failed to save contact sheet")

    print(f"Images included: {len(cards)}")
    print(f"Saved contact sheet: {OUTPUT}")


if __name__ == "__main__":
    main()
