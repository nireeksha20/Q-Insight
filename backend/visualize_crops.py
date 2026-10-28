
from pathlib import Path
import cv2
import numpy as np

from main import read_image, crop_product_region

BASE = Path(__file__).resolve().parent.parent
GOOD = BASE / "data" / "validation" / "good"
BAD = BASE / "data" / "validation" / "defective"
OUT = BASE / "data" / "crop_audit.jpg"

TILE_W, TILE_H = 240, 200


def make_tile(image, label):
    if image is None or image.size == 0:
        image = np.zeros((100, 100, 3), dtype=np.uint8)

    h, w = image.shape[:2]
    scale = min(TILE_W / w, 150 / h)
    resized = cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale)))
    )

    canvas = np.full((TILE_H, TILE_W, 3), 255, dtype=np.uint8)
    y = 35 + (150 - resized.shape[0]) // 2
    x = (TILE_W - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized

    cv2.putText(
        canvas, label, (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )
    return canvas


def main():
    tiles = []

    for i in range(1, 17):
        for label, folder in [("GOOD", GOOD), ("DEFECTIVE", BAD)]:
            path = folder / f"image{i}.jpg"
            image = read_image(path) if path.exists() else None

            if image is None:
                print(f"Could not read: {path}")
                continue

            cropped = crop_product_region(image)
            tiles.append(make_tile(cropped, f"Pair {i} - {label}"))

    if not tiles:
        print("No images were loaded.")
        return

    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full(
        (rows * TILE_H, columns * TILE_W, 3),
        235, dtype=np.uint8
    )

    for index, tile in enumerate(tiles):
        y = (index // columns) * TILE_H
        x = (index % columns) * TILE_W
        sheet[y:y + TILE_H, x:x + TILE_W] = tile

    if not cv2.imwrite(str(OUT), sheet):
        raise RuntimeError(f"Could not save image: {OUT}")

    print(f"Crop audit saved to: {OUT}")


if __name__ == "__main__":
    main()
