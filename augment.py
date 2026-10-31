import os
import random
import cv2
import numpy as np

# Set directories
INCOMING_DIR = r"C:\Users\H P\Documents\PROJECTS\Q-Insight\data\single_switch_v1\incoming"
OUTPUT_DIR = os.path.join(INCOMING_DIR, "generated_good")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_base_switch():
    # Create a 500x500 white background wall canvas
    canvas = np.full((500, 500, 3), 220, dtype=np.uint8)

    # Draw switch plate border (outer plate)
    cv2.rectangle(canvas, (120, 100), (380, 400), (245, 245, 245), -1)
    cv2.rectangle(canvas, (120, 100), (380, 400), (180, 180, 180), 2)

    # Draw inner switch mechanism plate
    cv2.rectangle(canvas, (210, 200), (290, 300), (235, 235, 235), -1)
    cv2.rectangle(canvas, (210, 200), (290, 300), (160, 160, 160), 1)

    # Draw the toggle mechanism accent line
    cv2.line(canvas, (210, 250), (290, 250), (190, 190, 190), 2)
    return canvas

def augment_image(img):
    # 1. Random positional variations (Rotation & Translation)
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    img_mod = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 2. Random lighting variations (Brightness & Contrast)
    brightness = random.randint(-40, 40)
    contrast = random.uniform(0.7, 1.3)
    img_mod = cv2.convertScaleAbs(img_mod, alpha=contrast, beta=brightness)

    # 3. Add soft lighting shadow gradient (Simulate window/overhead sun angles)
    mask = np.zeros_like(img_mod, dtype=np.float32)
    direction = random.choice(['left', 'right', 'top', 'bottom'])
    if direction == 'left':
        for x in range(w): mask[:, x] = x / w
    else:
        for x in range(w): mask[:, x] = (w - x) / w
    img_mod = cv2.convertScaleAbs(img_mod * (0.6 + 0.4 * mask))

    return img_mod

# Look for standard base images you've saved in incoming
base_images = [f for f in os.listdir(INCOMING_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

target_count = 100
generated = 0
print(f"Generating {target_count} unique training variations...")

while generated < target_count:
    if base_images:
        base_name = random.choice(base_images)
        img_path = os.path.join(INCOMING_DIR, base_name)
        img = cv2.imread(img_path)
    else:
        # Fallback if folder is empty: draw its own base switch plate
        img = generate_base_switch()

    if img is not None:
        aug_img = augment_image(img)
        out_name = f"aug_switch_plate_{generated:03d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), aug_img)
        generated += 1

print(f" Successfully populated {OUTPUT_DIR} with {target_count} variations.")
