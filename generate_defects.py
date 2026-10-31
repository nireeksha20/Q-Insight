import os
import random
import cv2
import numpy as np

# Set directories
INPUT_DIR = r"C:\Users\H P\Documents\PROJECTS\Q-Insight\data\single_switch_v1\incoming\perspective_angles"
OUTPUT_DIR = r"C:\Users\H P\Documents\PROJECTS\Q-Insight\data\single_switch_v1\incoming\generated_defective"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def inject_crack(img):
    # Draw a jagged hairline fracture line across the plate
    img_mod = img.copy()
    h, w = img_mod.shape[:2]

    # Pick a random starting point around the inner switch
    start_x = random.randint(w//3, 2*w//3)
    start_y = random.randint(h//3, 2*h//3)

    current_pt = (start_x, start_y)
    segments = random.randint(3, 6)

    for _ in range(segments):
        next_x = current_pt[0] + random.randint(-25, 25)
        next_y = current_pt[1] + random.randint(15, 45) # Force downward crack path
        next_x = np.clip(next_x, 0, w - 1)
        next_y = np.clip(next_y, 0, h - 1)

        # Draw dark fracture line
        cv2.line(img_mod, current_pt, (next_x, next_y), (40, 40, 40), thickness=random.randint(1, 2))
        current_pt = (next_x, next_y)

    return img_mod

def inject_scuffs(img):
    # Simulate production scuffs, burn blemishes, or dirt markings
    img_mod = img.copy()
    h, w = img_mod.shape[:2]

    for _ in range(random.randint(2, 5)):
        center = (random.randint(w//4, 3*w//4), random.randint(h//4, 3*h//4))
        axes = (random.randint(5, 15), random.randint(2, 5))
        angle = random.randint(0, 180)

        # Transparent smudge blending
        overlay = img_mod.copy()
        cv2.ellipse(overlay, center, axes, angle, 0, 360, (70, 70, 70), -1)
        alpha = random.uniform(0.3, 0.5)
        cv2.addWeighted(overlay, alpha, img_mod, 1 - alpha, 0, dst=img_mod)

    return img_mod

# Get images from the perspective angles folder
images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not images:
    print("No clean images found to inject defects into!")
else:
    print("Generating synthetic structural anomalies for the defective set...")

    for idx, img_name in enumerate(images):
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            # Alternate anomaly creation types
            if idx % 2 == 0:
                defect_img = inject_crack(img)
                label = "crack"
            else:
                defect_img = inject_scuffs(img)
                label = "scuff"

            out_name = f"defect_{label}_{idx:03d}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), defect_img)

    print(f"Successfully populated {OUTPUT_DIR} with defective variations.")
