import os
import cv2
import numpy as np

# Set directories
INPUT_DIR = r"C:\Users\H P\Documents\PROJECTS\Q-Insight\data\single_switch_v1\incoming\generated_good"
OUTPUT_DIR = r"C:\Users\H P\Documents\PROJECTS\Q-Insight\data\single_switch_v1\incoming\perspective_angles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_perspective_tilt(img, tilt_type):
    h, w = img.shape[:2]
    # Source corners of the original image
    src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])

    # Define destination points based on tilt type (Left, Right, Top, Bottom look)
    shift = int(w * 0.15) # Controls severity of the angle look

    if tilt_type == 'tilt_left':
        # Simulates looking from the left side (Right side compresses)
        dst_pts = np.float32([[0, 0], [w-1-shift, shift], [0, h-1], [w-1-shift, h-1-shift]])
    elif tilt_type == 'tilt_right':
        # Simulates looking from the right side (Left side compresses)
        dst_pts = np.float32([[shift, shift], [w-1, 0], [shift, h-1-shift], [w-1, h-1]])
    elif tilt_type == 'tilt_up':
        # Simulates looking from above (Bottom side compresses)
        dst_pts = np.float32([[0, 0], [w-1, 0], [shift, h-1-shift], [w-1-shift, h-1-shift]])
    else: # tilt_down
        # Simulates looking from below (Top side compresses)
        dst_pts = np.float32([[shift, shift], [w-1-shift, shift], [0, h-1], [w-1, h-1]])

    # Calculate transformation matrix and warp
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return warped

# Get images from your generated_good folder
images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not images:
    print("No images found in your generated_good folder!")
else:
    print(f"Applying extreme perspective angle views to your dataset...")
    tilts = ['tilt_left', 'tilt_right', 'tilt_up', 'tilt_down']

    for idx, img_name in enumerate(images[:25]): # Transform a subset to create 100 variations
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            for tilt in tilts:
                angled_img = apply_perspective_tilt(img, tilt)
                out_name = f"angled_{tilt}_{idx:03d}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), angled_img)

    print(f"Successfully populated {OUTPUT_DIR} with 3D-perspective variations.")
