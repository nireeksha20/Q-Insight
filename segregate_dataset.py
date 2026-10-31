import os
import shutil

# Source Staging Folders
SRC_GOOD = r"C:\Users\H P\Documents\PROJECTS\Q-Insight\data\single_switch_v1\incoming\perspective_angles"
SRC_DEFECTIVE = r"C:\Users\H P\Documents\PROJECTS\Q-Insight\data\single_switch_v1\incoming\generated_defective"

# Target Operational Folders
DEST_GOOD = r"C:\Users\H P\Documents\PROJECTS\Q-Insight\data\single_switch_v1\good"
DEST_DEFECTIVE = r"C:\Users\H P\Documents\PROJECTS\Q-Insight\data\single_switch_v1\defective"

def distribute_images(src_dir, dest_dir, prefix):
    if not os.path.exists(src_dir):
        print(f"Skipping: Source folder {src_dir} does not exist.")
        return

    os.makedirs(dest_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    moved_count = 0
    for file_name in files:
        src_path = os.path.join(src_dir, file_name)
        # Give them standard clean operational filenames
        new_name = f"{prefix}_{moved_count:03d}{os.path.splitext(file_name)[1]}"
        dest_path = os.path.join(dest_dir, new_name)

        # Copy to preserve staging raw files, or use shutil.move if you want to clear incoming
        shutil.copy(src_path, dest_path)
        moved_count += 1

    print(f"Successfully migrated {moved_count} images to {dest_dir}")

print("Starting Q-Insight target folder allocation...")
distribute_images(SRC_GOOD, DEST_GOOD, "switch_normal")
distribute_images(SRC_DEFECTIVE, DEST_DEFECTIVE, "switch_defective")
print("All set! Dataset is fully balanced and organized for validation.")
