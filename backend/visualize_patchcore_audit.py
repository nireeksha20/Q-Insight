
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "data" / "anomalib_dataset_audit_v1"
CHECKPOINT = (
    ROOT / "data" / "patchcore_audit_results"
    / "Patchcore" / "q_insight_audit_v1"
    / "v0" / "weights" / "lightning" / "model.ckpt"
)
OUTPUT_DIR = ROOT / "data" / "patchcore_audit_heatmaps"

# Only visualize these four candidate pairs.
TARGETS = {2, 3, 11, 16}


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    datamodule = Folder(
        name="q_insight_audit_v1",
        root=DATASET,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/defective",
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        seed=42,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
    )
    datamodule.setup("test")

    model = Patchcore.load_from_checkpoint(
        str(CHECKPOINT), map_location="cpu"
    )
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    with torch.inference_mode():
        for batch in datamodule.test_dataloader():
            output = model(batch.image)

            if not hasattr(output, "anomaly_map"):
                raise RuntimeError(
                    "This model output has no anomaly_map. "
                    "Inspect the output fields before proceeding."
                )

            maps = output.anomaly_map.detach().cpu().numpy()
            paths = batch.image_path
            scores = output.pred_score.detach().cpu().reshape(-1).tolist()

            for i, path_value in enumerate(paths):
                path = Path(str(path_value))

                if path.stem not in {
                    f"image{n}" for n in TARGETS
                }:
                    continue

                label = "defective" if "defective" in path.parts else "good"
                original = cv2.imread(str(path))

                if original is None:
                    print(f"Skipping unreadable image: {path}")
                    continue

                anomaly = np.squeeze(maps[i])
                if anomaly.ndim != 2:
                    raise RuntimeError(
                        f"Unexpected anomaly-map shape: {maps[i].shape}"
                    )

                anomaly = cv2.resize(
                    anomaly.astype(np.float32),
                    (original.shape[1], original.shape[0]),
                )

                # Normalize only for visualization, not for model scoring.
                normalized = cv2.normalize(
                    anomaly, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)

                heatmap = cv2.applyColorMap(
                    normalized, cv2.COLORMAP_JET
                )
                overlay = cv2.addWeighted(
                    original, 0.55, heatmap, 0.45, 0
                )

                # Side-by-side: original | heatmap overlay
                combined = np.hstack([original, overlay])
                output_path = (
                    OUTPUT_DIR / f"{path.stem}_{label}_heatmap.jpg"
                )

                if not cv2.imwrite(str(output_path), combined):
                    raise RuntimeError(f"Could not save {output_path}")

                saved += 1
                print(
                    f"Saved {output_path.name} "
                    f"| score={scores[i]:.6f}"
                )

    print(f"\nSaved {saved} visualizations to:")
    print(OUTPUT_DIR)

    if saved != 8:
        print(
            "WARNING: Expected 8 images (4 pairs × 2 labels). "
            "Check the test dataset and output filenames."
        )


if __name__ == "__main__":
    main()
