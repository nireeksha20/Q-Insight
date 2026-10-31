
from pathlib import Path
import cv2
import numpy as np
import torch

from anomalib.data import Folder
from anomalib.models import Patchcore

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "single_switch_v1" / "prepared_v1"
PHOTOS = ROOT / "data" / "real_switch_pilot" / "damaged_faceplate"

CHECKPOINT = (
    ROOT / "data" / "patchcore_single_switch_v1_results"
    / "Patchcore" / "q_insight_single_switch_v1"
    / "v1" / "weights" / "lightning" / "model.ckpt"
)

OUTPUT = ROOT / "data" / "real_faceplate_heatmaps"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    dm = Folder(
        name="q_insight_real_faceplate_heatmaps",
        root=DATA,
        normal_dir=DATA / "train" / "good",
        normal_test_dir=DATA / "test" / "good",
        abnormal_dir=PHOTOS,
        test_split_mode="from_dir",
        val_split_mode="from_train",
        val_split_ratio=0.2,
        train_batch_size=8,
        eval_batch_size=1,
        num_workers=0,
        seed=42,
    )
    dm.setup()

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    checkpoint = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    model.load_state_dict(
        checkpoint.get("state_dict", checkpoint), strict=True
    )
    model.eval()

    photo_root = str(PHOTOS.resolve()).lower()
    saved = 0

    with torch.inference_mode():
        for batch in dm.test_dataloader():
            if getattr(batch, "image", None) is not None:
                batch.image = batch.image.to("cpu")
            if getattr(batch, "gt_label", None) is not None:
                batch.gt_label = batch.gt_label.to("cpu")
            if getattr(batch, "gt_mask", None) is not None:
                batch.gt_mask = batch.gt_mask.to("cpu")

            predictions = model.predict_step(batch, 0)

            for pred in predictions:
                path_text = str(getattr(pred, "image_path", ""))
                if not path_text:
                    continue

                image_path = Path(path_text)
                if photo_root not in str(image_path.resolve()).lower():
                    continue

                anomaly_map = getattr(pred, "anomaly_map", None)
                if anomaly_map is None:
                    print(f"No anomaly map returned for {image_path.name}")
                    print("Available fields:", getattr(pred, "__dict__", {}).keys())
                    continue

                heat = anomaly_map.detach().cpu().numpy().squeeze()
                if heat.ndim != 2:
                    print(f"Unexpected map shape for {image_path.name}: {heat.shape}")
                    continue

                heat = np.nan_to_num(heat)
                low, high = float(heat.min()), float(heat.max())

                if high > low:
                    heat8 = ((heat - low) / (high - low) * 255).astype(np.uint8)
                else:
                    heat8 = np.zeros_like(heat, dtype=np.uint8)

                original = cv2.imread(str(image_path))
                if original is None:
                    print(f"Could not read {image_path}")
                    continue

                original = cv2.resize(
                    original, (heat.shape[1], heat.shape[0])
                )
                colored = cv2.applyColorMap(heat8, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(original, 0.55, colored, 0.45, 0)

                cv2.imwrite(str(OUTPUT / f"{image_path.stem}_heatmap.png"), colored)
                cv2.imwrite(str(OUTPUT / f"{image_path.stem}_overlay.png"), overlay)
                saved += 1
                print(f"Saved heatmap and overlay for {image_path.name}")

    print(f"\nFinished. Saved overlays for {saved} images.")
    print(f"Output folder: {OUTPUT}")


if __name__ == "__main__":
    main()
