
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "single_switch_v1" / "prepared_v1"
VALIDATION_ROOT = PROJECT_ROOT / "data" / "validation"
CHECKPOINT = (
    PROJECT_ROOT / "data" / "patchcore_single_switch_v1_results"
    / "Patchcore" / "q_insight_single_switch_v1" / "v1"
    / "weights" / "lightning" / "model.ckpt"
)
THRESHOLD = 57.58588


def main():
    for folder in ("good", "defective"):
        path = VALIDATION_ROOT / folder
        if not path.is_dir() or not any(path.iterdir()):
            raise FileNotFoundError(f"Missing or empty folder: {path}")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    dm = Folder(
        name="q_insight_real_validation",
        root=DATA_ROOT,
        normal_dir=DATA_ROOT / "train" / "good",
        normal_test_dir=VALIDATION_ROOT / "good",
        abnormal_dir=VALIDATION_ROOT / "defective",
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

    results = []
    with torch.inference_mode():
        for batch in dm.test_dataloader():
            if batch.image is not None:
                batch.image = batch.image.cpu()

            predictions = model.predict_step(batch, 0)
            for prediction in predictions:
                path = Path(str(prediction.image_path))
                actual = (
                    "defective"
                    if "defective" in str(path).lower()
                    else "good"
                )
                score = float(
                    prediction.pred_score.detach().cpu().reshape(-1)[0]
                )
                predicted = "defective" if score >= THRESHOLD else "good"
                results.append((path.name, actual, score, predicted))

    print("\nREAL-WORLD VALIDATION RESULTS")
    print("-" * 76)
    for name, actual, score, predicted in results:
        print(
            f"{name:16} actual={actual:10} "
            f"score={score:9.4f} predicted={predicted}"
        )

    for actual in ("good", "defective"):
        group = [r for r in results if r[1] == actual]
        correct = sum(r[3] == actual for r in group)
        print(f"\n{actual.upper()}: {correct}/{len(group)} correctly classified")
        if group:
            scores = [r[2] for r in group]
            print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
            print(f"  Mean score:  {sum(scores) / len(scores):.4f}")

    print(f"\nTotal images evaluated: {len(results)}")
    print(f"Threshold used: {THRESHOLD}")
    print("No model weights or dataset files were modified.")


if __name__ == "__main__":
    main()
