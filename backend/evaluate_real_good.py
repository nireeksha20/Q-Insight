
from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Patchcore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "single_switch_v1" / "prepared_v1"
VALIDATION_GOOD = PROJECT_ROOT / "data" / "validation" / "good"
CHECKPOINT = (
    PROJECT_ROOT / "data" / "patchcore_single_switch_v1_results"
    / "Patchcore" / "q_insight_single_switch_v1" / "v1"
    / "weights" / "lightning" / "model.ckpt"
)

THRESHOLD = 57.58588

def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not VALIDATION_GOOD.is_dir():
        raise FileNotFoundError(f"Validation folder not found: {VALIDATION_GOOD}")

    dm = Folder(
        name="q_insight_real_good_check",
        root=DATA_ROOT,
        normal_dir=DATA_ROOT / "train" / "good",
        normal_test_dir=VALIDATION_GOOD,
        abnormal_dir=DATA_ROOT / "test" / "defective",
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

    good_count = 0
    scores = []

    with torch.inference_mode():
        for batch in dm.test_dataloader():
            if batch.image is not None:
                batch.image = batch.image.cpu()

            predictions = model.predict_step(batch, 0)

            for prediction in predictions:
                path = str(getattr(prediction, "image_path", ""))
                if VALIDATION_GOOD.resolve().as_posix().lower() not in (
                    Path(path).resolve().as_posix().lower()
                ):
                    continue

                score = float(
                    prediction.pred_score.detach().cpu().reshape(-1)[0]
                )
                scores.append(score)
                good_count += 1

                result = "FLAGGED" if score >= THRESHOLD else "PASS"
                print(f"{Path(path).name:30} score={score:.6f}  {result}")

    print(f"\nValidation images scored: {good_count}")
    if scores:
        print(f"Minimum score: {min(scores):.6f}")
        print(f"Maximum score: {max(scores):.6f}")
        print(f"Mean score:    {sum(scores) / len(scores):.6f}")
        print(f"Flagged as anomalous: {sum(s >= THRESHOLD for s in scores)}")
    else:
        print("No validation images were matched. Check the dataloader paths.")

if __name__ == "__main__":
    main()
