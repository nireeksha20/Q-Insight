
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "single_switch_v1"
    / "prepared_v1"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "data"
    / "patchcore_single_switch_v1_results"
    / "Patchcore"
    / "q_insight_single_switch_v1"
    / "v1"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

if __name__ == "__main__":
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    datamodule = Folder(
        name="q_insight_single_switch_v1",
        root=DATA_ROOT,
        normal_dir=DATA_ROOT / "train" / "good",
        normal_test_dir=DATA_ROOT / "test" / "good",
        abnormal_dir=DATA_ROOT / "test" / "defective",
        test_split_mode="from_dir",
        val_split_mode="from_train",
        val_split_ratio=0.2,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        seed=42,
    )

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=("layer2", "layer3"),
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine(
        default_root_dir=str(
            PROJECT_ROOT / "data" / "patchcore_single_switch_v1_evaluation"
        ),
        accelerator="cpu",
        devices=1,
        logger=False,
    )

    print("Checkpoint:", CHECKPOINT)
    print("Starting evaluation on the held-out test set...")

    results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(CHECKPOINT),
    )

    print("\nEvaluation results:")
    print(results)
