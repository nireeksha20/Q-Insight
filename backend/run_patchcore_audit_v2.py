
from pathlib import Path

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "anomalib_dataset_audit_v2"
RESULTS = ROOT / "data" / "patchcore_audit_v2_results"


def main():
    expected = [
        DATASET / "train" / "good",
        DATASET / "test" / "good",
        DATASET / "test" / "defective",
    ]

    for folder in expected:
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder missing: {folder}")

    datamodule = Folder(
        name="q_insight_audit_v2",
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

    print(f"Dataset: {DATASET}")
    print(f"Results: {RESULTS}")
    print(f"Training normal images: {len(datamodule.train_data)}")
    print(f"Test images: {len(datamodule.test_data)}")

    model = Patchcore()

    engine = Engine(
        default_root_dir=RESULTS,
        accelerator="auto",
        devices=1,
    )

    print("\nTraining a separate PatchCore v2 model...")
    engine.fit(model=model, datamodule=datamodule)

    print("\nRunning test evaluation...")
    metrics = engine.test(model=model, datamodule=datamodule)

    print("\nReturned metrics (preliminary):")
    print(metrics)

    print("\nV2 experiment finished.")
    print("The v1 model and existing API were not modified.")


if __name__ == "__main__":
    main()
