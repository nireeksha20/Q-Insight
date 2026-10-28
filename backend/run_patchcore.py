
from pathlib import Path

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "data" / "anomalib_dataset"
RESULTS = BASE / "data" / "patchcore_results"


def main():
    datamodule = Folder(
        name="q_insight",
        root=DATASET,
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir="test/defective",
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        seed=42,
    )

    model = Patchcore()
    engine = Engine(
        default_root_dir=RESULTS,
        accelerator="auto",
        devices=1,
    )

    print("Starting PatchCore training...")
    engine.fit(model=model, datamodule=datamodule)

    print("Evaluating held-out normal and defective images...")
    engine.test(model=model, datamodule=datamodule)

    print(f"Experiment outputs: {RESULTS}")


if __name__ == "__main__":
    main()
