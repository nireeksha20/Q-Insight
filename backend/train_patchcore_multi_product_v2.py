from pathlib import Path

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "anomalib_multi_product_v1"
RESULTS_DIR = PROJECT_ROOT / "data" / "patchcore_multi_product_v2_results"

datamodule = Folder(
    name="q_insight_multi_product_v2",
    root=DATA_ROOT,
    normal_dir=DATA_ROOT / "train" / "good",
    normal_test_dir=DATA_ROOT / "test" / "good",
    abnormal_dir=DATA_ROOT / "test" / "defective",
    test_split_mode="from_dir",
    val_split_mode="from_train",
    val_split_ratio=0.2,
    train_batch_size=8,
    eval_batch_size=1,
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
    default_root_dir=str(RESULTS_DIR),
    accelerator="cpu",
    devices=1,
    max_epochs=1,
    logger=False,
)

if __name__ == "__main__":
    if RESULTS_DIR.exists() and any(RESULTS_DIR.iterdir()):
        raise RuntimeError(f"Results already exist: {RESULTS_DIR}")

    datamodule.setup()
    print("Training images:", len(datamodule.train_data))
    print("Validation images:", len(datamodule.val_data))
    print("Test images:", len(datamodule.test_data))
    print("Results:", RESULTS_DIR)

    print("\nStarting corrected experiment...")
    engine.fit(model=model, datamodule=datamodule)
    print("\nTraining finished:", RESULTS_DIR)
