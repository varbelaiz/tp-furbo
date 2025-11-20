"""Fine-tune the best baseline detector into a dedicated ball detector."""

from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Hyperparameters
EPOCHS = 40
IMG_SIZE = 960
FRACTION = 0.5
BATCH = -1

RUNS_ROOT = PROJECT_ROOT / "runs"

BALL_DATA = PROJECT_ROOT / "data" / "tracking" / "YOLO_ball" / "ball.yaml"

# Fine-tuned from the train4 detection run
FALLBACK = PROJECT_ROOT / "runs" / "detect" / "train4" / "weights" / "best.pt"

DEFAULT_RUN_NAME = "ball"


def serialized_model_file(checkpoint: str = "best", use_run: str = DEFAULT_RUN_NAME) -> Path:
    """
    Path to a checkpoint inside a specific run.
    Example: runs/<use_run>/weights/<checkpoint>.pt
    """
    if use_run is None:
        raise ValueError("use_run cannot be None in serialized_model_file()")
    return RUNS_ROOT / use_run / "weights" / f"{checkpoint}.pt"


def train(
    data: Path,
    use_run: str | None = None,
    fallback: Path = FALLBACK,
    epochs: int = EPOCHS,
    augment: bool = True,
):
    print(f"Torch version: {torch.__version__}")

    if torch.cuda.is_available():
        device = 0
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
        torch.cuda.empty_cache()
    else:
        device = "cpu"
        print("WARNING: CUDA is not available, training will be slow.")

    # Decide on the base model and whether to resume
    if use_run is not None:
        # Explicit run name: try to resume from the last checkpoint
        last_ckpt = serialized_model_file("last", use_run)
        if last_ckpt.exists():
            print(f"Resuming training from: {last_ckpt}")
            base_model_path = last_ckpt
            resume = True
        else:
            print(f"No checkpoint found for run '{use_run}'.")
            print(f"Fine-tuning from fallback: {fallback}")
            base_model_path = fallback
            resume = False
        run_name = use_run
    else:
        # No run name given, so always start a new run
        run_name = DEFAULT_RUN_NAME
        base_model_path = fallback
        resume = False
        print("Starting a NEW ball run (YOLO will create ball, ball2, ...).")
        print(f"Fine-tuning from fallback: {fallback}")

    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found at: {base_model_path}")

    if not data.exists():
        raise FileNotFoundError(f"Dataset yaml not found at: {data}")

    print(f"Loading model from: {base_model_path}")
    model = YOLO(str(base_model_path))

    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=device,
        optimizer="auto",
        amp=True,
        cache=True,
        workers=8,
        fraction=FRACTION,
        patience=20,
        project=str(RUNS_ROOT),
        name=run_name,
        resume=resume,
        augment=augment,
    )


if __name__ == "__main__":
    print("Starting ball-only training...")
    train(
        data=BALL_DATA,
        use_run=None,  # None starts a new run on every execution
        augment=True,
    )
