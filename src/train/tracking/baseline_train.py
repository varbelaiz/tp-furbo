"""Train the baseline YOLO detector for players, referees and the ball."""

from pathlib import Path
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EPOCHS = 40
IMG_SIZE = 960          # A slightly larger input helps with the ball
FALLBACK = "yolo11m.pt"  # Start from a smaller model while iterating
FRACTION = 0.5
BATCH = -1              # Auto batch size, based on available VRAM

BASELINE_DATA = PROJECT_ROOT / "data" / "tracking" / "YOLO_baseline" / "tracker.yaml"


def serialized_model_file(checkpoint: str = "best", use_run: str = "train") -> Path:
    """Path to a checkpoint inside a specific detection run."""
    return PROJECT_ROOT / f"runs/detect/{use_run}/weights/{checkpoint}.pt"


def train(
    data,
    fallback: str = FALLBACK,
    epochs: int = EPOCHS,
):
    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        device = 0
    else:
        print("WARNING: CUDA is not available, training will be slow.")
        device = "cpu"

    print(f"Loading model: {fallback}")
    model = YOLO(fallback)

    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=device,
        optimizer="auto",
        amp=True,
        cache=True,
        workers=8,       # Can be raised to 12-16 when the VM has more CPU
        fraction=FRACTION,
        patience=20,     # Early stopping
    )


if __name__ == "__main__":
    print("Starting the baseline training run...")
    train(data=BASELINE_DATA)
