import torch
import os
from ultralytics import YOLO

EPOCHS = 40          # baja un poco para iterar más rápido
IMG_SIZE = 960       # algo más grande ayuda a la pelota
FALLBACK = "yolo11m.pt"  # empezar con modelo más chico para probar
FRACTION = .5      # usa todo el dataset
TIME = None          # sin límite duro de horas (o quitalo directamente)
BATCH = -1           # auto batch según VRAM

def serialized_model_file(checkpoint="best", use_run="train"):
    return f"runs/detect/{use_run}/weights/{checkpoint}.pt"

def train(
    data,
    use_run=None,
    fallback=FALLBACK,
    epochs=EPOCHS,
    augment=True,
):
    print(f"Torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        device = 0
    else:
        print("WARNING: CUDA is not available.")
        device = "cpu"

    # Siempre arrancá “limpio” al principio hasta que tengas un buen setup
    resume_training = False
    use_model = fallback

    print(f"Loading model: {use_model}")
    model = YOLO(use_model)

    model.train(
        data=data,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,          # que calcule el máximo posible
        device='cuda',
        optimizer="auto",
        amp=True,
        cache=True,
        workers=8,         # podés subir a 12–16 si la VM tiene CPU
        fraction=FRACTION,     
        patience=20,       # early stopping
        # no pongas 'time=' para no cortar por horas
    )


if __name__ == "__main__":
    print("Iniciando proceso de entrenamiento...")
    train(
        data="data/tracking/YOLO/tracker.yaml",
        use_run=None,
        augment=True,
    )
