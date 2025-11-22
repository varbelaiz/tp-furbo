from ultralytics import YOLO
import os
import wandb

def main():
    # --- CONFIGURACIÓN ---
    MODEL_NAME = 'yolov11n-pose.pt'
    DATA_YAML = 'soccernet.yaml'
    EPOCHS = 100
    IMGSZ = 640
    BATCH_SIZE = 8
    PROJECT_NAME = 'entrenamiento_cancha'
    RUN_NAME = 'run_cloud_v2' 

    # Verificar dataset
    if not os.path.exists(DATA_YAML):
        print(f"❌ ERROR: Falta {DATA_YAML}")
        return

    print(f"🚀 Iniciando entrenamiento...")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        mosaic=1.0, 
        plots=True,
        amp=False,
        save=True,        # This automatically saves 'best.pt' and 'last.pt'
        save_period=-1,   # -1 disables intermediate checkpoints
        device=0, 
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        workers=0,
        patience=15, 
        optimizer='AdamW' 
    )

    print("✅ Entrenamiento finalizado.")
    print(f"📂 Modelos guardados en: {PROJECT_NAME}/{RUN_NAME}/weights/")

if __name__ == '__main__':
    main()