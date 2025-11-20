from ultralytics import YOLO
import os
import wandb

def main():
    # --- CONFIGURACIÓN ---
    MODEL_NAME = 'yolov8x-pose.pt'
    DATA_YAML = 'soccernet.yaml'
    EPOCHS = 100
    IMGSZ = 640
    BATCH_SIZE = 16 
    PROJECT_NAME = 'entrenamiento_cancha'
    RUN_NAME = 'run_cloud_v1' # Le puse cloud para diferenciar

    # Verificar dataset
    if not os.path.exists(DATA_YAML):
        print(f"❌ ERROR: Falta {DATA_YAML}")
        return

    # Iniciar W&B
    wandb.init(project="homografia-cancha", name=RUN_NAME, job_type="training")

    print(f"🚀 Iniciando entrenamiento...")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        mosaic=0.0, 
        plots=True,
        save=True,
        device=0, 
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,
        workers=8,
        patience=15, # Bajamos paciencia para ahorrar plata si se estanca
        optimizer='AdamW' # A veces converge más rápido en Pose
    )
    
    # Finalizar run de W&B
    wandb.finish()

    print("✅ Entrenamiento finalizado.")

if __name__ == '__main__':
    main()