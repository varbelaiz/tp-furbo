from ultralytics import YOLO
import os
import wandb

def main():
    # --- CONFIGURATION ---
    MODEL_NAME = 'yolov8x-pose.pt'
    DATA_YAML = 'soccernet.yaml'
    EPOCHS = 100
    IMGSZ = 640
    BATCH_SIZE = 8
    PROJECT_NAME = 'pitch_training'
    RUN_NAME = 'run_cloud_v3' 

    # Check the dataset file is present
    if not os.path.exists(DATA_YAML):
        print(f"ERROR: {DATA_YAML} is missing")
        return

    print("Starting training...")

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

    print("Training finished.")
    print(f"Models saved in: {PROJECT_NAME}/{RUN_NAME}/weights/")

if __name__ == '__main__':
    main()