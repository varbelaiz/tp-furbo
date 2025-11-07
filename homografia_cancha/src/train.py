import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
import argparse
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from google.cloud import storage

from src.dataloader import SoccerNetDataset
from src.soccerpitch import SoccerPitch

parser = argparse.ArgumentParser(description='Entrenar modelo de segmentación para SN-Calibration')
parser.add_argument('--SoccerNet_path', type=str, required=True, help='Ruta a la carpeta raíz de SoccerNet-Calibration (la que contiene /dataset)')
parser.add_argument('--epochs', type=int, default=50, help='Número de épocas de entrenamiento')
parser.add_argument('--batch_size', type=int, default=16, help='Tamaño del lote (ajusta según la VRAM de tu GPU)')
parser.add_argument('--lr', type=float, default=0.001, help='Tasa de aprendizaje (Learning Rate)')
parser.add_argument('--output_folder', type=str, default='./models/', help='Carpeta donde se guardarán los checkpoints')
parser.add_argument('--gcs_bucket', type=str, default=None, help='Nombre del Bucket de GCS para guardar el mejor checkpoint (ej. mi-proyecto-bucket-calibracion)') # <-- NUEVO
args = parser.parse_args()

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Sube un archivo a un bucket de GCS"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    print(f"Subiendo {source_file_name} a gs://{bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(source_file_name)
    print("¡Subida a GCS completa!")

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en dispositivo: {device}")

    os.makedirs(args.output_folder, exist_ok=True)

    log_dir = os.path.join(args.output_folder, 'tensorboard_logs')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logs de TensorBoard se guardarán en: {log_dir}")

    print("Cargando dataset de entrenamiento...")
    train_dataset = SoccerNetDataset(datasetpath=os.path.join(args.SoccerNet_path, 'dataset'), split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    print("Cargando dataset de validación...")
    val_dataset = SoccerNetDataset(datasetpath=os.path.join(args.SoccerNet_path, 'dataset'), split="val")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print("Cargando modelo DeeplabV3 (ResNet-50)...")
    
    num_classes = len(SoccerPitch.lines_classes) + 1
    
    model = deeplabv3_resnet50(pretrained=True, progress=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    model.to(device)
    
    print("Compilando el modelo con torch.compile()...")
    model = torch.compile(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.1, verbose=True
    )

    best_val_loss = np.inf

    print(f"--- Iniciando entrenamiento por {args.epochs} épocas ---")
    for epoch in range(args.epochs):
        
        # --- Loop de Entrenamiento ---
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{args.epochs} [Train]")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(train_loss=running_loss/len(progress_bar))
        
        avg_train_loss = running_loss / len(train_loader)

        # --- Loop de Validación ---
        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Época {epoch+1}/{args.epochs} [Valid]")

        with torch.no_grad(): # No necesitamos calcular gradientes en validación
            for images, masks in val_progress_bar:
                images = images.to(device)
                masks = masks.to(device).long()
                
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_progress_bar.set_postfix(val_loss=val_loss/len(val_progress_bar))

        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Época {epoch+1} completada. Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)

        # Loggear los gráficos en TensorBoard
        writer.add_scalar('Loss/train', avg_train_loss, epoch + 1)
        writer.add_scalar('Loss/val', avg_val_loss, epoch + 1)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate', current_lr, epoch + 1)
        writer.flush() # Forzar escritura a disco

        # Checkpointing
        last_model_path = os.path.join(args.output_folder, "checkpoint_last.pth")
        torch.save(model.state_dict(), last_model_path)

        # Si esta es la MEJOR Val Loss hasta ahora, guardalo como "best"
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(args.output_folder, "checkpoint_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"¡Nuevo mejor modelo! Guardando en {best_model_path}")

            if args.gcs_bucket:
                gcs_blob_name = f"{os.path.basename(args.output_folder.rstrip('/'))}/epoch_{epoch+1}_best_loss_{avg_val_loss:.4f}.pth"
                upload_to_gcs(args.gcs_bucket, best_model_path, gcs_blob_name)

    print("Entrenamiento finalizado.")
    
    final_model_path = os.path.join(args.output_folder, "modelo_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Modelo final guardado en: {final_model_path}")

    writer.close()

if __name__ == "__main__":
    train_model()