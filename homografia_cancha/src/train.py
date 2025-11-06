import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm
import argparse
import os

from src.dataloader import SoccerNetDataset
from src.soccerpitch import SoccerPitch


parser = argparse.ArgumentParser(description='Entrenar modelo de segmentación para SN-Calibration')
parser.add_argument('--SoccerNet_path', type=str, required=True, help='Ruta a la carpeta raíz de SoccerNet-Calibration')
parser.add_argument('--epochs', type=int, default=50, help='Número de épocas de entrenamiento')
parser.add_argument('--batch_size', type=int, default=8, help='Tamaño del lote (ajusta según la VRAM de tu GPU)')
parser.add_argument('--lr', type=float, default=0.001, help='Tasa de aprendizaje (Learning Rate)')
parser.add_argument('--output_path', type=str, default='default_model_path.pth', help='Ruta donde se guardará el modelo entrenado')
args = parser.parse_args()

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en dispositivo: {device}")

    print("Cargando dataset de entrenamiento...")
    train_dataset = SoccerNetDataset(datasetpath=args.SoccerNet_path, split="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    # print("Cargando dataset de validación...")
    # val_dataset = SoccerNetDataset(datasetpath=args.SoccerNet_path, split="val")
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print("Cargando modelo DeeplabV3 (ResNet-50)...")
    
    num_classes = len(SoccerPitch.lines_classes) + 1 
    
    model = deeplabv3_resnet50(pretrained=True, progress=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    
    model.to(device)

    print("Compilando el modelo con torch.compile()... (esto tarda un momento la primera vez)")
    model = torch.compile(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"--- Iniciando entrenamiento por {args.epochs} épocas ---")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Época {epoch+1}/{args.epochs}")
        
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/len(progress_bar))
        
        print(f"Época {epoch+1} completada. Pérdida promedio: {running_loss / len(train_loader)}")

        # Agregar lo mismo pero para val

    print(f"Entrenamiento finalizado. Guardando modelo en: {args.output_path}")
    torch.save(model.state_dict(), args.output_path)

if __name__ == "__main__":
    train_model()