"""
DataLoader used to train the segmentation network used for the prediction of extremities.
Incluye Data Augmentation (Albumentations) para combatir el overfitting.
"""

import json
import os
import time
from argparse import ArgumentParser

import cv2 as cv
import numpy as np
import albumentations as A  # <-- NUEVO
from torch.utils.data import Dataset
from tqdm import tqdm

from src.soccerpitch import SoccerPitch


class SoccerNetDataset(Dataset):
    def __init__(self,
                 datasetpath,
                 split="test",
                 width=640,
                 height=360,
                 mean="resources/mean.npy",
                 std="resources/std.npy"):
        
        print(f"Inicializando SoccerNetDataset (Split: {split})")
        
        # --- Parámetros de Normalización ---
        self.mean = np.load(mean)
        self.std = np.load(std)
        self.width = width
        self.height = height
        self.split = split  # <-- NUEVO

        # --- Carga de Rutas (Lógica Optimizada) ---
        dataset_dir = os.path.join(datasetpath, split)
        dataset_root = os.path.dirname(dataset_dir)
        mask_dir = os.path.join(dataset_root, f"{split}_masks")

        if not os.path.exists(dataset_dir):
            print(f"Error: El directorio del dataset '{dataset_dir}' no existe.")
            exit(-1)
        if not os.path.exists(mask_dir):
            print(f"Error: La carpeta de máscaras pre-procesadas '{mask_dir}' no existe.")
            print("Por favor, ejecutá 'python -m src.preprocess_masks' primero.")
            exit(-1)

        frames = [f for f in os.listdir(dataset_dir) if ".jpg" in f]

        self.data = []
        
        print(f"Cargando {split} desde '{dataset_dir}' y máscaras desde '{mask_dir}'...")
        for frame in frames:
            frame_index = frame.split(".")[0]
            mask_path = os.path.join(mask_dir, f"{frame_index}.png")
            
            if not os.path.exists(mask_path):
                continue
                
            img_path = os.path.join(dataset_dir, frame)
            
            self.data.append({
                "image_path": img_path,
                "mask_path": mask_path
            })
        
        print(f"Carga completa. {len(self.data)} items encontrados para el split '{split}'.")
        
        # --- INICIO DE MODIFICACIÓN (Albumentations) ---
        # Definir los pipelines de transformación
        self.transform = self.get_transforms(split, width, height)
        # --- FIN DE MODIFICACIÓN ---

    def __len__(self):
        return len(self.data)

    def get_transforms(self, split, width, height):
        """
        Define el pipeline de Augmentation.
        - 'train' recibe augmentations aleatorias.
        - 'val' solo recibe el redimensionamiento.
        """
        if split == 'train':
            # Pipeline para Entrenamiento (con Augmentation)
            return A.Compose([
                # Redimensionar primero
                A.Resize(height, width, interpolation=cv.INTER_LINEAR),
                
                # Augmentations geométricas (NO usamos Flip por la simetría de las etiquetas)
                A.ShiftScaleRotate(
                    shift_limit=0.05, 
                    scale_limit=0.05, 
                    rotate_limit=15, 
                    p=0.5,
                    interpolation=cv.INTER_LINEAR,
                    border_mode=cv.BORDER_CONSTANT, 
                    value=0, # Relleno de imagen (negro)
                    mask_value=0 # Relleno de máscara (background)
                ),
                
                # Augmentations de color (solo se aplican a la imagen)
                A.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                    hue=0.1, 
                    p=0.8
                ),
            ])
        else:
            # Pipeline para Validación (solo redimensionar)
            return A.Compose([
                A.Resize(height, width, interpolation=cv.INTER_LINEAR),
            ])

    def __getitem__(self, index):
        item = self.data[index]

        # --- 1. LECTURA DE DISCO (uint8) ---
        # Cargar imagen (H, W, 3) en BGR (estándar de OpenCV)
        img = cv.imread(item["image_path"]) 
        # Cargar máscara (H_orig, W_orig) en Grayscale (0-28)
        mask = cv.imread(item["mask_path"], cv.IMREAD_GRAYSCALE)
        
        # --- Safety Check (de tu código original) ---
        if mask is None:
            print(f"Error leyendo máscara: {item['mask_path']}")
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
        if img is None:
            print(f"Error leyendo imagen: {item['image_path']}")
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # --- 2. AUGMENTATION (Albumentations) ---
        # Aplicar transformaciones (Resize + Augs en 'train', solo Resize en 'val')
        # 'albumentations' aplica la interpolación correcta (NEAREST) a la máscara
        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']   # (h, w, 3) uint8
        mask = transformed['mask'] # (h, w) uint8

        # --- 3. NORMALIZACIÓN Y TRANSPOSICIÓN ---
        # Ahora que la imagen está augmentada, normalizamos
        img = np.asarray(img, np.float32) / 255.
        img -= self.mean
        img /= self.std
        
        # Transponer de (H, W, C) a (C, H, W) como espera PyTorch
        img = img.transpose((2, 0, 1))
        
        # La máscara ya está lista (H, W)
        
        return img, mask


if __name__ == "__main__":
    
    parser = ArgumentParser(description='Testeo del DataLoader con Augmentations')

    parser.add_argument('--SoccerNet_path', default="./dataset", type=str,
                        help='Path to the SoccerNet dataset folder (la carpeta que contiene train, val, etc.)')
    parser.add_argument('--split', required=False, type=str, default="train", 
                        help='Select the split to test (usa "train" para ver augmentations)')
    parser.add_argument('--num_tests', required=False, type=int, default=5,
                        help='Cuántas imágenes de testeo generar')
    args = parser.parse_args()

    print(f"--- Probando el DataLoader con Augmentations (Split: {args.split}) ---")
    
    start_time = time.time()
    # Usamos el split de 'train' para verificar las augmentations
    soccernet = SoccerNetDataset(args.SoccerNet_path, split=args.split, 
                                 width=640, height=360)
    
    print(f"Dataset cargado en {time.time() - start_time:.2f}s. Total de items: {len(soccernet)}")
    print(f"Se generarán {args.num_tests} imágenes de debug (ej. 'debug_img_0.png')")

    for i in range(args.num_tests):
        if i >= len(soccernet):
            break 
        
        print(f"\n--- Iteración {i} ---")
        img_tensor, mask = soccernet[i]
        
        print(f"Img tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, Clases: {len(np.unique(mask))}")

        # --- DE-NORMALIZACIÓN PARA VISUALIZAR ---
        # 1. Revertir transposición: (C, H, W) -> (H, W, C)
        img_vis = img_tensor.transpose((1, 2, 0))
        # 2. Revertir normalización: (img * std) + mean
        img_vis = (img_vis * soccernet.std) + soccernet.mean
        # 3. Revertir división /255 y asegurar rango [0, 255]
        img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)
        
        # Multiplicar la máscara para que los IDs sean visibles
        mask_vis = (mask * 10).astype(np.uint8)

        # --- GUARDAR EN DISCO (ya que cv.imshow no funciona en la VM) ---
        img_filename = f"debug_img_{i}.png"
        mask_filename = f"debug_mask_{i}.png"
        
        cv.imwrite(img_filename, img_vis)
        cv.imwrite(mask_filename, mask_vis)
        
        print(f"Imágenes de debug guardadas en: {img_filename} y {mask_filename}")

    end_time = time.time()
    print(f"\nTesteo completado en {end_time - start_time:.2f} segundos.")
    print("Verificá los archivos .png en esta carpeta para ver las augmentations.")