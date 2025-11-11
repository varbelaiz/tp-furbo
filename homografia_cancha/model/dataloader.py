import os
import glob
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import cv2
import sys

# Importar la función que SÍ necesitamos
try:
    from utils.utils_heatmap import draw_label_map
except ImportError:
    print("Error: No se pudo encontrar 'draw_label_map' en 'utils/utils_heatmap.py'")
    sys.exit(1)

NUM_NET1_CHANNELS = 58 # 57 keypoints + 1 background


class SoccerNetCalibrationDataset(Dataset):
    """
    Dataset MODIFICADO (Versión 3) para la Red 1 (Keypoints)
    Lee .npz y maneja sus propias transformaciones internamente.
    """
    def __init__(self, root_dir, split, augment=False):

        self.img_path = os.path.join(root_dir, split)
        self.kps_path = os.path.join(root_dir, f"{split}_keypoints_pn")
        
        self.npz_files = glob.glob(os.path.join(self.kps_path, '*.npz'))
        
        self.augment = augment
        self.img_size = (960, 540) # (W, H)
        self.heatmap_size = (240, 135) # (W, H) - 1/4 de resolución
        self.sigma = 1 

        # Definir transformaciones internamente
        if self.augment:
            # Transformaciones para ENTRENAMIENTO (con aumentación)
            self.transform = T.Compose([
                T.Resize((self.img_size[1], self.img_size[0])), # (H, W) para T.Resize
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # Transformaciones para VALIDACIÓN (sin aumentación)
            self.transform = T.Compose([
                T.Resize((self.img_size[1], self.img_size[0])), # (H, W) para T.Resize
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # NOTA: RandomHorizontalFlip falta porque requiere lógica de
        # swapping de keypoints que implementaremos si es necesario.
        # Empezar sin flip es más simple y funcionará.

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        npz_path = self.npz_files[idx]
        file_name = os.path.splitext(os.path.basename(npz_path))[0]
        
        try:
            kps_data = np.load(npz_path)
            keypoints = kps_data['keypoints_net1'] # Shape (58, 2)
        except Exception as e:
            print(f"Error cargando .npz: {npz_path}. {e}")
            return self.__getitem__((idx + 1) % len(self))

        # --- 1. Cargar Imagen y Aplicar Transformaciones ---
        img_name = os.path.join(self.img_path, file_name + '.jpg')
        image = Image.open(img_name).convert('RGB')
        img_tensor = self.transform(image)

        # --- 2. Generar Heatmap (Target) y Máscara ---
        heatmap = np.zeros((NUM_NET1_CHANNELS, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)
        mask = np.zeros(NUM_NET1_CHANNELS - 1, dtype=np.float32)

        scale_x = self.heatmap_size[0] / self.img_size[0]
        scale_y = self.heatmap_size[1] / self.img_size[1]

        for kp_id in range(NUM_NET1_CHANNELS - 1): # Iterar 0-56
            pt = keypoints[kp_id]
            
            if pt[0] != -1.0:
                x = int(pt[0] * scale_x)
                y = int(pt[1] * scale_y)
                
                if 0 <= x < self.heatmap_size[0] and 0 <= y < self.heatmap_size[1]:
                    draw_label_map(heatmap[kp_id], (x, y), self.sigma)
                    mask[kp_id] = 1.0

        heatmap[NUM_NET1_CHANNELS - 1] = 1.0 - np.max(heatmap[:-1], axis=0)

        return img_tensor, torch.from_numpy(heatmap).float(), torch.from_numpy(mask).float()