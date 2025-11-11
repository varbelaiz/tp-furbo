import os
import glob
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import sys

class SoccerNetCalibrationDataset(Dataset):
    """
    Dataset MODIFICADO (Versión 4 - Liviano)
    Lee heatmaps PRE-CALCULADOS desde archivos .npz (Fase 1).
    """
    def __init__(self, root_dir, split, augment=False):

        self.img_path = os.path.join(root_dir, split)
        self.kps_path = os.path.join(root_dir, f"{split}_keypoints_pn")
        
        self.npz_files = glob.glob(os.path.join(self.kps_path, '*.npz'))
        
        self.img_size = (960, 540) # (W, H)
        
        if augment:
            self.transform = T.Compose([
                T.Resize((self.img_size[1], self.img_size[0])), # (H, W)
                # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.img_size[1], self.img_size[0])), # (H, W)
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        npz_path = self.npz_files[idx]
        file_name = os.path.splitext(os.path.basename(npz_path))[0]
        
        try:
            # --- 1. Cargar Heatmaps/Mask PRE-CALCULADOS ---
            data = np.load(npz_path)
            heatmap = data['heatmap_net1']
            mask = data['mask_net1']
        except Exception as e:
            print(f"Error cargando .npz: {npz_path}. {e}")
            return self.__getitem__((idx + 1) % len(self))

        # --- 2. Cargar Imagen y Aplicar Transformaciones ---
        img_name = os.path.join(self.img_path, file_name + '.jpg')
        image = Image.open(img_name).convert('RGB')
        img_tensor = self.transform(image)
        
        # --- 3. NO HAY CÁLCULO DE HEATMAP ---
        # Simplemente convertimos a Tensores
        return img_tensor, torch.from_numpy(heatmap).float(), torch.from_numpy(mask).float()