import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import sys

class SoccerNetCalibrationDataset(Dataset):
    """
    Dataset MODIFICADO (Versión F - Kornia) para 'Keypoints'
    - NO transforma imágenes (solo las carga como numpy array [H,W,C] uint8).
    - Carga heatmaps .npz pre-calculados ('heatmap' y 'mask').
    - El train.py se encargará de mover a GPU y transformar.
    """
    def __init__(self, root_dir, split, augment=False): 
        
        self.img_path = os.path.join(root_dir, split)
        self.kps_path = os.path.join(root_dir, f"{split}_keypoints_pn")
        
        self.npz_files = sorted(glob.glob(os.path.join(self.kps_path, '*.npz')))
        
        if not self.npz_files:
            print(f"FATAL: No se encontraron archivos .npz en {self.kps_path}")
            sys.exit(1)
            
        self.img_size = (960, 540) 

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        npz_path = self.npz_files[idx]
        file_name = os.path.splitext(os.path.basename(npz_path))[0]
        img_name = os.path.join(self.img_path, file_name + '.jpg')

        try:
            data = np.load(npz_path)
            heatmap = data['heatmap'] 
            mask = data['mask']     

            image = Image.open(img_name).convert('RGB')
            img_np = np.array(image) 
        
        except Exception as e:
            print(f"Error cargando datos (idx {idx}): {img_name} o {npz_path}. Error: {e}")
            return None

        return img_np, heatmap, mask