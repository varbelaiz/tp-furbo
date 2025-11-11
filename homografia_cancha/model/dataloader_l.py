import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

# Asumimos que esta función existe en tu carpeta utils/
# La vi en tu screenshot: utils/utils_heatmap.py
try:
    from utils.utils_heatmap import draw_label_map
except ImportError:
    print("Error: No se pudo encontrar 'draw_label_map' en 'utils/utils_heatmap.py'")
    print("Esta función es necesaria para generar los heatmaps de GT.")
    sys.exit(1)

# Constantes de nuestro script de Fase 1
NUM_NET2_CHANNELS = 24 # 23 líneas + 1 background


class SoccerNetCalibrationDataset(Dataset):
    """
    Dataset MODIFICADO para la Red 2 (Líneas)
    Lee las extremidades de líneas pre-procesadas desde archivos .npz (Fase 1).
    """

    def __init__(self, root_dir, split, transform, main_cam_only=False):
        
        self.img_path = os.path.join(root_dir, split)
        self.kps_path = os.path.join(root_dir, f"{split}_keypoints_pn")
        
        # Usamos los .npz como fuente de verdad
        self.npz_files = glob.glob(os.path.join(self.kps_path, '*.npz'))
        
        self.transform = transform
        self.img_size = (960, 540) # (W, H)
        self.heatmap_size = (240, 135) # (W, H) - 1/4 de resolución
        self.sigma = 1 # Sigma para el heatmap Gaussiano

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Cargar el .npz
        npz_path = self.npz_files[idx]
        file_name = os.path.splitext(os.path.basename(npz_path))[0]
        
        try:
            kps_data = np.load(npz_path)
            lines = kps_data['keypoints_net2'] # Shape (24, 2, 2)
        except Exception as e:
            print(f"Error cargando .npz: {npz_path}. {e}")
            return self.__getitem__((idx + 1) % len(self))

        # Cargar Imagen
        img_name = os.path.join(self.img_path, file_name + '.jpg')
        image = Image.open(img_name).convert('RGB')

        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)

        # --- Generar Heatmap (Target) ---
        heatmap = np.zeros((NUM_NET2_CHANNELS, self.heatmap_size[1], self.heatmap_size[0]), dtype=np.float32)

        scale_x = self.heatmap_size[0] / self.img_size[0]
        scale_y = self.heatmap_size[1] / self.img_size[1]

        for line_id in range(NUM_NET2_CHANNELS - 1): # Iterar 0-22
            start_pt = lines[line_id, 0]
            end_pt = lines[line_id, 1]
            
            # Dibujar punto de inicio si es válido
            if start_pt[0] != -1.0:
                x_s = int(start_pt[0] * scale_x)
                y_s = int(start_pt[1] * scale_y)
                if 0 <= x_s < self.heatmap_size[0] and 0 <= y_s < self.heatmap_size[1]:
                    draw_label_map(heatmap[line_id], (x_s, y_s), self.sigma)

            # Dibujar punto final si es válido
            if end_pt[0] != -1.0:
                x_e = int(end_pt[0] * scale_x)
                y_e = int(end_pt[1] * scale_y)
                if 0 <= x_e < self.heatmap_size[0] and 0 <= y_e < self.heatmap_size[1]:
                    draw_label_map(heatmap[line_id], (x_e, y_e), self.sigma)

        # Canal de Background
        heatmap[NUM_NET2_CHANNELS - 1] = 1.0 - np.max(heatmap[:-1], axis=0)
        
        # Retornar (input, target) como espera utils_train_l.py
        return sample['image'], torch.from_numpy(heatmap).float()

# --- Las clases WorldCup2014Dataset y TSWorldCupDataset se pueden eliminar ---