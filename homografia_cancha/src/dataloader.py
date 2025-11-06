"""
DataLoader used to train the segmentation network used for the prediction of extremities.
"""

import json
import os
import time
from argparse import ArgumentParser

import cv2 as cv
import numpy as np
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
        self.mean = np.load(mean)
        self.std = np.load(std)
        self.width = width
        self.height = height


        # --- INICIO DE MODIFICACIÓN ---
        # Definir la carpeta de máscaras pre-procesadas
        dataset_dir = os.path.join(datasetpath, split)

        dataset_root = os.path.dirname(dataset_dir)
        mask_dir = os.path.join(dataset_root, f"{split}_masks")

        if not os.path.exists(dataset_dir): #
            print("Invalid dataset path !")
            exit(-1)
        if not os.path.exists(mask_dir):
            print(f"Error: La carpeta de máscaras pre-procesadas '{mask_dir}' no existe.")
            print("Por favor, ejecutá 'python -m src.preprocess_masks' primero.")
            exit(-1)

        frames = [f for f in os.listdir(dataset_dir) if ".jpg" in f] #

        self.data = []
        # self.n_samples = 0 # (Esta variable no se usaba)
        
        print(f"Cargando {split} desde '{dataset_dir}' y máscaras desde '{mask_dir}'...")
        for frame in frames:

            frame_index = frame.split(".")[0] #
            
            # La nueva "fuente de verdad" es el .png, no el .json
            mask_path = os.path.join(mask_dir, f"{frame_index}.png")
            
            # Si la máscara no existe, salteamos (equivale al check del json vacío)
            if not os.path.exists(mask_path): #
                continue
                
            img_path = os.path.join(dataset_dir, frame) #
            
            # Guardamos las rutas que importan
            self.data.append({
                "image_path": img_path,
                "mask_path": mask_path # Reemplaza "annotations"
            })
        # --- FIN DE MODIFICACIÓN ---

        # dataset_dir = os.path.join(datasetpath, split)
        # if not os.path.exists(dataset_dir):
        #     print("Invalid dataset path !")
        #     exit(-1)

        # frames = [f for f in os.listdir(dataset_dir) if ".jpg" in f]

        # self.data = []
        # self.n_samples = 0
        # for frame in frames:

        #     frame_index = frame.split(".")[0]
        #     annotation_file = os.path.join(dataset_dir, f"{frame_index}.json")
        #     if not os.path.exists(annotation_file):
        #         continue
        #     with open(annotation_file, "r") as f:
        #         groundtruth_lines = json.load(f)
        #     img_path = os.path.join(dataset_dir, frame)
        #     if groundtruth_lines:
        #         self.data.append({
        #             "image_path": img_path,
        #             "annotations": groundtruth_lines,
        #         })

    def __len__(self):
        return len(self.data)

    # def __getitem__(self, index):
    #     item = self.data[index]

    #     img = cv.imread(item["image_path"])
    #     img = cv.resize(img, (self.width, self.height), interpolation=cv.INTER_LINEAR)

    #     mask = np.zeros(img.shape[:-1], dtype=np.uint8)
    #     img = np.asarray(img, np.float32) / 255.
    #     img -= self.mean
    #     img /= self.std
    #     img = img.transpose((2, 0, 1))
    #     for class_number, class_ in enumerate(SoccerPitch.lines_classes):
    #         if class_ in item["annotations"].keys():
    #             key = class_
    #             line = item["annotations"][key]
    #             prev_point = line[0]
    #             for i in range(1, len(line)):
    #                 next_point = line[i]
    #                 cv.line(mask,
    #                         (int(prev_point["x"] * mask.shape[1]), int(prev_point["y"] * mask.shape[0])),
    #                         (int(next_point["x"] * mask.shape[1]), int(next_point["y"] * mask.shape[0])),
    #                         class_number + 1,
    #                         2)
    #                 prev_point = next_point
    #     return img, mask

    def __getitem__(self, index):
        item = self.data[index] #

        # --- LECTURA Y PROCESAMIENTO DE IMAGEN (Sin cambios) ---
        img = cv.imread(item["image_path"]) #
        img = cv.resize(img, (self.width, self.height), interpolation=cv.INTER_LINEAR) #

        img = np.asarray(img, np.float32) / 255. #
        img -= self.mean #
        img /= self.std #
        img = img.transpose((2, 0, 1)) #
        
        # --- INICIO DE MODIFICACIÓN (Lectura de Máscara) ---
        
        # 1. Leer la máscara pre-procesada del disco
        # La leemos en escala de grises (ya es 1 canal)
        mask = cv.imread(item["mask_path"], cv.IMREAD_GRAYSCALE)
        
        # (Opcional, pero buena práctica) Verificar si la máscara se leyó 
        # y tiene las dimensiones correctas
        if mask is None:
            print(f"Error leyendo máscara: {item['mask_path']}")
            # Devolver un tensor vacío para evitar un crash, 
            # aunque el __init__ debería prevenir esto
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
        elif mask.shape[0] != self.height or mask.shape[1] != self.width:
            # Redimensionar si es necesario (no debería pasar si preprocess.py usó bien los args)
            mask = cv.resize(mask, (self.width, self.height), interpolation=cv.INTER_NEAREST)

        # 2. Toda la lógica de dibujo desaparece.

        # --- FIN DE MODIFICACIÓN ---

        return img, mask #


if __name__ == "__main__":

    parser = ArgumentParser(description='dataloader') #

    parser.add_argument('--SoccerNet_path', default="./dataset", type=str,
                        help='Path to the SoccerNet dataset folder (la carpeta que contiene train, val, etc.)')
    # ... (el resto de tus argumentos están bien) ...
    parser.add_argument('--split', required=False, type=str, default="val", help='Select the split of data') #
    args = parser.parse_args()

    print("--- Probando el NUEVO DataLoader con máscaras pre-procesadas ---")
    
    start_time = time.time()
    # Usamos el split de 'val' para probar, es más chico
    soccernet = SoccerNetDataset(args.SoccerNet_path, split=args.split, 
                                 width=320, height=180) # Achicamos para testeo rápido
    
    print(f"Dataset cargado en {time.time() - start_time:.2f}s. Total de items: {len(soccernet)}")

    # Probemos 5 iteraciones
    for i in range(5):
        if i >= len(soccernet):
            break 
        
        print(f"\n--- Iteración {i} ---")
        img, mask = soccernet[i]
        
        print(f"Img shape (tensor): {img.shape}, Img dtype: {img.dtype}, Img min/max: {img.min():.2f}/{img.max():.2f}")
        print(f"Mask shape: {mask.shape}, Mask dtype: {mask.dtype}, Mask N° clases: {len(np.unique(mask))}")

        # --- DE-NORMALIZACIÓN PARA VISUALIZAR ---
        # 1. Revertir transposición: (C, H, W) -> (H, W, C)
        img_vis = img.transpose((1, 2, 0))
        # 2. Revertir normalización: (img * std) + mean
        img_vis = (img_vis * soccernet.std) + soccernet.mean
        # 3. Revertir división /255 y asegurar rango [0, 255]
        img_vis = np.clip(img_vis * 255, 0, 255).astype(np.uint8)
        
        # Multiplicamos la máscara por 10 o 20 para que los IDs de clase (1, 2, 3...)
        # se vuelvan visibles (10, 20, 30...)
        mask_vis = (mask * 10).astype(np.uint8)

        cv.imshow("Imagen (De-Normalizada)", img_vis)
        cv.imshow("Mascara (Pre-Procesada)", mask_vis)
        
        print("Mostrando imagen y máscara. Presioná cualquier tecla para continuar...")
        cv.waitKey(0)
    
    cv.destroyAllWindows()
    end_time = time.time()
    print(f"Testeo completado en {end_time - start_time:.2f} segundos.")


    # # Load the arguments
    # parser = ArgumentParser(description='dataloader')

    # parser.add_argument('--SoccerNet_path', default="./annotations/", type=str,
    #                     help='Path to the SoccerNet-V3 dataset folder')
    # parser.add_argument('--tiny', required=False, type=int, default=None, help='Select a subset of x games')
    # parser.add_argument('--split', required=False, type=str, default="test", help='Select the split of data')
    # parser.add_argument('--num_workers', required=False, type=int, default=4,
    #                     help='number of workers for the dataloader')
    # parser.add_argument('--resolution_width', required=False, type=int, default=1920,
    #                     help='width resolution of the images')
    # parser.add_argument('--resolution_height', required=False, type=int, default=1080,
    #                     help='height resolution of the images')
    # parser.add_argument('--preload_images', action='store_true',
    #                     help="Preload the images when constructing the dataset")
    # parser.add_argument('--zipped_images', action='store_true', help="Read images from zipped folder")

    # args = parser.parse_args()

    # start_time = time.time()
    # soccernet = SoccerNetDataset(args.SoccerNet_path, split=args.split)
    # with tqdm(enumerate(soccernet), total=len(soccernet), ncols=160) as t:
    #     for i, data in t:
    #         img = soccernet[i][0].astype(np.uint8).transpose((1, 2, 0))
    #         print(img.shape)
    #         print(img.dtype)
    #         cv.imshow("Normalized image", img)
    #         cv.waitKey(0)
    #         cv.destroyAllWindows()
    #         print(data[1].shape)
    #         cv.imshow("Mask", soccernet[i][1].astype(np.uint8))
    #         cv.waitKey(0)
    #         cv.destroyAllWindows()
    #         continue
    # end_time = time.time()
    # print(end_time - start_time)
