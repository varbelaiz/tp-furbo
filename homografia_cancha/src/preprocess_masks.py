"""
Script de pre-procesamiento (Solución MLOps).
Itera sobre los splits del dataset, genera las máscaras de segmentación
desde los JSONs y las guarda como archivos .png.
Esto elimina el cuello de botella de CPU en el DataLoader durante el entrenamiento.
"""
import os
import json
import glob
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from src.soccerpitch import SoccerPitch #

def process_split(dataset_root_path, split, target_width=640, target_height=360):
    """
    Procesa un split completo (ej. 'train') y guarda sus máscaras.
    """
    split_dir = os.path.join(dataset_root_path, split)
    output_dir = os.path.join(dataset_root_path, f"{split}_masks")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generando máscaras para '{split}' en '{output_dir}'...")
    
    # Buscamos todas las imágenes, que son la fuente de verdad
    image_files = glob.glob(os.path.join(split_dir, "*.jpg"))
    if not image_files:
        print(f"ADVERTENCIA: No se encontraron archivos .jpg en {split_dir}")
        return

    for img_path in tqdm(image_files, desc=f"Procesando {split}"):
        
        # Derivar nombres de archivos
        base_name = os.path.basename(img_path)
        frame_index = base_name.split(".")[0]
        annotation_file = os.path.join(split_dir, f"{frame_index}.json")

        # Verificar si el JSON de anotación existe
        if not os.path.exists(annotation_file):
            continue

        # Cargar las anotaciones
        with open(annotation_file, "r") as f:
            groundtruth_lines = json.load(f)
        
        # Omitir si no hay líneas en la anotación
        if not groundtruth_lines:
            continue

        # Crear la máscara vacía (target_height, target_width)
        # Exactamente como en el dataloader
        mask = np.zeros((target_height, target_width), dtype=np.uint8)

        # --- LÓGICA DE DIBUJO (copiada de dataloader.py) ---
        #
        for class_number, class_ in enumerate(SoccerPitch.lines_classes):
            if class_ in groundtruth_lines.keys():
                key = class_
                line = groundtruth_lines[key]
                prev_point = line[0]
                for i in range(1, len(line)):
                    next_point = line[i]
                    # Dibujar línea en la máscara usando las dimensiones TARGET
                    cv.line(mask,
                            (int(prev_point["x"] * target_width), int(prev_point["y"] * target_height)),
                            (int(next_point["x"] * target_width), int(next_point["y"] * target_height)),
                            class_number + 1, # ID de clase (1-indexado)
                            2) # Grosor de línea
                    prev_point = next_point
        # --- FIN DE LÓGICA DE DIBUJO ---

        # Guardar la máscara como .png
        output_mask_path = os.path.join(output_dir, f"{frame_index}.png")
        cv.imwrite(output_mask_path, mask)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Pre-procesador de Máscaras para SoccerNet-Calibration')
    
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Ruta a la carpeta raíz del dataset (ej. /data/tp-furbo/homografia_cancha/dataset)')
    
    parser.add_argument('--width', type=int, default=640, 
                        help='Ancho de la máscara final (debe coincidir con el dataloader)') #
    
    parser.add_argument('--height', type=int, default=360, 
                        help='Alto de la máscara final (debe coincidir con el dataloader)') #

    args = parser.parse_args()

    # Procesar los splits que nos importan
    splits_to_process = ["train", "val"] #
    
    for split in splits_to_process:
        process_split(args.dataset_path, split, args.width, args.height)

    print("\n-------------------------------------------------")
    print("¡Pre-procesamiento de máscaras completado!")
    print(f"Nuevas carpetas creadas: '{splits_to_process[0]}_masks' y '{splits_to_process[1]}_masks'")
    print("-------------------------------------------------")