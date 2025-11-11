#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_pn_keypoints.py (Versión Completa)

Script para la Fase 1 (Offline) del pipeline PnLCalib (Plan B).
Genera los datos de Ground Truth para la Red 1 (Keypoints)
y la Red 2 (Líneas).
"""

# 1. Librerías Estándar
import os
import json
import sys
import warnings
import shutil
import argparse
from collections import defaultdict

# 2. Librerías de Terceros
import cv2 
import numpy as np
from tqdm import tqdm
from ellipse import LsqEllipse 

# 3. Imports Locales (de tu proyecto)
from utils.utils_intersections import get_intersections
from utils.utils_ellipse_helpers import INTERSECTON_TO_PITCH_POINTS, get_pitch


# --- Silenciar Advertencias de NumPy ---
warnings.filterwarnings('ignore', category=np.exceptions.RankWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Configuración Global y Constantes ---
DATASET_DIRS = ['dataset/train', 'dataset/val']
OUTPUT_DIRS = {
    'dataset/train': 'dataset/train_keypoints_pn',
    'dataset/val': 'dataset/val_keypoints_pn'
}
INVALID_COORDS = np.array([-1.0, -1.0], dtype=np.float32)

# --- Configuración Red 1 (Keypoints) ---
NUM_NET1_CHANNELS = 58 
NET1_ID_MAP = {point_name: point_id for point_id, point_name in INTERSECTON_TO_PITCH_POINTS.items()}

# --- Configuración Red 2 (Líneas) ---
NET2_LINE_NAMES = [
    'Big rect. left bottom',    # 0
    'Big rect. left main',      # 1
    'Big rect. left top',       # 2
    'Big rect. right bottom',   # 3
    'Big rect. right main',     # 4
    'Big rect. right top',      # 5
    'Circle central',           # 6
    'Circle left',              # 7
    'Circle right',             # 8
    'Goal left crossbar',       # 9
    'Goal left post left ',     # 10 (¡Con el espacio al final!)
    'Goal left post right',     # 11
    'Goal right crossbar',      # 12
    'Goal right post left',     # 13
    'Goal right post right',    # 14
    'Middle line',              # 15
    'Side line bottom',         # 16
    'Side line left',           # 17
    'Side line right',          # 18
    'Side line top',            # 19
    'Small rect. left bottom',  # 20
    'Small rect. left main',    # 21
    'Small rect. left top',     # 22
    'Small rect. right top'     # 23 (Canal 23)
]
NUM_NET2_CHANNELS = 24 
INVALID_LINE = np.array([INVALID_COORDS, INVALID_COORDS], dtype=np.float32)


# --- Parsing de Argumentos ---
parser = argparse.ArgumentParser(description="Generador de Keypoints (Fase 1 - PnLCalib)")
parser.add_argument(
    '--clean', 
    action='store_true', 
    help="Borra los directorios de .npz antes de empezar."
)
args = parser.parse_args()

# --- Lógica de Limpieza (`--clean`) ---
if args.clean:
    print("🚩 Modo --clean activado: Borrando directorios antiguos...")
    for path in OUTPUT_DIRS.values():
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Directorio borrado: {path}")

# --- Funciones Principales ---

def parse_soccernet_json(json_path):
    """
    Parser (Corregido y Simplificado) para el JSON "plano" de SoccerNet.
    """
    try:
        with open(json_path, 'r') as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"\nAdvertencia: No se pudo leer el JSON {json_path}: {e}")
        return defaultdict(list)

    points_dict = defaultdict(list)

    for json_key, points_list in annotations.items():
        if isinstance(points_list, list) and len(points_list) > 0:
            try:
                converted_points = [
                    (point['x'], point['y']) 
                    for point in points_list if 'x' in point and 'y' in point
                ]
                
                if converted_points:
                    points_dict[json_key] = converted_points
            except Exception:
                pass 
    
    return points_dict


def process_single_annotation(json_path):
    """
    Procesa un único archivo JSON y genera los arrays de GT
    para la Red 1 (Keypoints) y la Red 2 (Líneas).
    """
    
    keypoints_net1_output = np.full((NUM_NET1_CHANNELS, 2), INVALID_COORDS, dtype=np.float32)
    keypoints_net2_output = np.full((NUM_NET2_CHANNELS, 2, 2), INVALID_COORDS, dtype=np.float32)

    points_data = parse_soccernet_json(json_path)

    if not points_data:
        return keypoints_net1_output, keypoints_net2_output

    # --- Red 1 (Keypoints) [Lógica de Falaleev] ---
    try:
        gt_points_dict, mask = get_intersections(
            points_data,
            img_size=(960, 540),
            within_image=False, 
            margin=0.0
        )
    except Exception:
        gt_points_dict = {}

    for point_id, coords in gt_points_dict.items():
        if coords is not None and 0 <= point_id <= 56:
            keypoints_net1_output[point_id] = np.array(coords, dtype=np.float32)

    # --- Red 2 (Líneas) [Lógica de PnLCalib] ---
    for i, line_name in enumerate(NET2_LINE_NAMES):
        if line_name in points_data:
            pts = points_data[line_name]
            
            if len(pts) >= 2:
                # El GT son el primer y último punto de la anotación
                start_point = np.array(pts[0], dtype=np.float32)
                end_point = np.array(pts[-1], dtype=np.float32)
                
                # Convertir de coordenadas relativas (0-1) a absolutas (960, 540)
                keypoints_net2_output[i, 0] = start_point * np.array([960.0, 540.0])
                keypoints_net2_output[i, 1] = end_point * np.array([960.0, 540.0])
            
    return keypoints_net1_output, keypoints_net2_output


def main():
    """
    Función principal: itera sobre los directorios, crea carpetas de 
    salida y llama al procesador para cada JSON.
    """
    
    print("Iniciando Fase 1 (Offline): Generación de Keypoints PnLCalib...")
    
    for data_dir in DATASET_DIRS:
        output_dir = OUTPUT_DIRS[data_dir]
        
        if not os.path.exists(data_dir):
            print(f"Advertencia: Directorio de entrada no encontrado: {data_dir}. Saltando.")
            continue
            
        if not os.path.exists(output_dir):
            print(f"Creando directorio de salida: {output_dir}")
            os.makedirs(output_dir)
            
        print(f"\n--- Procesando Directorio: {data_dir} ---")
        
        try:
            annotation_files = [
            f for f in os.listdir(data_dir) 
            if f.endswith('.json') and f != 'match_info.json'
            ]
            if not annotation_files:
                print(f"Error: No se encontraron archivos .json en {data_dir}")
                continue
        except FileNotFoundError:
            print(f"Error: Directorio {data_dir} no existe.")
            continue
            
        for json_file in tqdm(annotation_files, desc=f"Generando KPs para {data_dir}"):
            json_path = os.path.join(data_dir, json_file)
            base_name = os.path.splitext(json_file)[0]
            output_path = os.path.join(output_dir, f"{base_name}.npz")

            if os.path.exists(output_path):
                continue
            
            try:
                kps_net1, kps_net2 = process_single_annotation(json_path)
                
                # Guardar AMBOS arrays en el .npz
                np.savez_compressed(
                    output_path,
                    keypoints_net1=kps_net1,
                    keypoints_net2=kps_net2
                )
                
            except Exception as e:
                print(f"\nError fatal procesando {json_file}: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)

    print("\n--- Generación de Keypoints (Fase 1) Completada. ---")
    print(f"Archivos .npz guardados en: {OUTPUT_DIRS['dataset/train']} y {OUTPUT_DIRS['dataset/val']}")


if __name__ == "__main__":
    main()