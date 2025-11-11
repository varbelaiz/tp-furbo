#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_pn_keypoints.py (Versión 4 - Pre-calcula Heatmaps)

Script de Fase 1. Pre-calcula y guarda los heatmaps finales.
"""

import os
import json
import sys
import warnings
import shutil
import argparse
from collections import defaultdict
import cv2 
import numpy as np
from tqdm import tqdm
from ellipse import LsqEllipse 

from utils.utils_intersections import get_intersections
from utils.utils_ellipse_helpers import INTERSECTON_TO_PITCH_POINTS, get_pitch

try:
    from utils.utils_heatmap import draw_label_map
except ImportError:
    print("Error: No se pudo encontrar 'draw_label_map' en 'utils/utils_heatmap.py'")
    sys.exit(1)

warnings.filterwarnings('ignore', category=np.exceptions.RankWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATASET_DIRS = ['dataset/train', 'dataset/val']
OUTPUT_DIRS = {
    'dataset/train': 'dataset/train_keypoints_pn',
    'dataset/val': 'dataset/val_keypoints_pn'
}
INVALID_COORDS = np.array([-1.0, -1.0], dtype=np.float32)

# --- Configuración de Heatmap ---
IMG_SIZE = (960, 540) # W, H
HEATMAP_SIZE = (480, 270) # W, H (1/2 de resolución)
SCALE_X = HEATMAP_SIZE[0] / IMG_SIZE[0]
SCALE_Y = HEATMAP_SIZE[1] / IMG_SIZE[1]
SIGMA = 1

# --- Configuración Red 1 (Keypoints) ---
NUM_NET1_CHANNELS = 58 
NET1_ID_MAP = {point_name: point_id for point_id, point_name in INTERSECTON_TO_PITCH_POINTS.items()}

# --- Configuración Red 2 (Líneas) ---
NET2_LINE_NAMES = [
    'Big rect. left bottom', 'Big rect. left main', 'Big rect. left top',
    'Big rect. right bottom', 'Big rect. right main', 'Big rect. right top',
    'Circle central', 'Circle left', 'Circle right', 'Goal left crossbar',
    'Goal left post left ', 'Goal left post right', 'Goal right crossbar',
    'Goal right post left', 'Goal right post right', 'Middle line',
    'Side line bottom', 'Side line left', 'Side line right', 'Side line top',
    'Small rect. left bottom', 'Small rect. left main', 'Small rect. left top',
    'Small rect. right top'
]
NUM_NET2_CHANNELS = 24 
INVALID_LINE = np.array([INVALID_COORDS, INVALID_COORDS], dtype=np.float32)

parser = argparse.ArgumentParser(description="Generador de Keypoints (Fase 1 - PnLCalib)")
parser.add_argument(
    '--clean', 
    action='store_true', 
    help="Borra los directorios de .npz antes de empezar."
)
args = parser.parse_args()

if args.clean:
    print("🚩 Modo --clean activado: Borrando directorios antiguos...")
    for path in OUTPUT_DIRS.values():
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Directorio borrado: {path}")

def parse_soccernet_json(json_path):
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
    Procesa un JSON y genera los HEATMAPS y MASK finales.
    """
    
    # --- Inicializar Salidas ---
    heatmap_net1 = np.zeros((NUM_NET1_CHANNELS, HEATMAP_SIZE[1], HEATMAP_SIZE[0]), dtype=np.float32)
    mask_net1 = np.zeros(NUM_NET1_CHANNELS - 1, dtype=np.float32)
    heatmap_net2 = np.zeros((NUM_NET2_CHANNELS, HEATMAP_SIZE[1], HEATMAP_SIZE[0]), dtype=np.float32)
    
    points_data = parse_soccernet_json(json_path)

    if not points_data:
        return heatmap_net1, mask_net1, heatmap_net2

    # --- Red 1 (Keypoints) [Lógica de Falaleev] ---
    try:
        gt_points_dict, mask = get_intersections(
            points_data,
            img_size=IMG_SIZE,
            within_image=False, 
            margin=0.0
        )
    except Exception:
        gt_points_dict = {}

    for point_id, coords in gt_points_dict.items():
        if coords is not None and 0 <= point_id <= 56:
            pt = coords
            x = int(pt[0] * SCALE_X)
            y = int(pt[1] * SCALE_Y)
            if 0 <= x < HEATMAP_SIZE[0] and 0 <= y < HEATMAP_SIZE[1]:
                draw_label_map(heatmap_net1[point_id], (x, y), SIGMA)
                mask_net1[point_id] = 1.0

    heatmap_net1[NUM_NET1_CHANNELS - 1] = 1.0 - np.max(heatmap_net1[:-1], axis=0)

    # --- Red 2 (Líneas) [Lógica de PnLCalib] ---
    for i, line_name in enumerate(NET2_LINE_NAMES):
        if line_name in points_data:
            pts = points_data[line_name]
            if len(pts) >= 2:
                start_pt_rel = np.array(pts[0], dtype=np.float32)
                end_pt_rel = np.array(pts[-1], dtype=np.float32)
                
                start_pt_abs = start_pt_rel * np.array([IMG_SIZE[0], IMG_SIZE[1]])
                end_pt_abs = end_pt_rel * np.array([IMG_SIZE[0], IMG_SIZE[1]])
                
                x_s = int(start_pt_abs[0] * SCALE_X)
                y_s = int(start_pt_abs[1] * SCALE_Y)
                if 0 <= x_s < HEATMAP_SIZE[0] and 0 <= y_s < HEATMAP_SIZE[1]:
                    draw_label_map(heatmap_net2[i], (x_s, y_s), SIGMA)

                x_e = int(end_pt_abs[0] * SCALE_X)
                y_e = int(end_pt_abs[1] * SCALE_Y)
                if 0 <= x_e < HEATMAP_SIZE[0] and 0 <= y_e < HEATMAP_SIZE[1]:
                    draw_label_map(heatmap_net2[i], (x_e, y_e), SIGMA)
                    
    heatmap_net2[NUM_NET2_CHANNELS - 1] = 1.0 - np.max(heatmap_net2[:-1], axis=0)
            
    return heatmap_net1, mask_net1, heatmap_net2


def main():
    print("Iniciando Fase 1 (Offline): Pre-generando HEATMAPS...")
    
    for data_dir in DATASET_DIRS:
        output_dir = OUTPUT_DIRS[data_dir]
        
        if not os.path.exists(data_dir):
            print(f"Advertencia: Directorio de entrada no encontrado: {data_dir}. Saltando.")
            continue
        if not os.path.exists(output_dir):
            print(f"Creando directorio de salida: {output_dir}")
            os.makedirs(output_dir)
            
        print(f"\n--- Procesando Directorio: {data_dir} ---")
        
        ignore_list = ['match_info.json', 'per_match_info.json']
        try:
            annotation_files = [
                f for f in os.listdir(data_dir) 
                if f.endswith('.json') and f not in ignore_list
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
                # El pre-cálculo de heatmaps ocurre aquí
                heatmap_net1, mask_net1, heatmap_net2 = process_single_annotation(json_path)
                
                np.savez_compressed(
                    output_path,
                    heatmap_net1=heatmap_net1, # (58, 135, 240)
                    mask_net1=mask_net1,       # (57,)
                    heatmap_net2=heatmap_net2  # (24, 135, 240)
                )
                
            except Exception as e:
                print(f"\nError fatal procesando {json_file}: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)

    print("\n--- Generación de HEATMAPS (Fase 1) Completada. ---")
    print(f"Archivos .npz guardados en: {OUTPUT_DIRS['dataset/train']} y {OUTPUT_DIRS['dataset/val']}")


if __name__ == "__main__":
    main()