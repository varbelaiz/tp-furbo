#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
============================================================
PROYECTO TPF: SCRIPT MAESTRO (FASE 3: FUSIÓN Y RENDERIZADO)
============================================================
... (la doctrina sigue igual) ...
"""

import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm

# --- Importar utilidades del radar ---
from utils.radar_utils import (
    get_player_foot_points, 
    precompute_radar_field_lines,
    draw_tactical_radar
)


def process_video(video_path: str, detections_path: str, calibration_path: str, output_path: str):
    print("Cargando archivos JSON a memoria...")
    try:
        with open(detections_path, 'r') as f:
            all_detections = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de detecciones: {detections_path}")
        return
    try:
        with open(calibration_path, 'r') as f:
            all_calibrations = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de calibración: {calibration_path}")
        return
    print("Datos cargados.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video de entrada: {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    
    print("Pre-calculando líneas del radar (coordenadas locales)...")
    field_coords = precompute_radar_field_lines()
    
    print(f"Procesando video: {video_path} ({total_frames} frames)")
    pbar = tqdm(total=total_frames, unit="frame")
    
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        str_frame_id = f"frame_{frame_id:04d}" 
        detections = all_detections.get(str_frame_id)
        calibration = all_calibrations.get(str_frame_id)

        if detections and calibration and "homography_inverse" in calibration:
            try:
                H_inv = np.array(calibration["homography_inverse"])
                player_points_2D, player_ids = get_player_foot_points(detections)

                if player_points_2D is not None:
                    player_points_3D = cv2.perspectiveTransform(player_points_2D, H_inv)
                    
                    draw_tactical_radar(
                        frame, 
                        player_points_3D, 
                        player_ids, 
                        field_coords, 
                        frame_width 
                    )

            except Exception as e:
                print(f"\nError al procesar frame {frame_id}: {e}")
                pass
        
        writer.write(frame)
        pbar.update(1)
        frame_id += 1

    pbar.close()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"¡Proceso completado! Video de salida guardado en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 3: Fusión y Renderizado de Analíticas de Fútbol.")
    parser.add_argument("--video", type=str, required=True, help="Path al video de entrada (ej. video_input.mp4)")
    parser.add_argument("--detections", type=str, required=True, help="Path al JSON de detecciones (ej. detections.json)")
    parser.add_argument("--calibration", type=str, required=True, help="Path al JSON de calibración (ej. calibration.json)")
    parser.add_argument("--output", type=str, required=True, help="Path para guardar el video de salida (ej. video_output.mp4)")
    args = parser.parse_args()

    process_video(args.video, args.detections, args.calibration, args.output)