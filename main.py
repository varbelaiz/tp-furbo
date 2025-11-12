#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
============================================================
PROYECTO TPF: SCRIPT MAESTRO (FASE 3: FUSIÓN Y RENDERIZADO)
============================================================

DOCTRINA (QUÉ HACE ESTE SCRIPT):
Este es el "Director de Orquesta" final del pipeline. Su trabajo
NO es correr la IA (eso es muy lento). Su trabajo es tomar los
resultados PRE-CALCULADOS de las otras fases y fusionarlos en
un video de salida.

ASUME QUE YA EXISTEN:
1.  detections.json (Fase 1 - Detección/Trackeo):
    Un archivo JSON que mapea ID de frame a una lista de
    bounding boxes de jugadores.
    Ej: {"frame_0001": [{"id": 5, "bbox": [x1,y1,x2,y2]}, ...]}

2.  calibration.json (Fase 2 - Calibración):
    Un archivo JSON que mapea ID de frame a la matriz de
    homografía inversa necesaria para la proyección.
    Ej: {"frame_0001": {"homography_inverse": [[...], ... ]}}

PROCESO DE ESTE SCRIPT:
1.  Carga los dos archivos JSON completos en memoria (rápido).
2.  Abre el video de entrada y crea un video de salida (rápido).
3.  Itera sobre cada frame del video (rápido).
4.  Para cada frame:
    a. Busca las detecciones y la homografía en los JSONs.
    b. Calcula el punto de los pies de cada jugador (2D-pixel).
    c. Usa cv2.perspectiveTransform y la homografía para
       mapear los pies 2D a la cancha 3D (X,Y en metros).
    d. Dibuja la "Vista Táctica" (radar) en el frame.
    e. Escribe el frame procesado en el video de salida.
"""

import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm

# --- CONFIGURACIÓN DEL RADAR TÁCTICO ---
# Dimensiones del modelo 3D del mundo (en metros).
# (Basado en utils_ellipse_helpers.py, (105m, 68m) centrado en (0,0))
WORLD_WIDTH = 105.0  # Metros (X)
WORLD_HEIGHT = 68.0   # Metros (Y)
WORLD_X_MIN = -WORLD_WIDTH / 2.0
WORLD_Y_MIN = -WORLD_HEIGHT / 2.0

# Posición y tamaño del radar en el video de salida (en píxeles)
RADAR_W_PX = 400
RADAR_H_PX = int(RADAR_W_PX * (WORLD_HEIGHT / WORLD_WIDTH)) # Mantener aspect ratio
RADAR_MARGIN_PX = 20
RADAR_ALPHA = 0.7 # Transparencia

# Colores (B, G, R)
COLOR_FIELD = (25, 60, 25)
COLOR_LINES = (255, 255, 255)
COLOR_TEAM_A = (0, 100, 255) # Naranja/Rojo
COLOR_TEAM_B = (255, 255, 0) # Cyan/Azul
# ----------------------------------------

def get_player_foot_points(detections: list) -> tuple[np.ndarray, list]:
    """
    Toma la lista de detecciones y extrae el punto central de los pies.
    Formatea los puntos para cv2.perspectiveTransform.
    """
    player_points_2D = []
    player_ids = []

    for det in detections:
        bbox = det['bbox']
        # x_min, y_min, x_max, y_max
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_foot = bbox[3] # Punto más bajo del bounding box

        player_points_2D.append([[x_center, y_foot]])
        player_ids.append(det['id'])

    if not player_ids:
        return None, None
        
    # Formato requerido por cv2.perspectiveTransform: (N, 1, 2)
    return np.array(player_points_2D, dtype=np.float32), player_ids

def world_to_radar_coords(world_x: float, world_y: float, radar_x_offset: int, radar_y_offset: int) -> tuple[int, int]:
    """
    Convierte coordenadas del mundo (metros) a coordenadas del radar (píxeles).
    """
    # Normalizar coordenadas del mundo (0.0 a 1.0)
    norm_x = (world_x - WORLD_X_MIN) / WORLD_WIDTH
    norm_y = (world_y - WORLD_Y_MIN) / WORLD_HEIGHT
    
    # Escalar a píxeles del radar
    radar_x = int(norm_x * RADAR_W_PX)
    radar_y = int(norm_y * RADAR_H_PX)
    
    # Aplicar offset (posición en la pantalla)
    return (radar_x + radar_x_offset, radar_y + radar_y_offset)

def draw_tactical_radar(frame: np.ndarray, player_points_3D: np.ndarray, player_ids: list):
    """
    Dibuja la vista táctica (radar) sobre el frame.
    """
    frame_h, frame_w, _ = frame.shape
    
    # 1. Definir la posición del radar (ej. esquina superior izquierda)
    radar_x0 = RADAR_MARGIN_PX
    radar_y0 = RADAR_MARGIN_PX
    radar_x1 = radar_x0 + RADAR_W_PX
    radar_y1 = radar_y0 + RADAR_H_PX

    # 2. Crear el overlay semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (radar_x0, radar_y0), (radar_x1, radar_y1), COLOR_FIELD, -1)
    
    # (Opcional) Dibujar líneas simples del campo en el radar
    # (ej. línea de medio campo)
    mid_x_radar, _ = world_to_radar_coords(0.0, 0.0, radar_x0, radar_y0)
    cv2.line(overlay, (mid_x_radar, radar_y0), (mid_x_radar, radar_y1), COLOR_LINES, 1)

    # 3. Dibujar los jugadores
    for i, point_3D in enumerate(player_points_3D):
        world_x, world_y = point_3D[0][0], point_3D[0][1]
        player_id = player_ids[i]

        # Convertir metros a píxeles del radar
        radar_x, radar_y = world_to_radar_coords(world_x, world_y, radar_x0, radar_y0)

        # Determinar color (ej. IDs pares vs impares)
        color = COLOR_TEAM_A if player_id % 2 == 0 else COLOR_TEAM_B

        # Dibujar el punto del jugador
        cv2.circle(overlay, (radar_x, radar_y), radius=4, color=color, thickness=-1)

    # 4. Fusionar el overlay con el frame original
    cv2.addWeighted(overlay, RADAR_ALPHA, frame, 1 - RADAR_ALPHA, 0, frame)


def process_video(video_path: str, detections_path: str, calibration_path: str, output_path: str):
    """
    Función principal que lee, procesa y escribe el video.
    """
    # 1. Cargar todos los datos pre-calculados a memoria
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

    # 2. Abrir los streams de video (lectura y escritura)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video de entrada: {video_path}")
        return
        
    # Obtener propiedades del video original
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurar el video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    print(f"Procesando video: {video_path} ({total_frames} frames)")
    pbar = tqdm(total=total_frames, unit="frame")
    
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 3. Buscar los datos pre-calculados para este frame
        # (Usamos .get() para manejar frames sin datos de forma segura)
        str_frame_id = f"frame_{frame_id:04d}" # Asume formato 'frame_0001'
        detections = all_detections.get(str_frame_id)
        calibration = all_calibrations.get(str_frame_id)

        # 4. Si tenemos todos los datos, procesar. Si no, escribir el frame original.
        if detections and calibration and "homography_inverse" in calibration:
            try:
                # 4a. Obtener la homografía inversa
                H_inv = np.array(calibration["homography_inverse"])

                # 4b. Obtener puntos de los jugadores (formato 2D-pixel)
                player_points_2D, player_ids = get_player_foot_points(detections)

                if player_points_2D is not None:
                    # 4c. Proyectar a la cancha (formato 3D-mundo)
                    player_points_3D = cv2.perspectiveTransform(player_points_2D, H_inv)

                    # 4d. Dibujar el radar en el frame
                    draw_tactical_radar(frame, player_points_3D, player_ids)

            except Exception as e:
                print(f"\nError al procesar frame {frame_id}: {e}")
                # Si algo falla (ej. matriz no invertible), escribimos el frame original
                pass
        
        # 5. Escribir el frame (procesado o no) en el video de salida
        writer.write(frame)
        pbar.update(1)
        frame_id += 1

    # 6. Limpiar todo
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