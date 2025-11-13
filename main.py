#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
============================================================
PROYECTO TPF: SCRIPT MAESTRO (FASE 5: FUSIÓN CON UTILS MODULAR)
============================================================
"""

import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm
from collections import deque 
import supervision as sv

# --- NUEVAS IMPORTACIONES MODULARES ---
from pitch_annotator import (
    SoccerPitchConfiguration, 
    draw_pitch,
    draw_points_on_pitch,
    draw_highlight_on_pitch
)

# --- CONFIGURACIÓN DE DIBUJO ---
TRAJECTORY_MAX_LEN = 90
RADAR_ALPHA = 0.8
RADAR_MARGIN_PX = 20
RADAR_W_PX = 400 # Ancho fijo para el radar

# --- COLORES DE EQUIPOS Y PELOTA (Estilo Supervision) ---
COLOR_TEAM_A = sv.Color.from_hex('#FF1493') # Rosa
COLOR_TEAM_B = sv.Color.from_hex('#00BFFF') # Azul
COLOR_BALL = sv.Color.from_hex('#FFD700')   # Dorado
COLOR_BORDER = sv.Color.BLACK
COLOR_TEXT = sv.Color.WHITE


def get_player_foot_points(detections: list) -> tuple[np.ndarray, list, list]:
    player_points_2D = []
    player_ids = []
    player_jerseys = [] 
    MOCK_MODE = True 
    for det in detections:
        bbox = det['bbox']
        if MOCK_MODE: x_foot, y_foot = bbox[0], bbox[1]
        else: x_foot, y_foot = (bbox[0] + bbox[2]) / 2.0, bbox[3]
        player_points_2D.append([[x_foot, y_foot]])
        player_ids.append(det['id'])
        player_jerseys.append(det.get('jersey_number', None)) 
    if not player_ids: return None, None, None
    return np.array(player_points_2D, dtype=np.float32), player_ids, player_jerseys


def process_video(video_path: str, detections_path: str, calibration_path: str, actions_path: str, output_path: str):
    print("Cargando archivos JSON a memoria...")
    try:
        with open(detections_path, 'r') as f: all_detections = json.load(f)
    except FileNotFoundError: print(f"Error: No se encontró {detections_path}"); return
    try:
        with open(calibration_path, 'r') as f: all_calibrations = json.load(f)
    except FileNotFoundError: print(f"Error: No se encontró {calibration_path}"); return
    try:
        with open(actions_path, 'r') as f: all_actions = json.load(f)
    except FileNotFoundError: print(f"Error: No se encontró {actions_path}"); return
    print("Datos cargados.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Advertencia: No se pudo abrir {video_path}. Creando video DUMMY.")
        frame_width, frame_height = 1280, 720
        fps = 30
        total_frames = 270 
        def dummy_generator(w, h, num_frames):
            for _ in range(num_frames): yield True, np.zeros((h, w, 3), dtype=np.uint8)
            yield False, None
        cap = dummy_generator(frame_width, frame_height, total_frames)
    else:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    
    print("Configurando el dibujado del campo...")
    # --- CAMBIO: Inicializar Config, no Annotator ---
    PITCH_CONFIG = SoccerPitchConfiguration(width=105.0, height=68.0)
    
    # Dibujar un campo temporal para obtener el 'transformer' y el alto
    temp_radar, transformer = draw_pitch(PITCH_CONFIG, pitch_width_pixels=RADAR_W_PX)
    RADAR_H_PX = temp_radar.shape[0]
    print(f"Radar renderizado a {RADAR_W_PX}x{RADAR_H_PX} px")
    # -------------------------------------------------
    
    
    ball_trajectory_history = deque(maxlen=TRAJECTORY_MAX_LEN)
    current_action_highlight = None
    
    print(f"Procesando video: {video_path} ({total_frames} frames)")
    pbar = tqdm(total=total_frames, unit="frame")
    
    frame_id = 0
    
    while True:
        if isinstance(cap, cv2.VideoCapture): ret, frame = cap.read()
        else: ret, frame = next(cap)
            
        if not ret: break
            
        str_frame_id = f"frame_{frame_id:04d}" 
        detections = all_detections.get(str_frame_id)
        calibration = all_calibrations.get(str_frame_id)
        action_data = all_actions.get(str_frame_id) 

        overlay = frame.copy()

        if detections and calibration and "homography_inverse" in calibration:
            try:
                H_inv = np.array(calibration["homography_inverse"]) 
                player_points_2D, player_ids, player_jerseys = get_player_foot_points(detections)
                
                if player_points_2D is not None:
                    # 1. Transformar puntos 2D -> 3D (Mundo)
                    player_points_3D = cv2.perspectiveTransform(player_points_2D, H_inv)
                    player_points_world = player_points_3D.reshape(-1, 2)
                    
                    # 2. Dibujar el campo base
                    radar_frame, transformer = draw_pitch(PITCH_CONFIG, pitch_width_pixels=RADAR_W_PX)
                    
                    # 3. Separar datos
                    ball_xy, team_a_xy, team_b_xy = [], [], []
                    team_a_jerseys, team_b_jerseys = [], []

                    for i, (world_pos, pid, jersey) in enumerate(zip(player_points_world, player_ids, player_jerseys)):
                        if pid == 0:
                            ball_xy.append(world_pos)
                            ball_trajectory_history.append((world_pos[0], world_pos[1]))
                        elif pid % 2 == 0:
                            team_a_xy.append(world_pos)
                            team_a_jerseys.append(jersey)
                        else:
                            team_b_xy.append(world_pos)
                            team_b_jerseys.append(jersey)
                    
                    ball_xy, team_a_xy, team_b_xy = np.array(ball_xy), np.array(team_a_xy), np.array(team_b_xy)
                    
                    # 4. Comprobar NUEVA acción
                    if action_data:
                        current_action_highlight = {
                            "label": action_data["action"],
                            "frames_remaining": action_data["duration"],
                            "trajectory": list(ball_trajectory_history)
                        }
                    
                    # 5. Dibujar el resaltado
                    if current_action_highlight:
                        radar_frame = draw_highlight_on_pitch(
                            pitch=radar_frame, 
                            transformer=transformer, # Pasar el transformer
                            trajectory=current_action_highlight["trajectory"],
                            label=current_action_highlight["label"]
                        )
                        current_action_highlight["frames_remaining"] -= 1
                        if current_action_highlight["frames_remaining"] <= 0:
                            current_action_highlight = None
                    
                    # 6. Dibujar pelota y jugadores
                    radar_frame = draw_points_on_pitch(
                        pitch=radar_frame, xy=ball_xy, transformer=transformer, 
                        face_color=COLOR_BALL, edge_color=COLOR_BORDER, radius=6, thickness=1
                    )
                    radar_frame = draw_points_on_pitch(
                        pitch=radar_frame, xy=team_a_xy, transformer=transformer, 
                        face_color=COLOR_TEAM_A, edge_color=COLOR_BORDER, radius=9, thickness=1, 
                        text=team_a_jerseys, text_color=COLOR_TEXT
                    )
                    radar_frame = draw_points_on_pitch(
                        pitch=radar_frame, xy=team_b_xy, transformer=transformer, 
                        face_color=COLOR_TEAM_B, edge_color=COLOR_BORDER, radius=9, thickness=1, 
                        text=team_b_jerseys, text_color=COLOR_TEXT
                    )

                    # 7. Pegar el radar en el 'overlay'
                    y0 = RADAR_MARGIN_PX
                    y1 = y0 + RADAR_H_PX
                    x0 = frame_width - RADAR_W_PX - RADAR_MARGIN_PX
                    x1 = x0 + RADAR_W_PX
                    
                    if (y1 <= overlay.shape[0] and x1 <= overlay.shape[1]):
                        overlay[y0:y1, x0:x1] = radar_frame
                    
            except Exception as e:
                print(f"\nError al procesar frame {frame_id}: {e}")
                pass
        
        cv2.addWeighted(overlay, RADAR_ALPHA, frame, 1 - RADAR_ALPHA, 0, frame)
        writer.write(frame)
        pbar.update(1)
        frame_id += 1

    pbar.close()
    if isinstance(cap, cv2.VideoCapture): cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"¡Proceso completado! Video de salida guardado en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 5: Fusión con Utils Modular.")
    parser.add_argument("--video", type=str, required=True, help="Path al video de entrada (ej. video_input.mp4).")
    parser.add_argument("--detections", type=str, required=True, help="Path al JSON de detecciones (ej. detections.json)")
    parser.add_argument("--calibration", type=str, required=True, help="Path al JSON de calibración (ej. calibration.json)")
    parser.add_argument("--actions", type=str, required=True, help="Path al JSON de acciones (ej. actions.json)") 
    parser.add_argument("--output", type=str, required=True, help="Path para guardar el video de salida (ej. video_output.mp4)")
    args = parser.parse_args()

    process_video(args.video, args.detections, args.calibration, args.actions, args.output)