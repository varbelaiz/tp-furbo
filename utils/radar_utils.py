#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
============================================================
PROYECTO TPF: UTILIDADES PARA RADAR TÁCTICO (CORREGIDO)
============================================================
Contiene todas las funciones auxiliares para pre-calcular
y dibujar el radar táctico 2D.
"""

import cv2
import math
import numpy as np

# --- CONFIGURACIÓN DEL RADAR TÁCTICO ---
# Dimensiones del modelo 3D del mundo (en metros).
WORLD_WIDTH = 105.0
WORLD_HEIGHT = 68.0
WORLD_X_MIN = -WORLD_WIDTH / 2.0 # -52.5
WORLD_Y_MIN = -WORLD_HEIGHT / 2.0 # -34.0

# Posición y tamaño del radar en el video de salida (en píxeles)
RADAR_W_PX = 400
RADAR_H_PX = int(RADAR_W_PX * (WORLD_HEIGHT / WORLD_WIDTH)) # 258
RADAR_MARGIN_PX = 20
RADAR_ALPHA = 0.7 # Transparencia

# Colores (B, G, R)
COLOR_FIELD = (25, 60, 25)
COLOR_LINES = (255, 255, 255)
COLOR_TEAM_A = (0, 100, 255) # Naranja/Rojo
COLOR_TEAM_B = (255, 255, 0) # Cyan/Azul
# ----------------------------------------


def get_player_foot_points(detections: list) -> tuple[np.ndarray, list]:
    """ Extrae los puntos de los pies (x_center, y_bottom) de las detecciones. """
    player_points_2D = []
    player_ids = []
    for det in detections:
        bbox = det['bbox']
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_foot = bbox[3]
        player_points_2D.append([[x_center, y_foot]])
        player_ids.append(det['id'])
    if not player_ids:
        return None, None
    return np.array(player_points_2D, dtype=np.float32), player_ids

def world_to_radar_coords(world_x: float, world_y: float) -> tuple[int, int]:
    """ 
    Convierte coordenadas del mundo (metros) a coordenadas del 
    radar LOCAL (píxeles 0-W, 0-H).
    """
    norm_x = (world_x - WORLD_X_MIN) / WORLD_WIDTH
    norm_y = (world_y - WORLD_Y_MIN) / WORLD_HEIGHT
    
    radar_x = int(norm_x * (RADAR_W_PX - 1))
    radar_y = int(norm_y * (RADAR_H_PX - 1))
    
    return (radar_x, radar_y)

def precompute_radar_field_lines():
    """
    Pre-calcula todas las coordenadas de las líneas del campo en el 
    espacio de píxeles del radar LOCAL (0-W, 0-H).
    ¡YA NO USA OFFSETS!
    """
    coords = {}
    
    # --- Coordenadas del Mundo (en metros, centrado en 0,0) ---
    PITCH_LENGTH = 105.0
    PITCH_WIDTH = 68.0
    PENALTY_AREA_WIDTH = 40.32
    PENALTY_AREA_LENGTH = 16.5
    GOAL_AREA_WIDTH = 18.32
    GOAL_AREA_LENGTH = 5.5
    CENTER_CIRCLE_RADIUS = 9.15
    GOAL_LINE_TO_PENALTY_MARK = 11.0

    # Función de ayuda interna (World to Radar LOCAL)
    def w2r(x, y):
        # ¡Llama a la versión simple sin offsets!
        return world_to_radar_coords(x, y) 

    # --- Límites del Campo ---
    coords['touchline_top'] = (w2r(-PITCH_LENGTH/2, -PITCH_WIDTH/2), w2r(PITCH_LENGTH/2, -PITCH_WIDTH/2))
    coords['touchline_bottom'] = (w2r(-PITCH_LENGTH/2, PITCH_WIDTH/2), w2r(PITCH_LENGTH/2, PITCH_WIDTH/2))
    coords['goal_line_left'] = (w2r(-PITCH_LENGTH/2, -PITCH_WIDTH/2), w2r(-PITCH_LENGTH/2, PITCH_WIDTH/2))
    coords['goal_line_right'] = (w2r(PITCH_LENGTH/2, -PITCH_WIDTH/2), w2r(PITCH_LENGTH/2, PITCH_WIDTH/2))
    
    # --- Línea de Medio Campo ---
    coords['half_way_line'] = (w2r(0, -PITCH_WIDTH/2), w2r(0, PITCH_WIDTH/2))
    
    # --- Círculo Central ---
    coords['center_circle_center'] = w2r(0, 0)
    # Los ejes (radios) son los mismos, no necesitan offset
    radius_x_px = int(RADAR_W_PX * (CENTER_CIRCLE_RADIUS / WORLD_WIDTH))
    radius_y_px = int(RADAR_H_PX * (CENTER_CIRCLE_RADIUS / WORLD_HEIGHT))
    coords['center_circle_axes'] = (radius_x_px, radius_y_px)

    # --- Área Penal Izquierda ---
    l_pa_x1 = -PITCH_LENGTH / 2
    l_pa_x2 = -PITCH_LENGTH / 2 + PENALTY_AREA_LENGTH
    l_pa_y1 = -PENALTY_AREA_WIDTH / 2
    l_pa_y2 = PENALTY_AREA_WIDTH / 2
    coords['l_penalty_box'] = (w2r(l_pa_x1, l_pa_y1), w2r(l_pa_x2, l_pa_y2)) 
    
    # --- Área Penal Derecha ---
    r_pa_x1 = PITCH_LENGTH / 2 - PENALTY_AREA_LENGTH
    r_pa_x2 = PITCH_LENGTH / 2
    r_pa_y1 = -PENALTY_AREA_WIDTH / 2
    r_pa_y2 = PENALTY_AREA_WIDTH / 2
    coords['r_penalty_box'] = (w2r(r_pa_x1, r_pa_y1), w2r(r_pa_x2, r_pa_y2))

    # --- Área de Gol Izquierda ---
    l_ga_x1 = -PITCH_LENGTH / 2
    l_ga_x2 = -PITCH_LENGTH / 2 + GOAL_AREA_LENGTH
    l_ga_y1 = -GOAL_AREA_WIDTH / 2
    l_ga_y2 = GOAL_AREA_WIDTH / 2
    coords['l_goal_box'] = (w2r(l_ga_x1, l_ga_y1), w2r(l_ga_x2, l_ga_y2)) 
    
    # --- Área de Gol Derecha ---
    r_ga_x1 = PITCH_LENGTH / 2 - GOAL_AREA_LENGTH
    r_ga_x2 = PITCH_LENGTH / 2
    r_ga_y1 = -GOAL_AREA_WIDTH / 2
    r_ga_y2 = GOAL_AREA_WIDTH / 2
    coords['r_goal_box'] = (w2r(r_ga_x1, r_ga_y1), w2r(r_ga_x2, r_ga_y2))
    
    # --- Arcos de Penal ---
    l_penalty_mark_x = -PITCH_LENGTH / 2 + GOAL_LINE_TO_PENALTY_MARK
    r_penalty_mark_x = PITCH_LENGTH / 2 - GOAL_LINE_TO_PENALTY_MARK
    coords['l_penalty_arc_center'] = w2r(l_penalty_mark_x, 0)
    coords['r_penalty_arc_center'] = w2r(r_penalty_mark_x, 0)
    
    x_rel = PENALTY_AREA_LENGTH - GOAL_LINE_TO_PENALTY_MARK
    angle_rad = math.acos(x_rel / CENTER_CIRCLE_RADIUS)
    angle_deg = math.degrees(angle_rad)
    
    coords['l_arc_angles'] = (float(360.0 - angle_deg), float(angle_deg))
    coords['r_arc_angles'] = (float(180.0 - angle_deg), float(180.0 + angle_deg))

    return coords


def draw_tactical_radar(frame: np.ndarray, player_points_3D: np.ndarray, 
                          player_ids: list, field_coords: dict, frame_width: int):
    """
    Dibuja la vista táctica (radar) "BONITA" sobre el frame.
    ¡AHORA APLICA EL OFFSET AQUÍ!
    """
    # 1. Definir la posición del radar (TOP-RIGHT)
    radar_y0 = RADAR_MARGIN_PX
    radar_x0 = frame_width - RADAR_W_PX - RADAR_MARGIN_PX 
    radar_x1 = radar_x0 + RADAR_W_PX
    radar_y1 = radar_y0 + RADAR_H_PX

    # 2. Crear el overlay semi-transparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (radar_x0, radar_y0), (radar_x1, radar_y1), COLOR_FIELD, -1)
    
    # 3. Dibujar las líneas del campo APLICANDO EL OFFSET
    fc = field_coords # Alias corto
    line_thick = 1 
    
    # --- Funciones helper para aplicar offset ---
    def add_offset(pt):
        return (pt[0] + radar_x0, pt[1] + radar_y0)
    
    def add_offset_rect(pt1, pt2):
        return (add_offset(pt1), add_offset(pt2))

    # Líneas exteriores
    pt1, pt2 = add_offset_rect(fc['touchline_top'][0], fc['touchline_top'][1])
    cv2.line(overlay, pt1, pt2, COLOR_LINES, line_thick)
    pt1, pt2 = add_offset_rect(fc['touchline_bottom'][0], fc['touchline_bottom'][1])
    cv2.line(overlay, pt1, pt2, COLOR_LINES, line_thick)
    pt1, pt2 = add_offset_rect(fc['goal_line_left'][0], fc['goal_line_left'][1])
    cv2.line(overlay, pt1, pt2, COLOR_LINES, line_thick)
    pt1, pt2 = add_offset_rect(fc['goal_line_right'][0], fc['goal_line_right'][1])
    cv2.line(overlay, pt1, pt2, COLOR_LINES, line_thick)
    
    # Línea de medio campo
    pt1, pt2 = add_offset_rect(fc['half_way_line'][0], fc['half_way_line'][1])
    cv2.line(overlay, pt1, pt2, COLOR_LINES, line_thick)
    
    # Círculo central
    center = add_offset(fc['center_circle_center'])
    cv2.ellipse(overlay, center, fc['center_circle_axes'], 0, 0, 360, COLOR_LINES, line_thick)
    
    # Áreas penales
    pt1, pt2 = add_offset_rect(fc['l_penalty_box'][0], fc['l_penalty_box'][1])
    cv2.rectangle(overlay, pt1, pt2, COLOR_LINES, line_thick)
    pt1, pt2 = add_offset_rect(fc['r_penalty_box'][0], fc['r_penalty_box'][1])
    cv2.rectangle(overlay, pt1, pt2, COLOR_LINES, line_thick)

    # Áreas de gol
    pt1, pt2 = add_offset_rect(fc['l_goal_box'][0], fc['l_goal_box'][1])
    cv2.rectangle(overlay, pt1, pt2, COLOR_LINES, line_thick)
    pt1, pt2 = add_offset_rect(fc['r_goal_box'][0], fc['r_goal_box'][1])
    cv2.rectangle(overlay, pt1, pt2, COLOR_LINES, line_thick)
    
    # Arcos de penal
    # Arcos de penal (reemplazar las líneas corruptas por estas)
    center_l = add_offset(fc['l_penalty_arc_center'])
    center_r = add_offset(fc['r_penalty_arc_center'])

    # Dibuja arcos (startAngle, endAngle tomados de fc)
    cv2.ellipse(overlay,
                center_l,
                fc['center_circle_axes'],
                0,
                fc['l_arc_angles'][0],
                fc['l_arc_angles'][1],
                COLOR_LINES,
                line_thick)

    cv2.ellipse(overlay,
                center_r,
                fc['center_circle_axes'],
                0,
                fc['r_arc_angles'][0],
                fc['r_arc_angles'][1],
                COLOR_LINES,
                line_thick)
    
    # 4. Dibujar jugadores
    for i, point_3D in enumerate(player_points_3D):
        world_x, world_y = point_3D[0][0], point_3D[0][1]
        player_id = player_ids[i]
        
        # Convierte mundo a radar local (0-400)
        radar_x_local, radar_y_local = world_to_radar_coords(world_x, world_y)
        
        # Añade el offset para la posición global
        radar_x_global = radar_x_local + radar_x0
        radar_y_global = radar_y_local + radar_y0
        
        if (radar_x0 < radar_x_global < radar_x1) and (radar_y0 < radar_y_global < radar_y1):
            color = COLOR_TEAM_A if player_id % 2 == 0 else COLOR_TEAM_B
            cv2.circle(overlay, (radar_x_global, radar_y_global), radius=4, color=color, thickness=-1)

    # 5. Aplicar el overlay al frame
    cv2.addWeighted(overlay, RADAR_ALPHA, frame, 1 - RADAR_ALPHA, 0, frame)