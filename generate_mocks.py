import json
import numpy as np
import os

NUM_FRAMES = 270
DETECTIONS_FILE = "detections_MOCK.json"
CALIBRATION_FILE = "calibration_MOCK.json"
ACTIONS_FILE = "actions_MOCK.json" 

print(f"Generando {NUM_FRAMES} frames de datos mock...")

all_detections = {}
all_calibrations = {}
all_actions = {} 

# --- Configuración de la Calibración (Fácil) ---
identity_matrix = np.eye(3).tolist()
calib_data = {"homography_inverse": identity_matrix}

# --- Configuración de Detecciones (Con Movimiento) ---
# Coordenadas 3D (en metros) iniciales
player_2_pos = np.array([0.0, 0.0])       # Centro del campo
player_3_pos = np.array([-41.5, 0.0])    # Penal izquierdo
ball_pos = np.array([5.0, 5.0])          # Pelota

# --- CAMBIO: Velocidades reducidas ---
player_2_vel = np.array([0.1, 0.05])     # Moviéndose en diagonal
player_3_vel = np.array([0.0, -0.1])     # Moviéndose hacia la línea de banda
ball_vel = np.array([-0.05, 0.1])        # Pelota moviéndose

# --- Bucle Principal de Generación ---
for i in range(NUM_FRAMES):
    frame_key = f"frame_{i:04d}" 
    
    # 1. Guardar la calibración (siempre la misma)
    all_calibrations[frame_key] = calib_data
    
    # 2. Actualizar y guardar las detecciones
    p2_x, p2_y = player_2_pos
    p3_x, p3_y = player_3_pos
    b_x, b_y = ball_pos 
    
    detections_list = [
        {"id": 0, "bbox": [b_x, b_y, b_x, b_y], "jersey_number": None}, 
        {"id": 2, "bbox": [p2_x, p2_y, p2_x, p2_y], "jersey_number": "7"},
        {"id": 3, "bbox": [p3_x, p3_y, p3_x, p3_y], "jersey_number": "5"}
    ]
    all_detections[frame_key] = detections_list
    
    # 3. Mover los jugadores y la pelota para el próximo frame
    player_2_pos += player_2_vel
    player_3_pos += player_3_vel
    ball_pos += ball_vel

    # 4. Generar acciones MOCK
    if i == 100:
        all_actions[frame_key] = {"action": "Pass", "duration": 45}
    
    if i == 200:
        all_actions[frame_key] = {"action": "Shot", "duration": 60}


# --- Guardar los archivos JSON ---
print(f"Guardando {DETECTIONS_FILE}...")
with open(DETECTIONS_FILE, 'w') as f:
    json.dump(all_detections, f, indent=2)

print(f"Guardando {CALIBRATION_FILE}...")
with open(CALIBRATION_FILE, 'w') as f:
    json.dump(all_calibrations, f, indent=2)

print(f"Guardando {ACTIONS_FILE}...") 
with open(ACTIONS_FILE, 'w') as f:
    json.dump(all_actions, f, indent=2)

print("\n¡Archivos mock generados (con posiciones corregidas)!")