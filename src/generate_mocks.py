import json
import numpy as np

NUM_FRAMES = 270
DETECTIONS_FILE = "detections_MOCK.json"
CALIBRATION_FILE = "calibration_MOCK.json"

print(f"Generando {NUM_FRAMES} frames de datos mock...")

all_detections = {}
all_calibrations = {}

# --- Configuración de la Calibración (Fácil) ---
# La homografía inversa es la matriz identidad para todos los frames
identity_matrix = np.eye(3).tolist()
calib_data = {"homography_inverse": identity_matrix}

# --- Configuración de Detecciones (Con Movimiento) ---
# Coordenadas 3D (en metros) iniciales
player_2_pos = np.array([0.0, 0.0])         # Centro del campo
player_3_pos = np.array([-41.5, 0.0])      # Penal izquierdo

# Vectores de movimiento (metros por frame)
player_2_vel = np.array([0.2, 0.1])      # Moviéndose en diagonal
player_3_vel = np.array([0.0, -0.15])    # Moviéndose hacia la línea de banda

# --- Bucle Principal de Generación ---
for i in range(NUM_FRAMES):
    frame_key = f"frame_{i:04d}" # Formato "frame_0000", "frame_0001", etc.
    
    # 1. Guardar la calibración (siempre la misma)
    all_calibrations[frame_key] = calib_data
    
    # 2. Actualizar y guardar las detecciones
    
    # El truco: guardamos las coordenadas 3D (metros) en el campo 'bbox'
    # para que main.py las lea.
    p2_x, p2_y = player_2_pos
    p3_x, p3_y = player_3_pos
    
    detections_list = [
        {"id": 2, "bbox": [p2_x, p2_y, p2_x, p2_y]},
        {"id": 3, "bbox": [p3_x, p3_y, p3_x, p3_y]}
    ]
    
    all_detections[frame_key] = detections_list
    
    # 3. Mover los jugadores para el próximo frame
    player_2_pos += player_2_vel
    player_3_pos += player_3_vel

# --- Guardar los archivos JSON ---
print(f"Guardando {DETECTIONS_FILE}...")
with open(DETECTIONS_FILE, 'w') as f:
    json.dump(all_detections, f, indent=2)

print(f"Guardando {CALIBRATION_FILE}...")
with open(CALIBRATION_FILE, 'w') as f:
    json.dump(all_calibrations, f, indent=2)

print("\n¡Archivos mock generados!")
print("Ahora puedes correr main.py apuntando a estos nuevos JSONs.")