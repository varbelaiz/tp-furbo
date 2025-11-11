import numpy as np
import glob
import os

files = glob.glob('dataset/train_keypoints_pn/*.npz')
if not files:
    print("¡Error! No se encontraron archivos .npz en dataset/train_keypoints_pn/")
else:
    f = np.load(files[0])
    
    # --- Check Red 1 (Keypoints) ---
    if 'keypoints_net1' in f:
        kps_net1 = f['keypoints_net1']
        print(f"--- Inspeccionando {os.path.basename(files[0])} ---")
        print("\n✅ Red 1 (Keypoints) encontrada.")
        print(f"Shape: {kps_net1.shape} (Esperado: 58, 2)")
        valid_count_net1 = np.sum(kps_net1[:, 0] != -1.0)
        print(f"Puntos válidos: {valid_count_net1} de 57")
        if valid_count_net1 == 0:
            print("⚠️ Alerta: 0 puntos de keypoints encontrados.")
        else:
            print("  (Ej: ID 0:", kps_net1[0], ")")
    else:
        print("❌ Error: 'keypoints_net1' no encontrado en el .npz")

    # --- Check Red 2 (Líneas) ---
    if 'keypoints_net2' in f:
        kps_net2 = f['keypoints_net2']
        print("\n✅ Red 2 (Líneas) encontrada.")
        print(f"Shape: {kps_net2.shape} (Esperado: 24, 2, 2)")
        # Contar líneas válidas (donde el punto inicial 'x' no es -1.0)
        valid_count_net2 = np.sum(kps_net2[:, 0, 0] != -1.0)
        print(f"Líneas válidas: {valid_count_net2} de 23")
        if valid_count_net2 == 0:
             print("⚠️ Alerta: 0 líneas encontradas.")
        else:
            print("  (Ej: ID 0:", kps_net2[0], ")")
    else:
        print("❌ Error: 'keypoints_net2' no encontrado en el .npz")