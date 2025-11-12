import numpy as np
import glob
import os
from tqdm import tqdm

print("Iniciando inspección de TODOS los archivos .npz (versión FINAL, con chequeo de shape)...")

train_files = sorted(glob.glob('dataset/train_keypoints_pn/*.npz'))
val_files = sorted(glob.glob('dataset/val_keypoints_pn/*.npz'))
all_files = train_files + val_files

if not all_files:
    print("\n¡ERROR FATAL! No se encontraron archivos .npz en dataset/train_keypoints_pn/ o val_keypoints_pn/")
    exit()

print(f"Se inspeccionarán {len(all_files)} archivos .npz...")

bad_files = []
good_files = 0
expected_mask_shape = (57,) # El shape esperado para la máscara de 57 keypoints

for f_path in tqdm(all_files, desc="Inspeccionando"):
    try:
        f = np.load(f_path)
        
        # --- Check Red 1 (Keypoints) ---
        if 'heatmap_net1' not in f:
            print(f"\n❌ Error de Clave: {f_path} (Falta la clave 'heatmap_net1')")
            bad_files.append(f_path)
            continue
            
        if 'mask_net1' not in f:
            print(f"\n❌ Error de Clave: {f_path} (Falta la clave 'mask_net1')")
            bad_files.append(f_path)
            continue

        mask_shape = f['mask_net1'].shape
        if mask_shape != expected_mask_shape:
            print(f"\n❌ Error de Shape: {f_path} (mask_net1 tiene shape {mask_shape}, se esperaba {expected_mask_shape})")
            bad_files.append(f_path)
            continue

        # --- Check Red 2 (Líneas) ---
        if 'heatmap_net2' not in f:
            print(f"\n❌ Error de Clave: {f_path} (Falta la clave 'heatmap_net2')")
            bad_files.append(f_path)
            continue
            
        good_files += 1

    except Exception as e:
        print(f"\n🚨 Archivo Ilegible (Zip/Corrupto): {f_path}. Error: {e}")
        bad_files.append(f_path)

print("\n--- REPORTE DE INSPECCIÓN ---")
print(f"Archivos Buenos: {good_files} de {len(all_files)}")
print(f"Archivos Malos/Corruptos: {len(bad_files)}")

if bad_files:
    print("\nLista de archivos corruptos (primeros 50):")
    for i, bad_f in enumerate(bad_files):
        if i >= 50:
            print(f"... y {len(bad_files) - 50} más.")
            break
        print(f"  {bad_f}")
else:
    print("\n✅ ¡Felicitaciones! Todos los archivos .npz son válidos (claves Y shapes correctos).")