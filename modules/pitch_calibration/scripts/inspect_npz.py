import glob
import numpy as np
from tqdm import tqdm

print("Inspecting every .npz file...")

train_files = sorted(glob.glob('dataset/train_keypoints_pn/*.npz'))
val_files = sorted(glob.glob('dataset/val_keypoints_pn/*.npz'))
all_files = train_files + val_files

if not all_files:
    print("\nFATAL: no .npz files found in dataset/train_keypoints_pn/ or val_keypoints_pn/")
    exit()

print(f"Inspecting {len(all_files)} .npz files...")

bad_files = []
good_files = 0
expected_mask_shape = (57,) # Expected shape of the 57-keypoint mask

for f_path in tqdm(all_files, desc="Inspeccionando"):
    try:
        f = np.load(f_path)
        
        # --- Check Red 1 (Keypoints) ---
        if 'heatmap_net1' not in f:
            print(f"\nError de Clave: {f_path} (Falta la clave 'heatmap_net1')")
            bad_files.append(f_path)
            continue
            
        if 'mask_net1' not in f:
            print(f"\nError de Clave: {f_path} (Falta la clave 'mask_net1')")
            bad_files.append(f_path)
            continue

        mask_shape = f['mask_net1'].shape
        if mask_shape != expected_mask_shape:
            print(f"\nError de Shape: {f_path} (mask_net1 tiene shape {mask_shape}, se esperaba {expected_mask_shape})")
            bad_files.append(f_path)
            continue

        # --- Check network 2 (lines) ---
        if 'heatmap_net2' not in f:
            print(f"\nError de Clave: {f_path} (Falta la clave 'heatmap_net2')")
            bad_files.append(f_path)
            continue
            
        good_files += 1

    except Exception as e:
        print(f"\nArchivo Ilegible (Zip/Corrupto): {f_path}. Error: {e}")
        bad_files.append(f_path)

print("\n--- INSPECTION REPORT ---")
print(f"Archivos Buenos: {good_files} de {len(all_files)}")
print(f"Archivos Malos/Corruptos: {len(bad_files)}")

if bad_files:
    print("\nCorrupt files (first 50):")
    for i, bad_f in enumerate(bad_files):
        if i >= 50:
            print(f"... and {len(bad_files) - 50} more.")
            break
        print(f"  {bad_f}")
else:
    print("\nAll .npz files are valid (keys and shapes both correct).")