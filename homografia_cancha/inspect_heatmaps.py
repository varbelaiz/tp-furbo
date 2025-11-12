import numpy as np
import glob
import cv2
import os
import sys

NPZ_DIR = "dataset/train_keypoints_pn/"
IMG_DIR = "dataset/train/"
OUTPUT_IMAGE = "verification_output.jpg"

EXPECTED_HM_SHAPE_NET1 = (58, 270, 480)
EXPECTED_HM_SHAPE_NET2 = (24, 270, 480)
EXPECTED_MASK_SHAPE_NET1 = (57,)

print("--- Verificación de Pre-procesamiento (Heatmaps) ---")

npz_files = glob.glob(os.path.join(NPZ_DIR, '*.npz'))
if not npz_files:
    print(f"❌ Error: No se encontraron archivos .npz en {NPZ_DIR}")
    sys.exit(1)

test_npz_path = npz_files[0]
file_name = os.path.splitext(os.path.basename(test_npz_path))[0]
print(f"Inspeccionando archivo: {file_name}.npz")

try:
    data = np.load(test_npz_path)
except Exception as e:
    print(f"❌ Error fatal al cargar {test_npz_path}: {e}")
    sys.exit(1)

errors = False
if 'heatmap_net1' not in data:
    print("❌ Error: Falta la clave 'heatmap_net1'")
    errors = True
elif data['heatmap_net1'].shape != EXPECTED_HM_SHAPE_NET1:
    print(f"❌ Error: Shape de 'heatmap_net1' es {data['heatmap_net1'].shape}, se esperaba {EXPECTED_HM_SHAPE_NET1}")
    errors = True
else:
    print(f"✅ 'heatmap_net1' (Shape: {data['heatmap_net1'].shape})")

if 'mask_net1' not in data:
    print("❌ Error: Falta la clave 'mask_net1'")
    errors = True
elif data['mask_net1'].shape != EXPECTED_MASK_SHAPE_NET1:
    print(f"❌ Error: Shape de 'mask_net1' es {data['mask_net1'].shape}, se esperaba {EXPECTED_MASK_SHAPE_NET1}")
    errors = True
else:
    print(f"✅ 'mask_net1' (Shape: {data['mask_net1'].shape})")

if 'heatmap_net2' not in data:
    print("❌ Error: Falta la clave 'heatmap_net2'")
    errors = True
elif data['heatmap_net2'].shape != EXPECTED_HM_SHAPE_NET2:
    print(f"❌ Error: Shape de 'heatmap_net2' es {data['heatmap_net2'].shape}, se esperaba {EXPECTED_HM_SHAPE_NET2}")
    errors = True
else:
    print(f"✅ 'heatmap_net2' (Shape: {data['heatmap_net2'].shape})")
    
if errors:
    print("--- Verificación fallida. ---")
    sys.exit(1)

try:
    img_path = os.path.join(IMG_DIR, file_name + '.jpg')
    image = cv2.imread(img_path)
    if image is None:
        print(f"❌ Error: No se pudo cargar la imagen original {img_path}")
        sys.exit(1)
    
    image = cv2.resize(image, (960, 540))

    heatmap_net1 = data['heatmap_net1']
    combined_heatmap = np.max(heatmap_net1[:-1], axis=0)

    combined_heatmap_norm = cv2.normalize(combined_heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colored_heatmap = cv2.applyColorMap(combined_heatmap_norm, cv2.COLORMAP_JET)
    
    colored_heatmap_resized = cv2.resize(colored_heatmap, (960, 540))

    overlay = cv2.addWeighted(image, 0.6, colored_heatmap_resized, 0.4, 0)
    
    cv2.imwrite(OUTPUT_IMAGE, overlay)
    
    print(f"\n--- Verificación Visual Guardada ---")
    print(f"✅ Se ha guardado un overlay en: {OUTPUT_IMAGE}")
    print("Descarga esta imagen desde tu VM y ábrela en tu máquina local para verificar que los puntos de calor (heatpoints) tengan sentido.")

except Exception as e:
    print(f"❌ Error durante la verificación visual: {e}")