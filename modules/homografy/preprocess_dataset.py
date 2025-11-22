import json
import os
import glob
import shutil
import zipfile
from tqdm import tqdm
from SoccerNet.Downloader import SoccerNetDownloader as SNdl

# --- CONFIGURACIÓN DE RUTAS ---
RAW_DATA_DIR = "./soccernet_raw" 
BASE_INPUT_DIR = os.path.join(RAW_DATA_DIR, "calibration-2023")
BASE_OUTPUT_DIR = "./soccernet_dataset"

# --- DICCIONARIO DE PUNTOS ---
MAPPING_DEFINITIONS = {
    0: ("Side line left", "Side line top"),
    1: ("Side line left", "Big rect. left top"),
    2: ("Side line left", "Small rect. left top"),
    3: ("Side line left", "Small rect. left bottom"),
    4: ("Side line left", "Big rect. left bottom"),
    5: ("Side line left", "Side line bottom"),
    6: ("Small rect. left main", "Small rect. left top"),
    7: ("Small rect. left main", "Small rect. left bottom"),
    8: "Penalty mark left", # Point
    9: ("Big rect. left main", "Big rect. left top"),
    10: ("Big rect. left main", "Small rect. left top"), # Virtual
    11: ("Big rect. left main", "Small rect. left bottom"), # Virtual
    12: ("Big rect. left main", "Big rect. left bottom"),
    13: ("Middle line", "Side line top"),
    14: ("Middle line", "Circle central"), # Top Intersection
    15: "Center mark", # Point
    16: ("Middle line", "Circle central"), # Bottom Intersection
    17: ("Middle line", "Side line bottom"),
    18: ("Big rect. right main", "Big rect. right top"),
    19: ("Big rect. right main", "Small rect. right top"), # Virtual
    20: ("Big rect. right main", "Small rect. right bottom"), # Virtual
    21: ("Big rect. right main", "Big rect. right bottom"),
    22: "Penalty mark right", # Point
    23: ("Small rect. right main", "Small rect. right top"),
    24: ("Small rect. right main", "Small rect. right bottom"),
    25: ("Side line right", "Side line top"),
    26: ("Side line right", "Big rect. right top"),
    27: ("Side line right", "Small rect. right top"),
    28: ("Side line right", "Small rect. right bottom"),
    29: ("Side line right", "Big rect. right bottom"),
    30: ("Side line right", "Side line bottom"),
    31: ("Middle line", "Circle central") # Fallback for circle edge if needed
}

# --- FUNCIONES MATEMÁTICAS ---
def get_line_equation(p1, p2):
    x1, y1, x2, y2 = p1['x'], p1['y'], p2['x'], p2['y']
    return y2 - y1, x1 - x2, (y2 - y1)*x1 + (x1 - x2)*y1

def intersect_lines(line1, line2):
    p1a, p1b = line1[0], line1[-1]
    p2a, p2b = line2[0], line2[-1]
    A1, B1, C1 = get_line_equation(p1a, p1b)
    A2, B2, C2 = get_line_equation(p2a, p2b)
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-6: return None
    return (B2 * C1 - B1 * C2) / det, (A1 * C2 - A2 * C1) / det

# --- LOGICA DE DESCARGA ---
def check_and_download():
    print(f"🔍 Verificando datos en: {BASE_INPUT_DIR}...")
    if not os.path.exists(BASE_INPUT_DIR):
        print("⚡ Datos no encontrados. Iniciando descarga...")
        mySNdl = SNdl(LocalDirectory=RAW_DATA_DIR)
        mySNdl.downloadDataTask(task="calibration-2023", split=["train", "valid", "test", "challenge"])
        print("✅ Descarga completada.")
    else:
        print("✅ Los zips ya están descargados.")

# --- PIPELINE OPTIMIZADO ---
def process_pipeline_step(split_name):
    print(f"\n🚀 INICIANDO BLOQUE: {split_name.upper()}")
    
    # 1. DESCOMPRIMIR (Solo este split)
    zip_path = os.path.join(BASE_INPUT_DIR, f"{split_name}.zip")
    extract_path = os.path.join(BASE_INPUT_DIR, split_name)
    
    if not os.path.exists(extract_path):
        print(f"   📦 Descomprimiendo {split_name}.zip...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(BASE_INPUT_DIR)
        except Exception as e:
            print(f"   ❌ Error descomprimiendo {split_name}: {e}")
            return

    # 2. PROCESAR A YOLO
    print(f"   ⚙️ Procesando imágenes y etiquetas para {split_name}...")
    
    # Directorios destino
    output_img_dir = os.path.join(BASE_OUTPUT_DIR, split_name, "images")
    output_lbl_dir = os.path.join(BASE_OUTPUT_DIR, split_name, "labels")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(extract_path, "**/*.json"), recursive=True)
    
    processed_count = 0
    for json_path in tqdm(json_files, desc=f"   Convertendo {split_name}"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            yolo_header = [0, 0.5, 0.5, 1.0, 1.0]
            keypoints = []
            
            # Start Loop 0-31
            for i in range(32):
                px, py, vis = 0.0, 0.0, 0
                
                if i in MAPPING_DEFINITIONS:
                    definition = MAPPING_DEFINITIONS[i]
                    
                    # LOGIC FOR SINGLE POINTS (e.g., Penalty Spot)
                    if isinstance(definition, str):
                        if definition in data:
                            raw = data[definition]
                            # JSON often has list of points [{"x":...}, {"x":...}]
                            if isinstance(raw, list) and len(raw) > 0:
                                pt = raw[0]
                                if 'x' in pt and 'y' in pt:
                                    if 0 <= pt['x'] <= 1.0 and 0 <= pt['y'] <= 1.0:
                                        px, py, vis = pt['x'], pt['y'], 2

                    # LOGIC FOR INTERSECTIONS (e.g., Corners)
                    elif isinstance(definition, tuple):
                        l1_name, l2_name = definition
                        if l1_name in data and l2_name in data:
                            # Your intersect_lines function works on lists of points
                            # The JSON provides exactly that.
                            pt = intersect_lines(data[l1_name], data[l2_name])
                            if pt:
                                rx, ry = pt
                                if 0 <= rx <= 1.0 and 0 <= ry <= 1.0:
                                    px, py, vis = rx, ry, 2

                keypoints.extend([px, py, vis])

            # Guardar solo si hay suficientes puntos para una homografía (mínimo 6)
            if sum(1 for v in keypoints[2::3] if v == 2) >= 6:
                base_name = os.path.splitext(os.path.basename(json_path))[0]
                
                # Crear TXT
                txt_path = os.path.join(output_lbl_dir, base_name + ".txt")
                with open(txt_path, 'w') as out_f:
                    out_f.write(" ".join(map(str, yolo_header + keypoints)))
                
                # Copiar Imagen
                img_src = json_path.replace(".json", ".jpg")
                if not os.path.exists(img_src): img_src = json_path.replace(".json", ".png")
                
                if os.path.exists(img_src):
                    ext = os.path.splitext(img_src)[1]
                    img_dst = os.path.join(output_img_dir, base_name + ext)
                    shutil.copy2(img_src, img_dst)
                    processed_count += 1

        except Exception:
            pass

    print(f"   ✅ {processed_count} imágenes procesadas correctamente.")

    # 3. LIMPIEZA DE ESPACIO
    print(f"   🧹 Liberando espacio: Borrando carpeta temporal {extract_path}...")
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)
    print(f"   ✨ Bloque {split_name} finalizado y limpio.\n")


if __name__ == "__main__":
    # 1. Chequeo y Descarga
    check_and_download()
    
    # 2. Procesamiento Secuencial
    process_pipeline_step("train")
    process_pipeline_step("valid")
    process_pipeline_step("test")
    
    print("\n🎉 Dataset preparado exitosamente en './soccernet_dataset' (y disco limpio)")