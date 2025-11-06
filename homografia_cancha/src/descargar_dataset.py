import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
import os
import glob
import shutil
import zipfile  
from tqdm import tqdm

# Descargar los .zip
print("Inicializando descargador...")
downloader = SoccerNetDownloader(LocalDirectory="temp") 

splits_a_bajar = ["train", "valid"]
task_name = "calibration-2023"

try:
    print(f"Iniciando descarga de {task_name}...")
    downloader.downloadDataTask(task=task_name, split=splits_a_bajar)
    print("¡Descarga completa!")

except Exception as e:
    print(f"Error durante la descarga: {e}")
    print("Puede que los archivos ya existan. Continuando...")


# Descomprimirlos
print("--- Iniciando descompresión manual ---")
zip_source_path = os.path.join("temp", task_name) 
unzip_path = "dataset"

os.makedirs(unzip_path, exist_ok=True) 

for split in splits_a_bajar:
    zip_file_path = os.path.join(zip_source_path, f"{split}.zip")
    
    if os.path.exists(zip_file_path):
        print(f"Descomprimiendo {zip_file_path} en '{unzip_path}'...")
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path) 
            print(f"¡{split}.zip descomprimido!")
        except Exception as e:
            print(f"Error al descomprimir {zip_file_path}: {e}")
            continue
    else:
        print(f"ADVERTENCIA: No se encontró {zip_file_path}. ¿Ya estaba descomprimido?")


# Renombrar valid para que sea conciso con el dataloader
path_viejo = os.path.join(unzip_path, "valid")
path_nuevo = os.path.join(unzip_path, "val")

if os.path.exists(path_viejo) and not os.path.exists(path_nuevo):
    print(f"Renombrando '{path_viejo}' a '{path_nuevo}'...")
    os.rename(path_viejo, path_nuevo)
elif os.path.exists(path_nuevo):
    print("La carpeta 'val' ya existe. No se necesita renombrar.")
else:
    print("ADVERTENCIA: No se encontró la carpeta 'valid' para renombrar.")


print("---------------------------------------------------------------")
print("¡Estructura de carpetas lista!")
print(f"Tus datos están listos en la carpeta: '{unzip_path}'")
print(f"Ya podés borrar la carpeta 'temp/' (si querés).")
print("---------------------------------------------------------------")