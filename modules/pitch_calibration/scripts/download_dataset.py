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

# splits_a_bajar = ["train", "valid"]
splits_a_bajar = ["test"]
task_name = "calibration-2023"

try:
    print(f"Iniciando descarga de {task_name}...")
    downloader.downloadDataTask(task=task_name, split=splits_a_bajar)
    print("Download complete.")

except Exception as e:
    print(f"Error during download: {e}")
    print("The files may already exist. Continuing...")


# Descomprimirlos
print("--- Extracting the archives ---")
zip_source_path = os.path.join("temp", task_name) 
unzip_path = "../dataset"

os.makedirs(unzip_path, exist_ok=True) 

for split in splits_a_bajar:
    zip_file_path = os.path.join(zip_source_path, f"{split}.zip")
    
    if os.path.exists(zip_file_path):
        print(f"Descomprimiendo {zip_file_path} en '{unzip_path}'...")
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_path) 
            print(f"{split}.zip extracted.")
        except Exception as e:
            print(f"Error extracting {zip_file_path}: {e}")
            continue
    else:
        print(f"WARNING: {zip_file_path} not found. Was it already extracted?")


# Rename 'valid' to 'val' so it matches what the dataloader expects
path_viejo = os.path.join(unzip_path, "valid")
path_nuevo = os.path.join(unzip_path, "val")

if os.path.exists(path_viejo) and not os.path.exists(path_nuevo):
    print(f"Renombrando '{path_viejo}' a '{path_nuevo}'...")
    os.rename(path_viejo, path_nuevo)
elif os.path.exists(path_nuevo):
    print("The 'val' folder already exists, nothing to rename.")
else:
    print("WARNING: no 'valid' folder found to rename.")


print("---------------------------------------------------------------")
print("Folder structure ready.")
print(f"The data is ready in: '{unzip_path}'")
print("The 'temp/' folder can now be deleted.")
print("---------------------------------------------------------------")