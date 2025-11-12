# 🚀 Guía Rápida: Pipeline TPF Homografía

Esta es una guía de inicio rápido para ejecutar el pipeline completo de entrenamiento e inferencia desde cero, asumiendo que la VM está limpia.

## ⚙️ 1. Configuración Inicial (Hacer 1 Sola Vez)

1.  **Clonar el Repositorio:**
    ```bash
    git clone <URL_DE_TU_REPO>
    cd homografia_cancha
    ```

2.  **Instalar Dependencias:**
    ```bash
    # Instalar dependencias del proyecto
    pip install -r requirements.txt
    
    # Instalar herramientas de MLOps
    pip install google-cloud-storage
    pip install tensorboard
    sudo apt install tmux
    ```

## 🏃 2. Pipeline de Ejecución

Sigue estos pasos en orden.

### Paso 0: Descargar el Dataset
El script `descargar_dataset.py` baja los `.zip` a `temp/` y los descomprime en `dataset/`.


```bash
python -m src.descargar_dataset
```

### Paso 1: Pre-Procesar las Máscaras

Genera todas las máscaras (`_masks/`) de antemano. Esto es fundamental para que el `dataloader` sea rápido.

```bash
python -m src.preprocess_masks --dataset_path ./dataset
```

### Paso 2: Entrenar al "Pintor" (El Proceso Largo)

Usamos `tmux` para asegurar que el entrenamiento sobreviva si cerramos la terminal.

1.  **Iniciar la sesión segura:**

    ```bash
    tmux new -s "sesion"
    ```

2.  **Lanzar el entrenamiento (DENTRO de `tmux`):**
    Este comando entrena por 50 épocas, compila el modelo, guarda checkpoints locales en `models/` y sube el mejor a GCS.

    ```bash
    python -m src.train \
        --SoccerNet_path /home/franco/tp-furbo/homografia_cancha \
        --epochs 50 \
        --batch_size 16 \
        --output_folder ./models/run_final_50epochs \
        --gcs_bucket bucket-homo
    ```

3.  **Desconectarse (Dejar corriendo):**
    Una vez que el `tqdm` arranque, presiona: `Ctrl` + `B` (soltá), y luego `D`.

### Paso 3: Inferencia (Detective + Matemático)

Una vez que el entrenamiento termine, usa el `checkpoint_best.pth` para generar las predicciones JSON.

1.  **Dar permisos (Solo la 1ra vez):**

    ```bash
    chmod +x run_inference.sh
    chmod +x run_evaluation.sh
    ```

2.  **Correr la inferencia:**
    El script crea automáticamente la carpeta de salida en `outputs/`.

    ```bash
    ./run_inference.sh ./models/run_final_50epochs/checkpoint_best.pth
    ```

### Paso 4: Evaluación (El Crítico)

Revisa la performance de tu modelo.

```bash
./run_evaluation.sh ./outputs/checkpoint_best
```

*(Busca el `accuracy mean value` en el log de `evaluate_camera.py`).*

## 📊 3. Comandos Útiles de `tmux`

  * **Ver sesiones activas:**
    ```bash
    tmux ls
    ```
  * **Reconectarse a la sesión:**
    ```bash
    tmux attach -t "sesion"
    ```
  * **Matar una sesión (si se trabó):**
    ```bash
    tmux kill-session -t "sesion"
    ```
  * **Matar TODO `tmux`:**
    ```bash
    tmux kill-server
    ```

```
```