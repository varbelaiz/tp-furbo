#!/bin/bash

# Este script profesional toma 1 argumento:
# 1. La ruta al modelo .pth que se usará.
# Automáticamente creará una carpeta de salida en ./outputs/
# con el MISMO nombre que el archivo .pth (sin la extensión).

# --- Validación de Argumentos ---
if [ "$#" -ne 1 ]; then
    echo "Error: Uso incorrecto."
    echo "Uso: $0 /ruta/al/modelo.pth"
    echo "Ejemplo: $0 ./models/mi_modelo_v3_batch16.pth"
    echo "         (Esto creará automáticamente la salida en ./outputs/mi_modelo_v3_batch16/)"
    exit 1
fi

# --- Deducción de Nombres (La parte prolija) ---
MODELO_ENTRENADO=$1 # Ejemplo: ./models/mi_modelo_v3_batch16.pth

# 1. Extraer solo el nombre del archivo: "mi_modelo_v3_batch16.pth"
MODELO_FILENAME=$(basename -- "$MODELO_ENTRENADO")

# 2. Extraer el nombre sin la extensión .pth: "mi_modelo_v3_batch16"
MODELO_BASENAME="${MODELO_FILENAME%.*}"

# 3. Crear la ruta de salida automáticamente
OUTPUT_DIR="./outputs/$MODELO_BASENAME"

# --- Variables Fijas ---
SPLIT="val" # Vamos a probar en 'val'.
DATASET_PATH="./dataset"

echo "============================================================"
echo "Iniciando Pipeline de Inferencia (Automático)"
echo "  > Modelo:   $MODELO_ENTRENADO"
echo "  > Salida:   $OUTPUT_DIR/$SPLIT (¡Deducido autom!)"
echo "============================================================"
echo "Creando carpeta de salida $OUTPUT_DIR (si no existe)..."
mkdir -p $OUTPUT_DIR # -p crea la carpeta sin fallar si ya existe


# --- PASO 2: 'El Detective' (detect_extremities.py) ---
echo "--- PASO 2: Ejecutando 'El Detective' (detect_extremities.py) ---"

python -m src.detect_extremities \
    --soccernet $DATASET_PATH \
    --prediction $OUTPUT_DIR \
    --split $SPLIT \
    --model_path $MODELO_ENTRENADO \
    --resolution_width 640 \
    --resolution_height 360 #

if [ $? -ne 0 ]; then
    echo "¡El Paso 2 (detect_extremities.py) falló!"
    exit 1
fi


# --- PASO 3: 'El Matemático' (baseline_cameras.py) ---
echo "--- PASO 3: Ejecutando 'El Matemático' (baseline_cameras.py) ---"

python -m src.baseline_cameras \
    --soccernet $DATASET_PATH \
    --prediction $OUTPUT_DIR \
    --split $SPLIT #

if [ $? -ne 0 ]; then
    echo "¡El Paso 3 (baseline_cameras.py) falló!"
    exit 1
fi

echo "============================================================"
echo "¡Pipeline de Inferencia Completo!"
echo "Tus JSONs de cámara están listos en $OUTPUT_DIR/$SPLIT/"
echo "============================================================"