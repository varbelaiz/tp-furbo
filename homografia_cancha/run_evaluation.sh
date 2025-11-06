#!/bin/bash

# Este script profesional toma 1 argumento:
# 1. La ruta a la carpeta de SALIDA (la que contiene los JSONs de predicción).
#    (Ej: ./outputs/mi_modelo_v3_batch16)

# --- Validación de Argumentos ---
if [ "$#" -ne 1 ]; then
    echo "Error: Uso incorrecto."
    echo "Uso: $0 /ruta/a/la/carpeta/de/salida"
    echo "Ejemplo: $0 ./outputs/mi_modelo_v3_batch16"
    exit 1
fi

# --- Asignación de Nombres ---
OUTPUT_DIR=$1 # La carpeta de resultados (ej: ./outputs/mi_modelo_v3_batch16)

# --- Variables Fijas ---
SPLIT="val" # El split que evaluamos (debe coincidir con el de inferencia)
DATASET_PATH="./dataset"
THRESHOLD=5 # Umbral de píxeles para la evaluación

echo "============================================================"
echo "Iniciando 'El Crítico' (Evaluación)"
echo "  > Evaluando predicciones en: $OUTPUT_DIR/$SPLIT"
echo "============================================================"


# --- Evaluación del PASO 2 (El Detective) ---
echo "--- Evaluando Extremidades (evaluate_extremities.py) ---"

python -m src.evaluate_extremities \
    --soccernet $DATASET_PATH \
    --prediction $OUTPUT_DIR \
    --split $SPLIT \
    --threshold $THRESHOLD #

if [ $? -ne 0 ]; then
    echo "¡La evaluación de extremidades (Paso 2) falló!"
    exit 1
fi

echo "------------------------------------------------------------"
echo "--- Evaluando Parámetros de Cámara (evaluate_camera.py) ---"
# --- Evaluación del PASO 3 (El Matemático) ---

python -m src.evaluate_camera \
    --soccernet $DATASET_PATH \
    --prediction $OUTPUT_DIR \
    --split $SPLIT \
    --threshold $THRESHOLD #

if [ $? -ne 0 ]; then
    echo "¡La evaluación de cámara (Paso 3) falló!"
    exit 1
fi

echo "============================================================"
echo "¡Evaluación Completa!"
echo "Revisá los puntajes de Precisión, Recall y Accuracy en la terminal."
echo "============================================================"