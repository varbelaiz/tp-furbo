# **FURBO**

## Estructura del proyecto

No están todas las carpetas (`.gitignore`), pero se las paso por WhatsApp, asi que sigan esta estructura asi les corre todo de una.

### `data/`

- `tracking/`:
  - `raw/`: Videos de SoccerNet (SNMOT-060, SNMOT-061, etc.)
  - `YOLO_ball/`: Dataset del baseline pero con augmentation
  - `YOLO_baseline/`: Dataset base para jugadores y referees.

### `runs/`
- `ball/`: Modelos entrenados para detección de pelotas
- `detect/`: Modelos de detección general (el mejor es train4)

### `src/`


#### `src/inference/`

Scripts y notebooks para inferencia:
- `tracking/`: 
  - `inference.ipynb`: Notebook principal para probar modelos entrenados
- `inference.py`: Versión en Python del notebook.**Usen este cuando mergeen todo**
    <mark>OJO: No estoy usando ni embeddings ni tracking — la implementación esta es frame a frame, y lo que hago con segmentación y tracking NO está integrado. Cuando mergeemos hay que revisarlo.</mark>
  - `segmentation.ipynb`: Segmentación de equipos.
  - `tracking.ipynb`: Implementación de tracking
- `homography/`: Transformaciones homográficas

#### `src/train/`
- `tracking/`:
  - `baseline_train.py`: Entrenamiento del modelo baseline
  - `transformation.ipynb`: Transformaciones de datos
  - `ball/`: Entrenamiento específico para detección de pelotas
    - `fine_tuning.py`: Es el baseline finetuneado
    - `transformation.ipynb`: Data augmentation para el modelo de la bocha. 
- `homography/`: Entrenamiento para corrección homográfica

### `segment-anything-2-real-time/`

**NO** está en el repo porque pesa un gazillion de megas. 

Se instala así

```
uv pip install -q git+https://github.com/facebookresearch/segment-anything-2.git

wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

```

Lo uso solo en `src/inference/tracking/tracking.ipynb` a modo de prueba, no creo que lo terminemos usando, pero si quieren correr ese notebook lo van a tener que descargar. 