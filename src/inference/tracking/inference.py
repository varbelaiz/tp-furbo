from __future__ import annotations

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import random

from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[3]

BASELINE_RUN_NAME = "train4"  # Este es el mejor

BASELINE_MODEL_PATH = (
    BASE / "runs" / "detect" / BASELINE_RUN_NAME / "weights" / "best.pt"
)

BALL_MODEL_PATH = (
    BASE / "runs" / "ball" / "weights" / "best.pt"
)

# USTEDES CAMBIEN ESTO DONDE TENGAN SUS IMAGENES DE VALIDACION
VAL_IMAGES_DIR = BASE / "data" / "tracking" / "YOLO_baseline" / "images" / "val"

BASELINE_BALL_CLASS_ID = 2
BALL_CANONICAL_CLASS_ID = 2


def load_models() -> tuple[YOLO, YOLO, dict[int, str]]:
    """Carga los modelos baseline y ball, y devuelve también el mapping de clases."""

    baseline_model = YOLO(str(BASELINE_MODEL_PATH))
    ball_model = YOLO(str(BALL_MODEL_PATH))

    class_names = baseline_model.names  # {id: name}
    return baseline_model, ball_model, class_names


BASELINE_MODEL, BALL_MODEL, CLASS_NAMES = load_models()


def infer_ball_sliced(frame: np.ndarray, conf: float = 0.3) -> sv.Detections:
    """Corre el modelo de pelota con slicing sobre todo el frame."""
    h, w, _ = frame.shape

    def callback(patch: np.ndarray) -> sv.Detections:
        results = BALL_MODEL(patch, conf=conf, verbose=False)[0]
        det = sv.Detections.from_ultralytics(results)
        if len(det) > 0:
            det.class_id = np.full_like(det.class_id, BALL_CANONICAL_CLASS_ID)
        return det

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
        slice_wh=(w // 2 + 100, h // 2 + 100),
        overlap_wh=(100, 100),
        iou_threshold=0.1,
    )

    detections = slicer(frame)
    return detections


def infer_frame(
    frame: np.ndarray,
    baseline_conf: float = 0.25,
    ball_conf: float = 0.3,
) -> sv.Detections:
    """Corre baseline + ball y devuelve un único sv.Detections.

    - Ignora las detecciones de pelota del baseline.
    - Usa solo el modelo fine-tuneado para la pelota.
    """

    baseline_result = BASELINE_MODEL(frame, conf=baseline_conf, verbose=False)[0]
    det_baseline = sv.Detections.from_ultralytics(baseline_result)

    if len(det_baseline) > 0:
        non_ball_mask = det_baseline.class_id != BASELINE_BALL_CLASS_ID
        det_baseline = det_baseline[non_ball_mask]

    det_ball = infer_ball_sliced(frame, conf=ball_conf)

    merged = sv.Detections.merge([det_baseline, det_ball])
    return merged


def get_example_frame() -> np.ndarray:

    jpgs = sorted(VAL_IMAGES_DIR.glob("*.jpg"))


    img_path = random.choice(jpgs) 
    frame = cv2.imread(str(img_path))

    if frame is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    print(f"Using validation image: {img_path}")
    return frame


def main():

    frame = get_example_frame()

    detections = infer_frame(frame)

    bbox_annotator = sv.ColorAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = bbox_annotator.annotate(scene=frame.copy(), detections=detections)

    labels = [
        f"{CLASS_NAMES[int(cls_id)]} {conf:.2f}"
        for cls_id, conf in zip(detections.class_id, detections.confidence)
    ]

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels,
    )

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(annotated_rgb)
    plt.axis("off")
    plt.title("Combined detections (players, referee, ball)")
    # plt.savefig("combined_detections_example.png")
    plt.show()

if __name__ == "__main__":
    main()