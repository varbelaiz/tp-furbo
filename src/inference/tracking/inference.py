from __future__ import annotations

from pathlib import Path
from typing import Optional, Set

import numpy as np
import supervision as sv
from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[3]

BASELINE_RUN_NAME = "train3"

BASELINE_MODEL_PATH = (
    BASE / "runs" / "detect" / BASELINE_RUN_NAME / "weights" / "best.pt"
)
BALL_MODEL_PATH = BASE / "runs" / "ball" / "weights" / "best.pt"

BASELINE_BALL_CLASS_ID = 2
BALL_CANONICAL_CLASS_ID = 2

# Populated on first use by _ensure_models(). The weights are not tracked in the
# repository, so they are loaded lazily and importing this module stays cheap.
_MODELS: Optional[tuple[YOLO, YOLO, dict[int, str]]] = None


def load_models() -> tuple[YOLO, YOLO, dict[int, str]]:
    """Load the baseline and ball models along with their class mapping."""
    baseline_model = YOLO(str(BASELINE_MODEL_PATH))
    ball_model = YOLO(str(BALL_MODEL_PATH))
    class_names = baseline_model.names
    return baseline_model, ball_model, class_names


def _ensure_models() -> tuple[YOLO, YOLO, dict[int, str]]:
    """Return the loaded models, loading them on the first call."""
    global _MODELS
    if _MODELS is None:
        _MODELS = load_models()
    return _MODELS


def get_class_names() -> dict[int, str]:
    """Return the class-id to class-name mapping of the baseline model."""
    return _ensure_models()[2]


def get_player_class_ids() -> Set[int]:
    """Return the class ids the baseline model labels as 'player'."""
    return {
        cid for cid, name in get_class_names().items() if str(name).lower() == "player"
    }


def infer_ball_sliced(frame: np.ndarray, conf: float = 0.1) -> sv.Detections:
    """Run the ball model with slicing over the whole frame."""
    _, ball_model, _ = _ensure_models()
    h, w, _ = frame.shape

    def callback(patch: np.ndarray) -> sv.Detections:
        results = ball_model(patch, conf=conf, verbose=False)[0]
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
    baseline_conf: float = 0.35,
    ball_conf: float = 0.1,
) -> sv.Detections:
    """Run the baseline and ball models and return the merged sv.Detections."""
    baseline_model, _, _ = _ensure_models()

    baseline_result = baseline_model(frame, conf=baseline_conf, verbose=False)[0]
    det_baseline = sv.Detections.from_ultralytics(baseline_result)

    if len(det_baseline) > 0:
        non_ball_mask = det_baseline.class_id != BASELINE_BALL_CLASS_ID
        det_baseline = det_baseline[non_ball_mask]

    det_ball = infer_ball_sliced(frame, conf=ball_conf)
    merged = sv.Detections.merge([det_baseline, det_ball])
    return merged
