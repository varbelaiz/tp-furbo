import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repository root
sys.path.append(str(PROJECT_ROOT))

from src.inference.keypoints.detect_keypoints import (
    get_keypoint_detections,
    KEYPOINT_NAMES_29,
    KEYPOINT_NAMES_57,
)
from ultralytics import YOLO

N_KP = 29  # number of keypoints to evaluate: 29 or 57
if N_KP == 29:
    KEYPOINT_NAMES = KEYPOINT_NAMES_29
    OUT_PATH = "unified_output"
    MODEL = "models/keypoints/run_1/weights/best.pt"
else:
    KEYPOINT_NAMES = KEYPOINT_NAMES_57
    OUT_PATH = "unified_output_57"
    MODEL = "models/keypoints/57_kp/weights/best.pt"

KEYPOINT_LIST = [KEYPOINT_NAMES[i] for i in range(N_KP)]


def load_gt_kps(path):
    """Read a YOLO pose label file and return the ground-truth keypoints."""
    vals = list(map(float, open(path).read().split()))
    vals = vals[5:]  # Skip the class index and the bounding box
    kps = np.array(vals).reshape(-1, 3)
    return kps


def compute_l2(pred, gt, w, h):
    """Pixel-wise L2 distance between predicted and ground-truth keypoints."""
    gt_xy = gt[:, :2] * np.array([w, h])
    pred_xy = pred[:, :2]
    return np.sqrt(((pred_xy - gt_xy) ** 2).sum(axis=1))


def evaluate(images, labels, model_path, save_dir="evaluation_outputs"):
    """Evaluate a keypoint model and write the metrics to `save_dir`."""
    model = YOLO(model_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rmse_sum = np.zeros(N_KP)
    rmse_cnt = np.zeros(N_KP)

    vis_tp = 0
    vis_fp = 0
    vis_fn = 0

    # Per-frame confidences and errors, kept for the analysis notebook
    confidences_per_frame = []
    errors_per_frame = []
    frame_indices = []

    image_paths = sorted(Path(images).glob("*.jpg"))

    for im in tqdm(image_paths):
        lbl = Path(labels) / (im.stem + ".txt")
        if not lbl.exists():
            continue

        gt = load_gt_kps(lbl)

        img = cv2.imread(str(im))
        h, w = img.shape[:2]

        _, pred = get_keypoint_detections(model, img)
        if pred is None or len(pred) == 0:
            continue

        pred = pred[0]

        confidences_per_frame.append(pred[:, 2].tolist())
        frame_indices.append(im.stem)

        # Store the pixel-wise errors
        err = compute_l2(pred, gt, w, h)
        errors_per_frame.append(err.tolist())

        # RMSE is accumulated over visible keypoints only
        for i in range(N_KP):
            gt_vis = gt[i, 2] > 0
            pred_vis = pred[i, 2] > 0.4

            if gt_vis:
                rmse_sum[i] += err[i]
                rmse_cnt[i] += 1

                if pred_vis:
                    vis_tp += 1
                else:
                    vis_fn += 1
            else:
                if pred_vis:
                    vis_fp += 1

    # Final RMSE
    rmse = (rmse_sum / np.maximum(rmse_cnt, 1)).tolist()

    # JSON with the headline metrics
    json_out = {
        "rmse_per_keypoint": {
            KEYPOINT_LIST[i]: float(rmse[i]) for i in range(N_KP)
        },
        "visibility_metrics": {
            "precision": float(vis_tp / (vis_tp + vis_fp + 1e-9)),
            "recall": float(vis_tp / (vis_tp + vis_fn + 1e-9)),
            "true_positive": vis_tp,
            "false_positive": vis_fp,
            "false_negative": vis_fn
        }
    }

    with open(save_dir / "results_keypoints.json", "w") as f:
        json.dump(json_out, f, indent=4)

    # RMSE as CSV
    with open(save_dir / "results_keypoints.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["keypoint_name", "rmse"])
        for i in range(N_KP):
            writer.writerow([KEYPOINT_LIST[i], rmse[i]])

    # Visibility metrics as JSON
    with open(save_dir / "visibility.json", "w") as f:
        json.dump(json_out["visibility_metrics"], f, indent=4)

    # Per-frame confidences
    conf_header = ["frame"] + KEYPOINT_LIST
    with open(save_dir / "confidences.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(conf_header)
        for idx, conf_list in zip(frame_indices, confidences_per_frame):
            writer.writerow([idx] + conf_list)

    # Per-frame errors
    err_header = ["frame"] + KEYPOINT_LIST
    with open(save_dir / "errors_by_frame.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(err_header)
        for idx, err_list in zip(frame_indices, errors_per_frame):
            writer.writerow([idx] + err_list)

    print("\nSaved:")
    print(save_dir / "results_keypoints.json")
    print(save_dir / "results_keypoints.csv")
    print(save_dir / "visibility.json")
    print(save_dir / "confidences.csv")
    print(save_dir / "errors_by_frame.csv")


def visualize_samples(images_dir, labels_dir, model_path, out_dir, num_samples=10):
    """
    Render predicted keypoints against the visible ground truth.
    The images are written ready to be embedded in the analysis notebook.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    image_paths = sorted(Path(images_dir).glob("*.jpg"))

    # Pick a few evenly spaced frames
    indices = np.linspace(0, len(image_paths) - 1, num_samples).astype(int)

    for idx in indices:
        im_path = image_paths[idx]
        lbl_path = Path(labels_dir) / (im_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img_bgr = cv2.imread(str(im_path))
        h, w = img_bgr.shape[:2]
        annotated = img_bgr.copy()

        gt = load_gt_kps(lbl_path)

        _, pred = get_keypoint_detections(model, img_bgr)
        if pred is None or len(pred) == 0:
            continue
        pred = pred[0]

        # Predicted keypoints in green
        for (x, y, c) in pred:
            if c > 0.4:
                cv2.circle(annotated, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Visible ground-truth keypoints in red
        for (xn, yn, v) in gt:
            if v > 0:
                px = int(xn * w)
                py = int(yn * h)
                cv2.drawMarker(annotated, (px, py), (0, 0, 255),
                               markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)

        out_file = out_dir / f"{im_path.stem}.jpg"
        cv2.imwrite(str(out_file), annotated)

    print(f"\nVisualizations saved in: {out_dir}")


if __name__ == "__main__":
    IMAGES = PROJECT_ROOT / f"data/calibration/{OUT_PATH}/images/test"
    LABELS = PROJECT_ROOT / f"data/calibration/{OUT_PATH}/labels/test"

    SAVE_DIR = PROJECT_ROOT / f"src/evaluation/keypoints/evaluation_outputs/{N_KP}"
    evaluate(IMAGES, LABELS, MODEL, SAVE_DIR)

    VIS_OUT = PROJECT_ROOT / f"src/evaluation/keypoints/visualizations/{N_KP}"
    visualize_samples(IMAGES, LABELS, MODEL, VIS_OUT, num_samples=10)
