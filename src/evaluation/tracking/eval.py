import random
import re
import sys
import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import supervision as sv
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# Render plots to file without opening a window (headless)
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repository root
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.tracking.utils import (
    load_clip_frames_and_gt,
    match_class_boxes,
    match_for_confusion,
    build_pr_curve_monotone,
    calculate_f1,
)

# --- Configuration ---
BASELINE_MODEL_PATH = PROJECT_ROOT / "runs/detect/train3/weights/best.pt"
BALL_MODEL_PATH = PROJECT_ROOT / "runs/ball/weights/best.pt"
TEST_IMAGES_DIR = PROJECT_ROOT / "data/tracking/raw/test"
OUTPUT_IMG_DIR = PROJECT_ROOT / "outputs/evaluation/tracking"

BASELINE_CONF = 0.01
BALL_CONF = 0.01
IOU_THRESH = 0.5
N_CLIPS_TO_SAMPLE = 5
MAX_FRAMES_PER_CLIP = None


class InferenceEngine:
    def __init__(self, baseline_path, ball_path, device):
        print(f"Loading models on {device}...")
        self.baseline_model = YOLO(str(baseline_path)).to(device)
        self.ball_model = YOLO(str(ball_path)).to(device)
        self.names = self.baseline_model.names
        self.names_ball = self.ball_model.names

        # Class mapping
        self.name_to_class_id = {v: k for k, v in self.names.items()}
        self.PLAYER_ID = self.name_to_class_id.get("player")
        self.REFEREE_ID = self.name_to_class_id.get("referee")
        self.BALL_ID = 2  # Default fallback

    def infer_baseline(self, img_path):
        result = self.baseline_model(str(img_path), conf=BASELINE_CONF, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    def infer_ball(self, img_array):
        result = self.ball_model(img_array, conf=BALL_CONF, verbose=False)[0]
        det = sv.Detections.from_ultralytics(result)
        if len(det) > 0:
            det.class_id = np.full_like(det.class_id, 0)
        return det

    def infer_ball_sliced(self, img_array):
        h, w, _ = img_array.shape

        def callback(patch):
            results = self.ball_model(patch, conf=BALL_CONF, verbose=False)[0]
            det = sv.Detections.from_ultralytics(results)
            if len(det) > 0:
                det.class_id = np.full_like(det.class_id, 0)
            return det

        slicer = sv.InferenceSlicer(
            callback=callback,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            slice_wh=(w // 2 + 100, h // 2 + 100),
            overlap_wh=(100, 100),
            iou_threshold=0.1,
        )
        return slicer(img_array)


class Evaluator:
    def __init__(self, engine):
        self.engine = engine
        self.stats = {
            "all": {"tp": 0, "fp": 0, "fn": 0},
            self.engine.PLAYER_ID: {"tp": 0, "fp": 0, "fn": 0},
            self.engine.REFEREE_ID: {"tp": 0, "fp": 0, "fn": 0},
        }
        self.pr_data = {
            self.engine.PLAYER_ID: {"conf": [], "is_tp": []},
            self.engine.REFEREE_ID: {"conf": [], "is_tp": []},
        }
        self.confusion_true = []
        self.confusion_pred = []

        self.ball_models = ["baseline", "ball", "ball_sliced"]
        self.ball_stats = {m: {"tp": 0, "fp": 0, "fn": 0} for m in self.ball_models}
        self.ball_pr_data = {m: {"conf": [], "is_tp": []} for m in self.ball_models}

    def process_clips(self, clip_ids, frames_by_clip, gt_by_clip):
        print(f"Processing {len(clip_ids)} clips...")

        for cid in tqdm(clip_ids):
            frames_dict = frames_by_clip[cid]
            gt_dict = gt_by_clip[cid]
            frame_indices = sorted(frames_dict.keys())

            if MAX_FRAMES_PER_CLIP:
                frame_indices = frame_indices[:MAX_FRAMES_PER_CLIP]

            for frame_idx in frame_indices:
                self._process_frame(frames_dict[frame_idx], gt_dict[frame_idx])

    def _process_frame(self, img_path, gt_det):
        # Prepare the ground truth
        gt_mask_pr = np.isin(gt_det.class_id, [self.engine.PLAYER_ID, self.engine.REFEREE_ID])
        gt_det_pr = gt_det[gt_mask_pr]
        gt_mask_ball = (gt_det.class_id == self.engine.BALL_ID)
        gt_det_ball = gt_det[gt_mask_ball]
        gt_boxes_ball = gt_det_ball.xyxy if len(gt_det_ball) > 0 else np.empty((0, 4))

        # 1. Baseline inference
        pred_det_all = self.engine.infer_baseline(img_path)

        # --- Evaluate player and referee ---
        pred_mask_pr = np.isin(pred_det_all.class_id, [self.engine.PLAYER_ID, self.engine.REFEREE_ID])
        pred_det_pr = pred_det_all[pred_mask_pr]

        if len(gt_det_pr) > 0 or len(pred_det_pr) > 0:
            self._update_pr_stats(pred_det_pr, gt_det_pr)
            self._update_confusion_matrix(pred_det_pr, gt_det_pr)

        # --- Evaluate the ball across the 3 models ---
        # A. Baseline ball
        pred_mask_ball = (pred_det_all.class_id == self.engine.BALL_ID)
        pred_det_ball_base = pred_det_all[pred_mask_ball]
        self._update_ball_stats("baseline", pred_det_ball_base, gt_boxes_ball)

        # Load the image for the specialised models
        frame = cv2.imread(str(img_path))
        if frame is None:
            return

        # B. Dedicated ball model
        det_ball = self.engine.infer_ball(frame)
        self._update_ball_stats("ball", det_ball, gt_boxes_ball)

        # C. Dedicated ball model with slicing
        det_ball_sliced = self.engine.infer_ball_sliced(frame)
        self._update_ball_stats("ball_sliced", det_ball_sliced, gt_boxes_ball)

    def _update_pr_stats(self, pred, gt):
        for cls_id in [self.engine.PLAYER_ID, self.engine.REFEREE_ID]:
            gt_boxes = gt.xyxy[gt.class_id == cls_id]
            pred_mask = (pred.class_id == cls_id)
            pred_boxes = pred.xyxy[pred_mask]
            conf = pred.confidence[pred_mask] if len(pred_boxes) > 0 else np.empty((0,))

            tp, fp, fn, is_tp = match_class_boxes(pred_boxes, gt_boxes, IOU_THRESH)

            self.stats[cls_id]["tp"] += tp
            self.stats[cls_id]["fp"] += fp
            self.stats[cls_id]["fn"] += fn
            self.stats["all"]["tp"] += tp
            self.stats["all"]["fp"] += fp
            self.stats["all"]["fn"] += fn

            self.pr_data[cls_id]["conf"].extend(conf.tolist())
            self.pr_data[cls_id]["is_tp"].extend(is_tp.tolist())

    def _update_confusion_matrix(self, pred, gt):
        if len(pred) > 0 and len(gt) > 0:
            matches = match_for_confusion(pred.xyxy, gt.xyxy, IOU_THRESH)
            for p_idx, g_idx in matches:
                self.confusion_true.append(int(gt.class_id[g_idx]))
                self.confusion_pred.append(int(pred.class_id[p_idx]))

    def _update_ball_stats(self, model_name, pred, gt_boxes):
        if len(gt_boxes) > 0 or len(pred) > 0:
            pred_boxes = pred.xyxy
            conf = pred.confidence if len(pred) > 0 else np.empty((0,))
            tp, fp, fn, is_tp = match_class_boxes(pred_boxes, gt_boxes, IOU_THRESH)

            self.ball_stats[model_name]["tp"] += tp
            self.ball_stats[model_name]["fp"] += fp
            self.ball_stats[model_name]["fn"] += fn
            self.ball_pr_data[model_name]["conf"].extend(conf.tolist())
            self.ball_pr_data[model_name]["is_tp"].extend(is_tp.tolist())

    def print_summary(self):
        print("\n--- Summary ---")
        for name, data in [
            ("Global (player + referee)", self.stats["all"]),
            ("Player", self.stats[self.engine.PLAYER_ID]),
            ("Referee", self.stats[self.engine.REFEREE_ID])
        ]:
            p, r, f1 = calculate_f1(data["tp"], data["fp"], data["fn"])
            print(f"{name}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}")

        for m in self.ball_models:
            data = self.ball_stats[m]
            p, r, f1 = calculate_f1(data["tp"], data["fp"], data["fn"])
            print(f"Ball ({m}): P={p:.3f}, R={r:.3f}, F1={f1:.3f}")


class Plotter:
    def __init__(self, output_dir, class_names):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names

    def save_pr_curve(self, series, filename, title):
        plt.figure(figsize=(10, 6))
        for rec, prec, label in series:
            ap = np.trapz(prec, rec) if rec.size > 1 else 0.0
            plt.plot(rec, prec, label=f"{label} AP:{ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.savefig(self.output_dir / filename)
        plt.close()
        print(f"Saved {filename}")

    def save_confusion_matrix(self, y_true, y_pred, labels, filename):
        cm = np.zeros((len(labels), len(labels)), dtype=float)
        idx_map = {cid: i for i, cid in enumerate(labels)}

        for t, p in zip(y_true, y_pred):
            if t in idx_map and p in idx_map:
                cm[idx_map[t], idx_map[p]] += 1.0

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        label_names = [self.class_names[c] for c in labels]

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Normalized Confusion Matrix")

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
        print(f"Saved {filename}")


def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = InferenceEngine(BASELINE_MODEL_PATH, BALL_MODEL_PATH, device)
    evaluator = Evaluator(engine)

    # Discover the available clips
    clip_dir_pat = re.compile(r"SNMOT-(\d+)$")
    clip_to_frames = {}
    clip_id_to_dir = {}

    for d in TEST_IMAGES_DIR.iterdir():
        m = clip_dir_pat.match(d.name)
        if d.is_dir() and m and (d / "img1").is_dir():
            clip_id = int(m.group(1))
            clip_to_frames[clip_id] = len(list((d / "img1").glob("*.jpg")))
            clip_id_to_dir[clip_id] = d

    all_clips = sorted(clip_to_frames.keys())
    sampled_clips = random.sample(all_clips, k=min(N_CLIPS_TO_SAMPLE, len(all_clips)))

    # Load the data
    frames_by_clip, gt_by_clip = {}, {}
    print("Loading ground-truth data...")
    for cid in sampled_clips:
        f_idx, gt_idx, _ = load_clip_frames_and_gt(cid, clip_id_to_dir, engine.name_to_class_id)
        frames_by_clip[cid] = f_idx
        gt_by_clip[cid] = gt_idx

    # Run the evaluation
    evaluator.process_clips(sampled_clips, frames_by_clip, gt_by_clip)
    evaluator.print_summary()

    # Plots
    plotter = Plotter(OUTPUT_IMG_DIR, engine.names)

    # 1. Player and referee curves
    pr_series = []
    for cid in [engine.PLAYER_ID, engine.REFEREE_ID]:
        total = evaluator.stats[cid]["tp"] + evaluator.stats[cid]["fn"]
        r, p, _ = build_pr_curve_monotone(evaluator.pr_data[cid]["conf"], evaluator.pr_data[cid]["is_tp"], total)
        pr_series.append((r, p, engine.names[cid]))

    # Combined
    total_all = evaluator.stats["all"]["tp"] + evaluator.stats["all"]["fn"]
    all_conf = evaluator.pr_data[engine.PLAYER_ID]["conf"] + evaluator.pr_data[engine.REFEREE_ID]["conf"]
    all_tp = evaluator.pr_data[engine.PLAYER_ID]["is_tp"] + evaluator.pr_data[engine.REFEREE_ID]["is_tp"]
    r_all, p_all, _ = build_pr_curve_monotone(all_conf, all_tp, total_all)
    pr_series.append((r_all, p_all, "Global"))

    plotter.save_pr_curve(pr_series, "pr_curve_players.png", "PR Curve (Player + Referee)")

    # 2. Confusion matrix
    plotter.save_confusion_matrix(
        evaluator.confusion_true,
        evaluator.confusion_pred,
        [engine.PLAYER_ID, engine.REFEREE_ID],
        "confusion_matrix.png"
    )

    # 3. Ball curves
    ball_series = []
    for m in evaluator.ball_models:
        total = evaluator.ball_stats[m]["tp"] + evaluator.ball_stats[m]["fn"]
        r, p, _ = build_pr_curve_monotone(evaluator.ball_pr_data[m]["conf"], evaluator.ball_pr_data[m]["is_tp"], total)
        ball_series.append((r, p, m))

    plotter.save_pr_curve(ball_series, "pr_curve_ball.png", "PR Curve (Ball Models)")


if __name__ == "__main__":
    main()
