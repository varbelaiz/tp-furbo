import numpy as np
import supervision as sv
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path

def parse_gameinfo(gameinfo_path: Path) -> dict[int, dict]:
    cfg = ConfigParser()
    cfg.read(gameinfo_path)
    seq = cfg["Sequence"]
    tracklet_meta: dict[int, dict] = {}

    for key, value in seq.items():
        if not key.startswith("trackletid_"):
            continue

        idx = int(key.split("_")[1])
        raw = value.strip()
        cat_str, _, extra = raw.partition(";")
        cat_str = cat_str.strip().lower()

        role = "unknown"
        if "ball" in cat_str: role = "ball"
        elif "referee" in cat_str: role = "referee"
        elif "player" in cat_str or "goalkeeper" in cat_str: role = "player"

        team_side = None
        if "team left" in cat_str: team_side = "left"
        elif "team right" in cat_str: team_side = "right"

        tracklet_meta[idx] = {
            "role": role,
            "team_side": team_side,
            "raw": raw,
            "extra": extra.strip() if extra else None,
        }

    sides = sorted({m["team_side"] for m in tracklet_meta.values() if m["team_side"] is not None})
    side_to_ab = {side: f"team_{chr(ord('A') + i)}" for i, side in enumerate(sides[:2])}

    for m in tracklet_meta.values():
        m["team_ab"] = side_to_ab.get(m["team_side"])

    return tracklet_meta

def load_clip_frames_and_gt(clip_id: int, clip_id_to_dir: dict[int, Path], name_to_class_id: dict[str, int]) -> tuple[dict[int, Path], dict[int, sv.Detections], dict[int, dict]]:
    clip_dir = clip_id_to_dir[clip_id]
    img_dir = clip_dir / "img1"
    gt_path = clip_dir / "gt" / "gt.txt"
    gameinfo_path = clip_dir / "gameinfo.ini"

    tracklet_meta = parse_gameinfo(gameinfo_path)
    gt_raw = defaultdict(list)
    
    with gt_path.open("r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            parts = line.split(",")
            if len(parts) < 6: continue
            frame_idx, track_id, x, y, w, h = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            gt_raw[frame_idx].append((track_id, x, y, w, h))

    frames_by_idx = {}
    gt_by_idx = {}

    for img_path in sorted(img_dir.glob("*.jpg")):
        frame_idx = int(img_path.stem)
        frames_by_idx[frame_idx] = img_path
        objs = gt_raw.get(frame_idx, [])

        if not objs:
            gt_by_idx[frame_idx] = sv.Detections.empty()
            continue

        xyxy, class_ids, track_ids, roles, team_sides, team_ab = [], [], [], [], [], []

        for track_id, x, y, w, h in objs:
            meta = tracklet_meta.get(track_id)
            if not meta or meta["role"] == "unknown": continue

            cls_name = meta["role"] if meta["role"] in ["ball", "referee"] else "player"
            
            xyxy.append([x, y, x + w, y + h])
            class_ids.append(name_to_class_id[cls_name])
            track_ids.append(track_id)
            roles.append(meta["role"])
            team_sides.append(meta["team_side"])
            team_ab.append(meta["team_ab"])

        det = sv.Detections(
            xyxy=np.array(xyxy, dtype=float),
            class_id=np.array(class_ids, dtype=int),
        )
        det.data["track_id"] = np.array(track_ids, dtype=int)
        det.data["role"] = np.array(roles, dtype=object)
        det.data["team_side"] = np.array(team_sides, dtype=object)
        det.data["team_ab"] = np.array(team_ab, dtype=object)
        gt_by_idx[frame_idx] = det

    return frames_by_idx, gt_by_idx, tracklet_meta

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area <= 0: return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    
    return inter_area / union if union > 0 else 0.0

def match_class_boxes(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thresh: float = 0.5) -> tuple[int, int, int, np.ndarray]:
    num_pred, num_gt = len(pred_boxes), len(gt_boxes)
    if num_pred == 0 and num_gt == 0:
        return 0, 0, 0, np.zeros((0,), dtype=bool)

    matched_gt = set()
    tp, fp = 0, 0
    is_tp = np.zeros((num_pred,), dtype=bool)

    for i in range(num_pred):
        best_iou, best_j = 0.0, -1
        for j in range(num_gt):
            if j in matched_gt: continue
            iou = compute_iou(pred_boxes[i], gt_boxes[j])
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_iou >= iou_thresh and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)
            is_tp[i] = True
        else:
            fp += 1

    fn = num_gt - len(matched_gt)
    return tp, fp, fn, is_tp

def match_for_confusion(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thresh: float = 0.5):
    num_pred, num_gt = len(pred_boxes), len(gt_boxes)
    if num_pred == 0 or num_gt == 0: return []

    iou_mat = np.zeros((num_pred, num_gt))
    for i in range(num_pred):
        for j in range(num_gt):
            iou_mat[i, j] = compute_iou(pred_boxes[i], gt_boxes[j])

    used_pred, used_gt, matches = set(), set(), []
    
    while True:
        best_iou, best_i, best_j = 0.0, -1, -1
        for i in range(num_pred):
            if i in used_pred: continue
            for j in range(num_gt):
                if j in used_gt: continue
                if iou_mat[i, j] > best_iou:
                    best_iou, best_i, best_j = iou_mat[i, j], i, j
        
        if best_iou < iou_thresh or best_i < 0: break
        matches.append((best_i, best_j))
        used_pred.add(best_i)
        used_gt.add(best_j)

    return matches

def build_pr_curve_monotone(conf_list, is_tp_list, total_gt):
    conf = np.asarray(conf_list, dtype=float)
    is_tp = np.asarray(is_tp_list, dtype=bool)

    if total_gt <= 0 or conf.size == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([0.0, 0.0])

    order = np.argsort(-conf)
    conf_sorted = conf[order]
    tp_sorted = is_tp[order].astype(int)

    tps_cum = np.cumsum(tp_sorted)
    denom = tps_cum + np.cumsum(1 - tp_sorted)
    precision = np.where(denom > 0, tps_cum / denom, 0.0)
    recall = tps_cum / float(total_gt)

    precision_envelope = precision.copy()
    for i in range(precision_envelope.size - 2, -1, -1):
        if precision_envelope[i] < precision_envelope[i + 1]:
            precision_envelope[i] = precision_envelope[i + 1]

    return recall, precision_envelope, conf_sorted

def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1