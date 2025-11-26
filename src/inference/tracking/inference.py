from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel
import umap
from ultralytics import YOLO
from collections import defaultdict

BASE = Path(__file__).resolve().parents[3]

BASELINE_RUN_NAME = "train3"

BASELINE_MODEL_PATH = (
    BASE / "runs" / "detect" / BASELINE_RUN_NAME / "weights" / "best.pt"
)
BALL_MODEL_PATH = BASE / "runs" / "ball" / "weights" / "best.pt"

BASELINE_BALL_CLASS_ID = 2
BALL_CANONICAL_CLASS_ID = 2
DINO_MODEL_ID = "facebook/dinov2-base"

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


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


class TeamEmbeddingSegmenter:
    """
    Team segmentation using DINO + UMAP + KMeans, keyed on track IDs.

    Logic:
    1. Collect embeddings for each 'track_id' across the video.
    2. Compute the 'centroid' (mean) embedding for each track.
    3. Cluster the centroids to assign a team to each track.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        n_neighbors: int = 20,
        umap_min_dist: float = 0.0,
        n_components: int = 3,
        n_clusters: int = 2,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_neighbors = n_neighbors
        self.umap_min_dist = umap_min_dist
        self.n_components = n_components
        self.n_clusters = n_clusters

        self._processor = AutoImageProcessor.from_pretrained(
            DINO_MODEL_ID, use_fast=True
        )
        self._dino = AutoModel.from_pretrained(DINO_MODEL_ID).to(self.device).eval()

        # Stores the list of embeddings collected for each track_id.
        # Structure: { track_id: [tensor_emb1, tensor_emb2, ...] }
        self._track_embeddings: Dict[int, List[torch.Tensor]] = defaultdict(list)

        # Result mapping: { track_id: team_id }
        self._track_to_team: Dict[int, int] = {}
        self._fitted: bool = False

        # Populated by fit(), consumed by visualize_clusters()
        self.X_reduced: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None

    def collect_from_frame(
        self,
        frame_bgr: np.ndarray,
        detections: sv.Detections,
    ) -> None:
        """
        Collect embeddings for the detections that carry a tracker_id.
        Expects detections that contain ONLY players and that have a tracker_id.
        """
        if len(detections) == 0:
            return

        # Nothing to key the embeddings on without tracker IDs
        if detections.tracker_id is None:
            return

        h, w = frame_bgr.shape[:2]
        crops_pil: List[Image.Image] = []
        valid_indices: List[int] = []

        # Extract the crops
        for i, (xyxy, tracker_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop_bgr = frame_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue

            crops_pil.append(_bgr_to_pil(crop_bgr))
            valid_indices.append(i)

        if not crops_pil:
            return

        # DINO inference
        with torch.no_grad():
            inputs = self._processor(images=crops_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._dino(**inputs)
            # Take the CLS token
            emb_batch = outputs.last_hidden_state[:, 0, :]
            emb_batch = F.normalize(emb_batch, dim=1).cpu()

        # Store the embeddings grouped by track_id
        for i, idx_in_det in enumerate(valid_indices):
            track_id = int(detections.tracker_id[idx_in_det])
            self._track_embeddings[track_id].append(emb_batch[i])

    def fit(self) -> None:
        """
        Compute the mean embedding per track, run UMAP + KMeans, and map each
        track to a team.
        """
        if not self._track_embeddings:
            self._fitted = True
            return

        print(f"Segmenter: Processing {len(self._track_embeddings)} unique tracks...")

        # 1. Calculate the centroid (mean) embedding of each track
        track_ids = []
        mean_embeddings = []

        for tid, embs in self._track_embeddings.items():
            stack = torch.stack(embs)
            mean_emb = torch.mean(stack, dim=0)
            mean_emb = F.normalize(mean_emb.unsqueeze(0), dim=1).squeeze(0)

            track_ids.append(tid)
            mean_embeddings.append(mean_emb.numpy())

        X = np.array(mean_embeddings, dtype=np.float32)

        # 2. UMAP reduction
        reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.umap_min_dist,
            metric="cosine",
            random_state=42,
        )

        # Kept on the instance so visualize_clusters() can reuse it
        self.X_reduced = reducer.fit_transform(X)

        # 3. KMeans clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init="auto",
        )
        self.labels = kmeans.fit_predict(self.X_reduced)

        # 4. Build the track -> team map
        self._track_to_team = {
            tid: int(label) for tid, label in zip(track_ids, self.labels)
        }
        self._fitted = True
        print("Segmenter: Team fitting complete.")

    def visualize_clusters(self, output_path: str = "outputs/team_clusters.png"):
        """
        Plot the 2D PCA projection of the UMAP-reduced embeddings.
        """
        if self.X_reduced is None or self.labels is None:
            print("No data to plot (run fit() first).")
            return

        print("Generating cluster plot...")

        # PCA brings the 3 UMAP components down to 2 for plotting
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(self.X_reduced)

        plt.figure(figsize=(10, 7))

        unique_labels = np.unique(self.labels)
        colors = ['#FF0000', '#0000FF', '#FFFF00']  # Red, blue, yellow (if a third cluster appears)

        for k in unique_labels:
            mask = (self.labels == k)
            c = colors[k] if k < len(colors) else None
            plt.scatter(
                X2[mask, 0],
                X2[mask, 1],
                c=c,
                s=20,
                alpha=0.75,
                label=f"Team {k} (n={mask.sum()})"
            )

        plt.title(f"UMAP ({self.n_components}D) -> PCA (2D) coloured by KMeans")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.show()

    def get_team_id(self, track_id: int) -> int:
        """Return the team_id for a given track_id, or -1 when unknown."""
        if not self._fitted:
            return -1
        return self._track_to_team.get(track_id, -1)
