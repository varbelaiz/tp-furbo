import sys
import re
import random
import numpy as np
import cv2
import supervision as sv
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, deque
from typing import List, Callable, Generator, Tuple

# Add the current directory to the path so local packages resolve
sys.path.append(".")

from src.inference.keypoints.homography import HomographyTransformer
from src.inference.keypoints.detect_keypoints import load_keypoint_model, get_keypoint_detections
from src.inference.tracking.inference import (
    infer_frame,
    BALL_CANONICAL_CLASS_ID,
    TeamEmbeddingSegmenter,
)
from src.inference.draw_minimap import (
    overlay_radar,
    draw_categorized_pitch_view,
    draw_pitch_overlay,
    draw_kps_ids,
)
from sports.configs.soccer import SoccerPitchConfiguration

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================

# Options: "video", "images" (sequence), "single_image" (one photo)
SOURCE_MODE = "single_image"

VIDEO_PATH = "Union Berlin vs Bayern Munich 2-2 - 1080.mp4"
IMAGES_DIR = "data/tracking/YOLO_baseline/images/val"
SINGLE_IMAGE_PATH = "data/keypoints/imgs/00003.jpg"
IMAGES_FPS = 25

OUTPUT_FILENAME = "outputs/pipeline_output"  # .mp4 or .jpg is appended automatically

USE_57_POINTS = False
CONFIDENCE_THRESHOLD = 0.85

MODEL_29_PATH = "runs/keypoints/29_kp_extended_parallel2/weights/best.pt"
MODEL_57_PATH = "runs/keypoints/57_kp_v4_extended_plus1003/weights/best.pt"
KP_PATH = MODEL_57_PATH if USE_57_POINTS else MODEL_29_PATH

CLASS_NAMES = {0: "player", 1: "referee"}
PLAYER_CLASS_ID = 0
REFEREE_CLASS_ID = 1
ID_TEAM_0 = 0
ID_TEAM_1 = 1
ID_REFEREE = 2
ID_BALL = 3


# ==========================================
# CONFIGURATION HELPERS
# ==========================================

def get_points_from_detections(detections: sv.Detections) -> np.ndarray:
    """Extract the bottom-center coordinates of the detections."""
    if len(detections) == 0:
        return np.empty((0, 2))
    return detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)


def setup_homography_engine(use_57_points: bool, video_info: sv.VideoInfo) -> HomographyTransformer:
    """Initialise the perspective transformer."""
    mode_str = "57" if use_57_points else "29"
    print(f"Initialising homography (mode: {mode_str}, res: {video_info.width}x{video_info.height})...")

    return HomographyTransformer(
        mode=mode_str,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )


def setup_image_sequence(base_dir: str) -> Tuple[List[Path], sv.VideoInfo]:
    """
    Scan a directory for SoccerNet-style clips (SNMOT-X_Y.jpg), pick a random
    clip and return its frame paths in order.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    pat = re.compile(r"SNMOT-(\d+)_(\d+)\.jpg$")
    clip_to_frames = defaultdict(int)

    print(f"Scanning directory: {base_dir} ...")
    all_files = list(base_path.glob("SNMOT-*_*.jpg"))

    for p in all_files:
        m = pat.match(p.name)
        if m:
            clip_id = int(m.group(1))
            clip_to_frames[clip_id] += 1

    clip_ids = sorted(clip_to_frames.keys())
    if not clip_ids:
        raise FileNotFoundError(f"No SNMOT images found in {base_dir}")

    # Pick one clip at random
    selected_clip_id = random.choice(clip_ids)
    print(f"Selected clip: ID={selected_clip_id} ({clip_to_frames[selected_clip_id]} frames)")

    frame_paths = []
    for p in base_path.glob(f"SNMOT-{selected_clip_id}_*.jpg"):
        m = pat.match(p.name)
        if m:
            frame_no = int(m.group(2))
            frame_paths.append((frame_no, p))

    frame_paths.sort(key=lambda x: x[0])
    sorted_paths = [p for _, p in frame_paths]

    if not sorted_paths:
        raise ValueError("The selected clip has no valid frames.")

    first_img = cv2.imread(str(sorted_paths[0]))
    if first_img is None:
        raise ValueError(f"Could not read: {sorted_paths[0]}")

    height, width = first_img.shape[:2]
    video_info = sv.VideoInfo(
        width=width,
        height=height,
        fps=IMAGES_FPS,
        total_frames=len(sorted_paths),
    )

    return sorted_paths, video_info


def setup_single_image(image_path: str) -> Tuple[List[Path], sv.VideoInfo]:
    """Set up the environment to process a single image as a one-frame video."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read the image: {image_path}")

    height, width = img.shape[:2]

    # Synthetic single-frame VideoInfo
    video_info = sv.VideoInfo(
        width=width,
        height=height,
        fps=1,
        total_frames=1,
    )

    return [path], video_info


# ==========================================
# PASS 1: TRACKING AND EMBEDDINGS
# ==========================================

def run_tracking_pass(
    frames_callback: Callable[[], Generator[np.ndarray, None, None]],
    total_frames: int,
    player_tracker: sv.ByteTrack,
    segmenter: TeamEmbeddingSegmenter,
) -> List[sv.Detections]:
    """
    First pass: run detection and tracking, and collect colour embeddings used
    to classify teams.
    """
    all_tracked_detections = []
    print("\nPass 1/2: tracking and embedding collection...")

    frame_iterator = frames_callback()

    for frame in tqdm(frame_iterator, total=total_frames, desc="Tracking"):
        # 1. YOLO inference (detection)
        detections = infer_frame(frame)

        # 2. Split the ball off from the players
        is_ball = detections.class_id == BALL_CANONICAL_CLASS_ID
        ball_det = detections[is_ball]
        non_ball_det = detections[~is_ball]

        # 3. Tracking (players and referees only)
        non_ball_det = player_tracker.update_with_detections(non_ball_det)

        # 4. Give the ball a placeholder ID so the two sets can be merged
        if len(ball_det) > 0:
            ball_det.tracker_id = np.array([-1] * len(ball_det))

        # 5. Embeddings for team clustering
        if len(non_ball_det) > 0:
            is_player = non_ball_det.class_id == PLAYER_CLASS_ID
            players_for_embed = non_ball_det[is_player]
            if len(players_for_embed) > 0:
                segmenter.collect_from_frame(
                    frame_bgr=frame,
                    detections=players_for_embed,
                )

        # 6. Merge and store.
        # 'data' is cleared to avoid conflicts during the merge.
        non_ball_det.data = {}
        ball_det.data = {}
        merged = sv.Detections.merge([non_ball_det, ball_det])
        all_tracked_detections.append(merged)

    return all_tracked_detections


# ==========================================
# PASS 2: RENDERING AND OUTPUT
# ==========================================

def run_rendering_pass(
    frames_callback: Callable[[], Generator[np.ndarray, None, None]],
    video_info: sv.VideoInfo,
    output_base_name: str,
    all_tracked_detections: List[sv.Detections],
    segmenter: TeamEmbeddingSegmenter,
    kp_model,
    homography_engine,
    pitch_config: SoccerPitchConfiguration,
):
    """
    Second pass: compute the homography, project onto the radar and draw the
    final output.
    """
    # Decide the output type
    is_video_output = video_info.total_frames > 1
    output_path = f"{output_base_name}.mp4" if is_video_output else f"{output_base_name}.jpg"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    video_writer = None
    if is_video_output:
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_info.fps,
            (video_info.width, video_info.height),
        )

    palette = sv.ColorPalette.from_hex(["#FF4B4B", "#6B8EFF", "#FFE66D", "#d282f5"])
    color_annotator = sv.ColorAnnotator(color=palette, opacity=0.5)
    label_annotator = sv.LabelAnnotator(color=palette, text_scale=0.5, text_padding=5)

    # History used to draw the ball trail on the radar
    buffer_size = int(video_info.fps * 5)  # 5 seconds of history
    ball_history = deque(maxlen=buffer_size)

    print(f"\nPass 2/2: rendering ({'VIDEO' if is_video_output else 'IMAGE'})...")

    frame_iterator = frames_callback()

    for frame_idx, frame in enumerate(tqdm(frame_iterator, total=video_info.total_frames, desc="Render")):
        detections = all_tracked_detections[frame_idx]

        # --- Team assignment ---
        visual_ids = np.zeros(len(detections), dtype=int)
        real_team_ids = np.full(len(detections), -1, dtype=int)

        if detections.tracker_id is not None:
            for i, (class_id, track_id) in enumerate(zip(detections.class_id, detections.tracker_id)):
                cid = int(class_id)
                if cid == BALL_CANONICAL_CLASS_ID:
                    visual_ids[i] = ID_BALL
                elif cid == REFEREE_CLASS_ID:
                    visual_ids[i] = ID_REFEREE
                elif cid == PLAYER_CLASS_ID:
                    # Use the segmenter fitted during pass 1
                    t_id = segmenter.get_team_id(int(track_id))
                    if t_id == 0:
                        visual_ids[i] = ID_TEAM_0
                        real_team_ids[i] = 0
                    elif t_id == 1:
                        visual_ids[i] = ID_TEAM_1
                        real_team_ids[i] = 1
                    else:
                        visual_ids[i] = ID_REFEREE  # Fall back to referee when the team is unclear
                else:
                    visual_ids[i] = ID_REFEREE

        detections.data["team_id"] = real_team_ids
        detections.class_id = visual_ids

        # --- Keypoints and homography ---
        _, kps_data = get_keypoint_detections(kp_model, frame)
        view_transformer = None
        if kps_data is not None:
            view_transformer = homography_engine.transform_to_pitch_keypoints(kps_data)

        # --- Radar / minimap ---
        tactical_map = None

        # Boolean filters used to split the detections apart
        ball_mask = detections.class_id == ID_BALL

        if view_transformer is not None:
            team0_mask = detections.data["team_id"] == 0
            team1_mask = detections.data["team_id"] == 1
            unknown_mask = ~(team0_mask | team1_mask | ball_mask)

            # Project the points
            points_ball = view_transformer.transform_points(get_points_from_detections(detections[ball_mask]))
            points_team0 = view_transformer.transform_points(get_points_from_detections(detections[team0_mask]))
            points_team1 = view_transformer.transform_points(get_points_from_detections(detections[team1_mask]))
            points_unknown = view_transformer.transform_points(get_points_from_detections(detections[unknown_mask]))

            # Update the ball history
            if len(points_ball) > 0:
                ball_history.append(points_ball[0])
            else:
                ball_history.append(None)

            # Draw the radar
            tactical_map = draw_categorized_pitch_view(
                config=pitch_config,
                team0_xy=points_team0,
                team1_xy=points_team1,
                unknown_xy=points_unknown,
                ball_xy=points_ball,
                ball_trail=list(ball_history)
            )
        else:
            # No homography, so the radar stays empty for this frame
            ball_history.append(None)
            tactical_map = draw_categorized_pitch_view(
                config=pitch_config,
                team0_xy=np.array([]), team1_xy=np.array([]),
                unknown_xy=np.array([]), ball_xy=np.array([]),
                ball_trail=None
            )

        # --- Draw on the frame ---
        labels = []
        for cid, tid in zip(detections.class_id, detections.tracker_id):
            if cid == ID_BALL:
                labels.append("")
            elif cid == ID_REFEREE:
                labels.append("REF")
            elif cid == ID_TEAM_0:
                labels.append(f"#{tid} T0")
            elif cid == ID_TEAM_1:
                labels.append(f"#{tid} T1")
            else:
                labels.append(f"#{tid}")

        frame = color_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Overlays
        if view_transformer is not None:
            frame = draw_pitch_overlay(frame, view_transformer.homography)

        if kps_data is not None:
            frame = draw_kps_ids(frame, kps_data, CONFIDENCE_THRESHOLD)

        frame = overlay_radar(frame, tactical_map)

        # --- Save ---
        if is_video_output:
            video_writer.write(frame)
        else:
            cv2.imwrite(output_path, frame)

    if video_writer:
        video_writer.release()

    print(f"Done. Output written to: {output_path}")


# ==========================================
# MAIN
# ==========================================

def main():
    print(f"Starting pipeline | mode: {SOURCE_MODE.upper()}")

    # 1. Data source setup
    video_info = None
    frame_generator_func = None

    if SOURCE_MODE == "video":
        video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
        frame_generator_func = lambda: sv.get_video_frames_generator(VIDEO_PATH)
        print(f"Source: video ({VIDEO_PATH})")

    elif SOURCE_MODE == "images":
        image_paths, video_info = setup_image_sequence(IMAGES_DIR)

        def images_generator():
            for p in image_paths:
                img = cv2.imread(str(p))
                if img is not None:
                    yield img

        frame_generator_func = images_generator
        print(f"Source: image sequence ({len(image_paths)} frames)")

    elif SOURCE_MODE == "single_image":
        image_paths, video_info = setup_single_image(SINGLE_IMAGE_PATH)

        def single_image_generator():
            img = cv2.imread(str(image_paths[0]))
            if img is not None:
                yield img

        frame_generator_func = single_image_generator
        print(f"Source: single image ({SINGLE_IMAGE_PATH})")

    else:
        raise ValueError(f"Unknown mode: {SOURCE_MODE}")

    # 2. Load models and helpers
    kp_model = load_keypoint_model(KP_PATH)
    pitch_config = SoccerPitchConfiguration()
    homography_engine = setup_homography_engine(USE_57_POINTS, video_info)

    player_tracker = sv.ByteTrack(frame_rate=video_info.fps)
    segmenter = TeamEmbeddingSegmenter()

    # 3. Run the pipeline

    # --- Step 1: detection and tracking ---
    all_tracked_detections = run_tracking_pass(
        frames_callback=frame_generator_func,
        total_frames=video_info.total_frames,
        player_tracker=player_tracker,
        segmenter=segmenter,
    )

    print("\nComputing team clusters...")
    segmenter.fit()
    segmenter.visualize_clusters()  # Optional: renders the clustering plot

    # --- Step 2: final rendering ---
    run_rendering_pass(
        frames_callback=frame_generator_func,
        video_info=video_info,
        output_base_name=OUTPUT_FILENAME,
        all_tracked_detections=all_tracked_detections,
        segmenter=segmenter,
        kp_model=kp_model,
        homography_engine=homography_engine,
        pitch_config=pitch_config,
    )


if __name__ == "__main__":
    main()
