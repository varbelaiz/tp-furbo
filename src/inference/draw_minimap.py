import cv2
import numpy as np
import supervision as sv
from typing import List, Optional, Tuple, Dict
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

# Colour configuration
TEAM_COLORS: Dict[int, sv.Color] = {
    0: sv.Color(231, 76, 60),    # Red (team 0)
    1: sv.Color(52, 152, 219),   # Blue (team 1)
    -1: sv.Color(149, 165, 166)  # Grey (unknown / referee)
}

COLOR_WHITE = sv.Color.WHITE
COLOR_BLACK = sv.Color.BLACK
COLOR_YELLOW = sv.Color(255, 255, 0)
COLOR_GREEN_PITCH = sv.Color(34, 139, 34)

# Drawing dimensions for the minimap
PLAYER_RADIUS = 15
BALL_RADIUS = 8
PITCH_SCALE = 0.1
PITCH_PADDING = 20


def draw_ball_trajectory(
    img: np.ndarray,
    trajectory: List[Optional[np.ndarray]],
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw the ball trail on the minimap.
    `None` entries in the trajectory break the line so discontinuous segments
    are not joined together.
    """
    if not trajectory or len(trajectory) < 2:
        return img

    h, w = img.shape[:2]

    # Walk the trajectory in consecutive pairs
    for i in range(len(trajectory) - 1):
        pt1 = trajectory[i]
        pt2 = trajectory[i + 1]

        # A gap in the sequence cuts the line here
        if pt1 is None or pt2 is None:
            continue

        # Convert pitch centimetres into minimap pixels
        cx1 = int(pt1[0] * PITCH_SCALE) + PITCH_PADDING
        cy1 = int(pt1[1] * PITCH_SCALE) + PITCH_PADDING

        cx2 = int(pt2[0] * PITCH_SCALE) + PITCH_PADDING
        cy2 = int(pt2[1] * PITCH_SCALE) + PITCH_PADDING

        # Only draw when both endpoints fall inside the image
        if (0 <= cx1 < w and 0 <= cy1 < h and 0 <= cx2 < w and 0 <= cy2 < h):
            cv2.line(img, (cx1, cy1), (cx2, cy2), color, thickness)

    return img


def draw_categorized_pitch_view(
    config: SoccerPitchConfiguration,
    team0_xy: np.ndarray,
    team1_xy: np.ndarray,
    unknown_xy: np.ndarray,
    ball_xy: Optional[np.ndarray] = None,
    ball_trail: Optional[List[Optional[np.ndarray]]] = None
) -> np.ndarray:
    """
    Render the full minimap (radar) with players, ball and ball trail.
    """
    # 1. Draw the pitch background
    pitch_image = draw_pitch(
        config=config,
        padding=PITCH_PADDING,
        scale=PITCH_SCALE,
        line_thickness=4,
        line_color=COLOR_WHITE,
        background_color=COLOR_GREEN_PITCH
    )

    # 2. Draw the historical trail (underneath the players)
    if ball_trail is not None:
        pitch_image = draw_ball_trajectory(
            pitch_image,
            ball_trail,
            color=(255, 255, 255),
            thickness=2
        )

    # Small local helper to avoid repetition
    def _draw_group(points, color):
        if points is not None and len(points) > 0:
            return draw_points_on_pitch(
                config=config,
                xy=points,
                face_color=color,
                edge_color=COLOR_WHITE,
                radius=PLAYER_RADIUS,
                padding=PITCH_PADDING,
                scale=PITCH_SCALE,
                pitch=pitch_image
            )
        return pitch_image

    # 3. Draw the players, grouped by team
    pitch_image = _draw_group(unknown_xy, TEAM_COLORS[-1])
    pitch_image = _draw_group(team0_xy, TEAM_COLORS[0])
    pitch_image = _draw_group(team1_xy, TEAM_COLORS[1])

    # 4. Draw the current ball position (highlighted marker)
    if ball_xy is not None and len(ball_xy) > 0:
        pitch_image = draw_points_on_pitch(
            config=config,
            xy=ball_xy,
            face_color=COLOR_YELLOW,
            edge_color=COLOR_BLACK,
            radius=BALL_RADIUS,
            padding=PITCH_PADDING,
            scale=PITCH_SCALE,
            pitch=pitch_image
        )

    return pitch_image


def overlay_radar(
    frame: np.ndarray,
    radar_img: np.ndarray,
    width_percentage: float = 0.3,
    margin: int = 20
) -> np.ndarray:
    """
    Composite the radar image onto the original frame (top-right corner).
    """
    video_h, video_w = frame.shape[:2]
    radar_h, radar_w = radar_img.shape[:2]

    new_w = int(video_w * width_percentage)
    new_h = int(radar_h * (new_w / radar_w))

    radar_resized = cv2.resize(radar_img, (new_w, new_h))

    y_start = margin
    y_end = margin + new_h
    x_start = video_w - margin - new_w
    x_end = video_w - margin

    frame[y_start:y_end, x_start:x_end] = radar_resized

    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)

    return frame


def draw_pitch_overlay(frame: np.ndarray, homography_matrix: np.ndarray) -> np.ndarray:
    """
    Project the virtual pitch lines onto the video frame using the inverse
    homography. Useful as a visual check on calibration quality.
    """
    if homography_matrix is None:
        return frame

    try:
        # The inverse maps world (pitch) coordinates back to image pixels
        inv_H = np.linalg.inv(homography_matrix)
    except np.linalg.LinAlgError:
        return frame

    L, W = 10500.0, 6800.0

    polys = [
        [[0, 0], [L, 0], [L, W], [0, W]],
        [[L / 2, 0], [L / 2, W]],
        [[0, (W - 4032) / 2], [1650, (W - 4032) / 2], [1650, (W + 4032) / 2], [0, (W + 4032) / 2]],
        [[L, (W - 4032) / 2], [L - 1650, (W - 4032) / 2], [L - 1650, (W + 4032) / 2], [L, (W + 4032) / 2]]
    ]

    overlay = frame.copy()

    for p in polys:
        pts = np.array(p, dtype=np.float32).reshape(-1, 1, 2)
        # Project the world points onto the screen
        projected = cv2.perspectiveTransform(pts, inv_H)
        # Draw the polylines
        cv2.polylines(overlay, [projected.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

    # Blend with transparency
    return cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)


def draw_kps_ids(frame: np.ndarray, keypoints: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Draw the detected keypoints along with their indices, for debugging.
    """
    if keypoints is None:
        return frame

    kps = keypoints[0] if len(keypoints.shape) == 3 else keypoints

    for i, (x, y, conf) in enumerate(kps):
        if conf < threshold:
            continue

        # Draw the point
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 1)

        # Draw the index and confidence
        text = f"{i} ({conf:.2f})"
        cv2.putText(frame, text, (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return frame
