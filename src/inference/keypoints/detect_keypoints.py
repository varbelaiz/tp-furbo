"""Core keypoint detection functions for soccer analysis.

This module provides the core functionality for detecting soccer field keypoints
using YOLO pose estimation models and geometric calculations.

Adapted from https://github.com/Adit-jain/Soccer_Analysis
"""

import numpy as np
from typing import Tuple, List, Dict

from ultralytics import YOLO
import supervision as sv

# Soccer field keypoint names (29 keypoints)
KEYPOINT_NAMES_29 = {
    0: "sideline_top_left",
    1: "big_rect_left_top_pt1",
    2: "big_rect_left_top_pt2",
    3: "big_rect_left_bottom_pt1",
    4: "big_rect_left_bottom_pt2",
    5: "small_rect_left_top_pt1",
    6: "small_rect_left_top_pt2",
    7: "small_rect_left_bottom_pt1",
    8: "small_rect_left_bottom_pt2",
    9: "sideline_bottom_left",
    10: "left_semicircle_right",
    11: "center_line_top",
    12: "center_line_bottom",
    13: "center_circle_top",
    14: "center_circle_bottom",
    15: "field_center",
    16: "sideline_top_right",
    17: "big_rect_right_top_pt1",
    18: "big_rect_right_top_pt2",
    19: "big_rect_right_bottom_pt1",
    20: "big_rect_right_bottom_pt2",
    21: "small_rect_right_top_pt1",
    22: "small_rect_right_top_pt2",
    23: "small_rect_right_bottom_pt1",
    24: "small_rect_right_bottom_pt2",
    25: "sideline_bottom_right",
    26: "right_semicircle_left",
    27: "center_circle_left",
    28: "center_circle_right",
}

# Soccer field keypoint names (57 keypoints)
KEYPOINT_NAMES_57 = {
    0:  "L_GOAL_TL_POST",
    1:  "L_GOAL_TR_POST",
    2:  "L_GOAL_BL_POST",
    3:  "L_GOAL_BR_POST",
    4:  "L_GOAL_AREA_BR_CORNER",
    5:  "L_GOAL_AREA_TR_CORNER",
    6:  "L_GOAL_AREA_BL_CORNER",
    7:  "L_GOAL_AREA_TL_CORNER",
    8:  "L_PENALTY_AREA_BR_CORNER",
    9:  "L_PENALTY_AREA_TR_CORNER",
    10: "L_PENALTY_AREA_BL_CORNER",
    11: "L_PENALTY_AREA_TL_CORNER",
    12: "BL_PITCH_CORNER",
    13: "TL_PITCH_CORNER",
    14: "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
    15: "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
    16: "R_PENALTY_AREA_BL_CORNER",
    17: "R_PENALTY_AREA_TL_CORNER",
    18: "R_PENALTY_AREA_BR_CORNER",
    19: "R_PENALTY_AREA_TR_CORNER",
    20: "R_GOAL_AREA_BL_CORNER",
    21: "R_GOAL_AREA_TL_CORNER",
    22: "R_GOAL_AREA_BR_CORNER",
    23: "R_GOAL_AREA_TR_CORNER",
    24: "R_GOAL_TL_POST",
    25: "R_GOAL_TR_POST",
    26: "R_GOAL_BL_POST",
    27: "R_GOAL_BR_POST",
    28: "BR_PITCH_CORNER",
    29: "TR_PITCH_CORNER",
    30: "CENTER_CIRCLE_TANGENT_TR",
    31: "CENTER_CIRCLE_TANGENT_TL",
    32: "CENTER_CIRCLE_TANGENT_BR",
    33: "CENTER_CIRCLE_TANGENT_BL",
    34: "CENTER_CIRCLE_TR",
    35: "CENTER_CIRCLE_TL",
    36: "CENTER_CIRCLE_BR",
    37: "CENTER_CIRCLE_BL",
    38: "CENTER_CIRCLE_R",
    39: "CENTER_CIRCLE_L",
    40: "T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION",
    41: "B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION",
    42: "CENTER_MARK",
    43: "LEFT_CIRCLE_R",
    44: "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    45: "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    46: "LEFT_CIRCLE_TANGENT_T",
    47: "LEFT_CIRCLE_TANGENT_B",
    48: "L_PENALTY_MARK",
    49: "L_MIDDLE_PENALTY",
    50: "RIGHT_CIRCLE_L",
    51: "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    52: "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    53: "RIGHT_CIRCLE_TANGENT_T",
    54: "RIGHT_CIRCLE_TANGENT_B",
    55: "R_PENALTY_MARK",
    56: "R_MIDDLE_PENALTY",
}

# Keypoint indices on the left and right halves of the pitch
POINTS_LEFT_57 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  31, 33, 35, 37, 39, 43, 44, 45, 46, 47, 48, 49]
POINTS_RIGHT_57 = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                   29, 30, 32, 34, 36, 38, 50, 51, 52, 53, 54, 55, 56]


# ================================================
# Core detection functions
# ================================================

def load_keypoint_model(model_path: str) -> YOLO:
    """Load and return a YOLO pose estimation model for keypoint detection.

    Args:
        model_path: Path to the YOLO pose model file

    Returns:
        YOLO model instance configured for pose estimation
    """
    model = YOLO(model_path)
    return model


def detect_keypoints_in_frames(model: YOLO, frames) -> List:
    """Detect keypoints in video frames using a YOLO pose model.

    Args:
        model: Loaded YOLO pose model
        frames: Video frames or a single frame

    Returns:
        Detection results from the YOLO pose model, containing keypoints
    """
    return model(frames)


def get_keypoint_detections(keypoint_model: YOLO, frame: np.ndarray) -> Tuple[sv.Detections, np.ndarray]:
    """Get keypoint detections and extract the keypoint coordinates.

    Args:
        keypoint_model: Loaded YOLO pose model
        frame: Input frame as a numpy array

    Returns:
        Tuple of (detections, keypoints), where keypoints has shape (N, K, 3):
        N detections with K keypoints each, holding (x, y, visibility).
        K is 29 or 57 depending on the loaded model.
    """
    results = detect_keypoints_in_frames(keypoint_model, frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    # Extract keypoints when the model provides them
    keypoints = None
    if hasattr(results, 'keypoints') and results.keypoints is not None:
        keypoints = results.keypoints.data.cpu().numpy()  # Shape: (N, K, 3)

    return detections, keypoints


def normalize_keypoints(keypoints: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Normalize keypoint coordinates to the 0-1 range.

    Args:
        keypoints: Array of keypoints with shape (N, K, 3)
        image_width: Width of the source image
        image_height: Height of the source image

    Returns:
        Normalized keypoints array with coordinates in the 0-1 range
    """
    if keypoints is None or keypoints.size == 0:
        return keypoints

    normalized_keypoints = keypoints.copy()
    normalized_keypoints[:, :, 0] /= image_width   # Normalize x coordinates
    normalized_keypoints[:, :, 1] /= image_height  # Normalize y coordinates
    # Visibility values (index 2) remain unchanged

    return normalized_keypoints


def denormalize_keypoints(keypoints: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Denormalize keypoint coordinates from the 0-1 range to image coordinates.

    Args:
        keypoints: Array of normalized keypoints with shape (N, K, 3)
        image_width: Width of the target image
        image_height: Height of the target image

    Returns:
        Denormalized keypoints array with pixel coordinates
    """
    if keypoints is None or keypoints.size == 0:
        return keypoints

    denormalized_keypoints = keypoints.copy()
    denormalized_keypoints[:, :, 0] *= image_width   # Denormalize x coordinates
    denormalized_keypoints[:, :, 1] *= image_height  # Denormalize y coordinates
    # Visibility values remain unchanged

    return denormalized_keypoints


def filter_visible_keypoints(keypoints: np.ndarray, confidence_threshold: float = 0.5) -> np.ndarray:
    """Filter keypoints based on visibility confidence.

    Args:
        keypoints: Array of keypoints with shape (N, K, 3)
        confidence_threshold: Minimum confidence for a keypoint to count as visible

    Returns:
        Filtered keypoints, with low-confidence points set to (0, 0, 0)
    """
    if keypoints is None or keypoints.size == 0:
        return keypoints

    filtered_keypoints = keypoints.copy()

    # Zero out the keypoints that are not visible enough
    invisible_mask = keypoints[:, :, 2] < confidence_threshold
    filtered_keypoints[invisible_mask] = 0

    return filtered_keypoints


def extract_field_corners(keypoints: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Extract the four corner points of the soccer field from detected keypoints.

    Indices follow the 29-keypoint layout (KEYPOINT_NAMES_29).

    Args:
        keypoints: Array of keypoints with shape (N, 29, 3)

    Returns:
        Dictionary containing the corner coordinates:
        {'top_left': (x, y), 'top_right': (x, y),
         'bottom_left': (x, y), 'bottom_right': (x, y)}
    """
    if keypoints is None or keypoints.size == 0:
        return {'top_left': (0, 0), 'top_right': (0, 0),
                'bottom_left': (0, 0), 'bottom_right': (0, 0)}

    corners = {}

    if keypoints.shape[0] > 0:
        kpts = keypoints[0]  # First detection

        # Indices of the field boundary points in the 29-keypoint layout
        corners['top_left'] = (float(kpts[0, 0]), float(kpts[0, 1])) if kpts[0, 2] > 0 else (0, 0)
        corners['top_right'] = (float(kpts[16, 0]), float(kpts[16, 1])) if kpts[16, 2] > 0 else (0, 0)
        corners['bottom_left'] = (float(kpts[9, 0]), float(kpts[9, 1])) if kpts[9, 2] > 0 else (0, 0)
        corners['bottom_right'] = (float(kpts[25, 0]), float(kpts[25, 1])) if kpts[25, 2] > 0 else (0, 0)

    return corners


def calculate_field_dimensions(corners: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Calculate the field dimensions from the corner keypoints.

    Args:
        corners: Dictionary of corner coordinates

    Returns:
        Dictionary containing the field measurements:
        {'width': field_width, 'height': field_height, 'area': field_area}
    """
    top_left = corners['top_left']
    top_right = corners['top_right']
    bottom_left = corners['bottom_left']

    # Calculate the field dimensions
    width = np.sqrt((top_right[0] - top_left[0]) ** 2 + (top_right[1] - top_left[1]) ** 2)
    height = np.sqrt((bottom_left[0] - top_left[0]) ** 2 + (bottom_left[1] - top_left[1]) ** 2)
    area = width * height

    return {
        'width': float(width),
        'height': float(height),
        'area': float(area)
    }
