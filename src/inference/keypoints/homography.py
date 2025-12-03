import numpy as np
import cv2
from typing import Optional, Tuple

from src.inference.keypoints.detect_keypoints import POINTS_LEFT_57, POINTS_RIGHT_57


class SimpleViewTransformer:
    """
    Lightweight helper that applies a perspective transform to 2D points using a
    precomputed homography matrix.

    This is a trimmed-down version of the `ViewTransformer` class from Roboflow's
    `sports` library, reduced to drop external dependencies and focus purely on
    coordinate transformation.
    """

    def __init__(self, homography: np.ndarray):
        """
        Initialise the transformer.

        Args:
            homography (np.ndarray): 3x3 homography matrix.
        """
        self.homography = homography

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Apply the homography to a set of (x, y) points.

        Args:
            points (np.ndarray): Array of shape (N, 2) in image coordinates.

        Returns:
            np.ndarray: Array of shape (N, 2) with the points transformed onto
                the pitch plane (in cm). Returns an empty array on invalid input.
        """
        if points is None or len(points) == 0:
            return np.array([])

        # cv2.perspectiveTransform expects (N, 1, 2):
        # N points, 1 channel, 2 coordinates (x, y)
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)

        try:
            transformed_points = cv2.perspectiveTransform(points_reshaped, self.homography)
            return transformed_points.reshape(-1, 2)
        except cv2.error:
            return np.array([])


class HomographyTransformer:
    """
    Computes the homography between the keypoints detected in the image (pixels)
    and the real-world model (a football pitch measured in centimetres).
    """

    # Standard pitch dimensions in centimetres
    PITCH_LENGTH = 10500.0
    PITCH_WIDTH = 6800.0
    Y_CENTER = PITCH_WIDTH / 2.0
    X_CENTER = PITCH_LENGTH / 2.0

    # Box and goalpost dimensions
    R_POST = 366.0      # Post-to-post radius (approximate, for the 57-point model)
    D_GOAL = 550.0      # Goal area depth
    W_GOAL = 916.0      # Goal area half-width
    D_PEN = 1650.0      # Penalty area depth
    W_PEN = 2016.0      # Penalty area half-width
    R_CIRCLE = 915.0    # Centre circle radius

    def __init__(
        self,
        mode: str = '29',
        confidence_threshold: float = 0.5,
    ):
        """
        Configure the homography transformer.

        Args:
            mode (str): '57' for the extended model, '29' for the base YOLO model.
            confidence_threshold (float): Minimum confidence for a keypoint to be used.
        """
        self.mode = str(mode)
        self.confidence_threshold = confidence_threshold

        # Configuration depends on the chosen keypoint model
        if self.mode == '57':
            self.world_points = self._get_world_points_57()
            # Points that are usually out of frame or noisy (the crossbars)
            self.top_gates_indices = [0, 1, 24, 25]
            # Anchor indices used to filter out weak detections
            self.anchor_indices = set(list(range(2, 30)) + [14, 15, 42, 48, 55])
            self.left_side_indices = POINTS_LEFT_57
            self.right_side_indices = POINTS_RIGHT_57
            self.mapping_29 = None

        elif self.mode == '29':
            self.world_points, self.mapping_29 = self._get_world_points_29()
            self.top_gates_indices = []
            self.anchor_indices = None  # In 29-point mode every keypoint is used
            self.left_side_indices = list(range(0, 13))
            self.right_side_indices = list(range(16, 30))

        else:
            raise ValueError(f"Mode must be '57' or '29', got: {mode}")

    def _get_world_points_57(self) -> np.ndarray:
        """
        Build the 57-point pitch model.
        The ordering matches the 'extended' model.
        """
        pts = np.full((57, 2), [self.X_CENTER, self.Y_CENTER], dtype=np.float32)

        # --- Left-hand side ---
        # Goalposts
        pts[2] = [0, self.Y_CENTER + self.R_POST]
        pts[3] = [0, self.Y_CENTER - self.R_POST]
        # Goal area
        pts[6] = [0, self.Y_CENTER + self.W_GOAL]
        pts[7] = [0, self.Y_CENTER - self.W_GOAL]
        pts[4] = [self.D_GOAL, self.Y_CENTER + self.W_GOAL]
        pts[5] = [self.D_GOAL, self.Y_CENTER - self.W_GOAL]
        # Penalty area
        pts[10] = [0, self.Y_CENTER + self.W_PEN]
        pts[11] = [0, self.Y_CENTER - self.W_PEN]
        pts[8] = [self.D_PEN, self.Y_CENTER + self.W_PEN]
        pts[9] = [self.D_PEN, self.Y_CENTER - self.W_PEN]

        # --- Right-hand side ---
        # Goalposts
        pts[26] = [self.PITCH_LENGTH, self.Y_CENTER + self.R_POST]
        pts[27] = [self.PITCH_LENGTH, self.Y_CENTER - self.R_POST]
        # Goal area
        pts[22] = [self.PITCH_LENGTH, self.Y_CENTER + self.W_GOAL]
        pts[23] = [self.PITCH_LENGTH, self.Y_CENTER - self.W_GOAL]
        pts[20] = [self.PITCH_LENGTH - self.D_GOAL, self.Y_CENTER + self.W_GOAL]
        pts[21] = [self.PITCH_LENGTH - self.D_GOAL, self.Y_CENTER - self.W_GOAL]
        # Penalty area
        pts[18] = [self.PITCH_LENGTH, self.Y_CENTER + self.W_PEN]
        pts[19] = [self.PITCH_LENGTH, self.Y_CENTER - self.W_PEN]
        pts[16] = [self.PITCH_LENGTH - self.D_PEN, self.Y_CENTER + self.W_PEN]
        pts[17] = [self.PITCH_LENGTH - self.D_PEN, self.Y_CENTER - self.W_PEN]

        # --- Corners and halfway line ---
        # Corners
        pts[12] = [0, self.PITCH_WIDTH]
        pts[13] = [0, 0]
        pts[28] = [self.PITCH_LENGTH, self.PITCH_WIDTH]
        pts[29] = [self.PITCH_LENGTH, 0]
        # Halfway line
        pts[14] = [self.X_CENTER, self.PITCH_WIDTH]
        pts[15] = [self.X_CENTER, 0]
        pts[42] = [self.X_CENTER, self.Y_CENTER]  # Centre mark

        # Penalty marks (approximate for this model)
        pts[48] = [1100.0, self.Y_CENTER]
        pts[55] = [self.PITCH_LENGTH - 1100.0, self.Y_CENTER]

        return pts

    def _get_world_points_29(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Define the 29 semantic pitch points in centimetres.
        The mapping is compatible with standard YOLO keypoint models.
        """
        pts = np.zeros((29, 2), dtype=np.float32)

        # --- Left-hand side (0-10) ---
        pts[0] = [0, 0]                                     # Top-left corner
        pts[1] = [0, self.Y_CENTER - self.W_PEN]            # Penalty area, top, goal line
        pts[2] = [self.D_PEN, self.Y_CENTER - self.W_PEN]   # Penalty area, top, front corner
        pts[3] = [0, self.Y_CENTER + self.W_PEN]            # Penalty area, bottom, goal line
        pts[4] = [self.D_PEN, self.Y_CENTER + self.W_PEN]   # Penalty area, bottom, front corner

        pts[5] = [0, self.Y_CENTER - self.W_GOAL]           # Goal area, top, goal line
        pts[6] = [self.D_GOAL, self.Y_CENTER - self.W_GOAL]  # Goal area, top, front corner
        pts[7] = [0, self.Y_CENTER + self.W_GOAL]           # Goal area, bottom, goal line
        pts[8] = [self.D_GOAL, self.Y_CENTER + self.W_GOAL]  # Goal area, bottom, front corner

        pts[9] = [0, self.PITCH_WIDTH]                      # Bottom-left corner
        pts[10] = [1100.0 + self.R_CIRCLE, self.Y_CENTER]   # Tip of the D (penalty mark + radius)

        # --- Centre (11-15) ---
        pts[11] = [self.X_CENTER, 0]                              # Halfway line, top
        pts[12] = [self.X_CENTER, self.PITCH_WIDTH]               # Halfway line, bottom
        pts[13] = [self.X_CENTER, self.Y_CENTER - self.R_CIRCLE]  # Centre circle, top
        pts[14] = [self.X_CENTER, self.Y_CENTER + self.R_CIRCLE]  # Centre circle, bottom
        pts[15] = [self.X_CENTER, self.Y_CENTER]                  # Centre mark

        # --- Right-hand side (16-26) ---
        pts[16] = [self.PITCH_LENGTH, 0]                    # Top-right corner

        pts[17] = [self.PITCH_LENGTH, self.Y_CENTER - self.W_PEN]
        pts[18] = [self.PITCH_LENGTH - self.D_PEN, self.Y_CENTER - self.W_PEN]
        pts[19] = [self.PITCH_LENGTH, self.Y_CENTER + self.W_PEN]
        pts[20] = [self.PITCH_LENGTH - self.D_PEN, self.Y_CENTER + self.W_PEN]

        pts[21] = [self.PITCH_LENGTH, self.Y_CENTER - self.W_GOAL]
        pts[22] = [self.PITCH_LENGTH - self.D_GOAL, self.Y_CENTER - self.W_GOAL]
        pts[23] = [self.PITCH_LENGTH, self.Y_CENTER + self.W_GOAL]
        pts[24] = [self.PITCH_LENGTH - self.D_GOAL, self.Y_CENTER + self.W_GOAL]

        pts[25] = [self.PITCH_LENGTH, self.PITCH_WIDTH]     # Bottom-right corner
        pts[26] = [self.PITCH_LENGTH - (1100.0 + self.R_CIRCLE), self.Y_CENTER]  # Tip of the right-hand D

        # --- Extra centre circle points (27-28) ---
        pts[27] = [self.X_CENTER - self.R_CIRCLE, self.Y_CENTER]  # Centre circle, left
        pts[28] = [self.X_CENTER + self.R_CIRCLE, self.Y_CENTER]  # Centre circle, right

        # Identity mapping (0->0, 1->1, ...) because the points are defined in order
        mapping = np.arange(29, dtype=np.int32)

        return pts, mapping

    def transform_to_pitch_keypoints(self, detected_keypoints: np.ndarray) -> Optional[SimpleViewTransformer]:
        """
        Compute the homography matrix for a set of detected keypoints.

        Args:
            detected_keypoints (np.ndarray): Detected keypoints of shape (N, 3)
                -> [x, y, conf], or shape (1, N, 3).

        Returns:
            SimpleViewTransformer | None: The transformer if the homography is
                valid, None if it could not be computed.
        """
        if detected_keypoints is None or len(detected_keypoints) == 0:
            return None

        # Normalise the shape so it is always (N, 3)
        kps = detected_keypoints[0] if len(detected_keypoints.shape) == 3 else detected_keypoints.copy()

        # Drop the known noisy points (57-point mode only)
        if self.mode == '57' and self.top_gates_indices:
            kps[self.top_gates_indices, 2] = 0.0

        # Build the confidence mask
        mask = kps[:, 2] > self.confidence_threshold

        # Restrict to the allowed anchors, when applicable
        if self.anchor_indices is not None:
            for i in range(len(mask)):
                if i not in self.anchor_indices:
                    mask[i] = False

        # A homography needs at least 4 points
        if np.sum(mask) < 4:
            return None

        # Build the source (screen) and destination (pitch) point sets
        src = kps[mask, :2].astype(np.float32)

        if self.mode == '57':
            dst = self.world_points[mask].astype(np.float32)
        else:
            indices = np.where(mask)[0]
            dst = self.world_points[self.mapping_29[indices]].astype(np.float32)

        # Homography estimation with RANSAC
        H = None
        try:
            if len(src) > 4:
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            else:
                H, _ = cv2.findHomography(src, dst)
        except cv2.error:
            return None

        if H is not None:
            det = np.linalg.det(H)
            if np.abs(det) > 1e-5:
                return SimpleViewTransformer(H)

        return None
