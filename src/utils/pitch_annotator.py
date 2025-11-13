import cv2
import dataclasses
import numpy as np
import supervision as sv
from typing import List, Optional, Tuple, Union
import math

# --- CONFIGURACIÓN DE HIGHLIGHT ---
COLOR_HIGHLIGHT = sv.Color.WHITE 
ANTI_ALIAS = cv2.LINE_AA
# ---------------------------------

@dataclasses.dataclass
class SoccerPitchConfiguration:
    width: float = 105.0
    height: float = 68.0
    center_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0
    penalty_area_width: float = 40.32
    penalty_area_height: float = 16.5
    goal_width: float = 7.32
    goal_height: float = 5.5
    goal_area_width: float = 18.32
    goal_area_height: float = 5.5


@dataclasses.dataclass
class ViewTransformer:
    source: np.ndarray
    target: np.ndarray

    def __post_init__(self) -> None:
        self.matrix = cv2.getPerspectiveTransform(
            self.source.astype(np.float32), self.target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.matrix)
        return transformed_points.reshape(-1, 2)


def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color.from_hex("#2f8613"),
    line_color: sv.Color = sv.Color.WHITE,
    line_thickness: int = 2,
    pitch_width_pixels: int = 1000,
) -> Tuple[np.ndarray, ViewTransformer]:
    
    pitch_height_pixels = int(
        pitch_width_pixels * (config.height / config.width)
    )
    pitch = np.full(
        (pitch_height_pixels, pitch_width_pixels, 3),
        background_color.as_bgr(),
        dtype=np.uint8,
    )
    
    transformer = ViewTransformer(
        source=np.array(
            [
                [-config.width / 2, -config.height / 2],
                [config.width / 2, -config.height / 2],
                [-config.width / 2, config.height / 2],
                [config.width / 2, config.height / 2],
            ]
        ),
        target=np.array(
            [
                [0, 0],
                [pitch_width_pixels, 0],
                [0, pitch_height_pixels],
                [pitch_width_pixels, pitch_height_pixels],
            ]
        ),
    )

    def _t(points: np.ndarray) -> np.ndarray:
        return transformer.transform_points(points=points).astype(int)

    # pitch boundaries
    points = _t(
        points=np.array(
            [
                [-config.width / 2, -config.height / 2],
                [config.width / 2, -config.height / 2],
                [config.width / 2, config.height / 2],
                [-config.width / 2, config.height / 2],
            ]
        )
    )
    cv2.polylines(
        pitch, [points], True, line_color.as_bgr(), line_thickness, lineType=ANTI_ALIAS
    )

    # center line
    points = _t(
        points=np.array([[0, -config.height / 2], [0, config.height / 2]])
    )
    cv2.line(
        pitch,
        tuple(points[0]),
        tuple(points[1]),
        line_color.as_bgr(),
        line_thickness,
        lineType=ANTI_ALIAS
    )

    center_point = _t(points=np.array([[0, 0]]))[0]
    
    # Calcular radio UNA VEZ, en el eje X
    radius = _t(points=np.array([[config.center_circle_radius, 0]]))[0][0] - center_point[0]
    
    cv2.circle( 
        pitch,
        tuple(center_point),
        radius, # Usar radius
        line_color.as_bgr(),
        line_thickness,
        lineType=ANTI_ALIAS
    )

    # left penalty area
    points = _t(
        points=np.array(
            [
                [-config.width / 2, -config.penalty_area_width / 2],
                [
                    -config.width / 2 + config.penalty_area_height,
                    -config.penalty_area_width / 2,
                ],
                [
                    -config.width / 2 + config.penalty_area_height,
                    config.penalty_area_width / 2,
                ],
                [-config.width / 2, config.penalty_area_width / 2],
            ]
        )
    )
    cv2.polylines(
        pitch, [points], False, line_color.as_bgr(), line_thickness, lineType=ANTI_ALIAS
    )

    # right penalty area
    points = _t(
        points=np.array(
            [
                [config.width / 2, -config.penalty_area_width / 2],
                [
                    config.width / 2 - config.penalty_area_height,
                    -config.penalty_area_width / 2,
                ],
                [
                    config.width / 2 - config.penalty_area_height,
                    config.penalty_area_width / 2,
                ],
                [config.width / 2, config.penalty_area_width / 2],
            ]
        )
    )
    cv2.polylines(
        pitch, [points], False, line_color.as_bgr(), line_thickness, lineType=ANTI_ALIAS
    )

    # left goal area
    points = _t(
        points=np.array(
            [
                [-config.width / 2, -config.goal_area_width / 2],
                [-config.width / 2 + config.goal_area_height, -config.goal_area_width / 2],
                [-config.width / 2 + config.goal_area_height, config.goal_area_width / 2],
                [-config.width / 2, config.goal_area_width / 2],
            ]
        )
    )
    cv2.polylines(
        pitch, [points], False, line_color.as_bgr(), line_thickness, lineType=ANTI_ALIAS
    )

    # right goal area
    points = _t(
        points=np.array(
            [
                [config.width / 2, -config.goal_area_width / 2],
                [config.width / 2 - config.goal_area_height, -config.goal_area_width / 2],
                [config.width / 2 - config.goal_area_height, config.goal_area_width / 2],
                [config.width / 2, config.goal_area_width / 2],
            ]
        )
    )
    cv2.polylines(
        pitch, [points], False, line_color.as_bgr(), line_thickness, lineType=ANTI_ALIAS
    )

    # left penalty arc
    center_point = _t(
        points=np.array(
            [[-config.width / 2 + config.penalty_spot_distance, 0]]
        )
    )[0]
    x_rel = config.penalty_area_height - config.penalty_spot_distance
    angle_rad = math.acos(x_rel / config.center_circle_radius)
    angle_deg = math.degrees(angle_rad)
    cv2.ellipse(
        pitch,
        tuple(center_point),
        (radius, radius), # Usar (radius, radius)
        0,
        360 - angle_deg,
        360 + angle_deg,
        line_color.as_bgr(),
        line_thickness,
        lineType=ANTI_ALIAS
    )

    # right penalty arc
    center_point = _t(
        points=np.array(
            [[config.width / 2 - config.penalty_spot_distance, 0]]
        )
    )[0]
    cv2.ellipse(
        pitch,
        tuple(center_point),
        (radius, radius), # Usar (radius, radius)
        0,
        180 - angle_deg,
        180 + angle_deg,
        line_color.as_bgr(),
        line_thickness,
        lineType=ANTI_ALIAS
    )
    
    return pitch, transformer


def draw_points_on_pitch(
    pitch: np.ndarray,
    xy: np.ndarray,
    transformer: ViewTransformer,
    face_color: sv.Color,
    edge_color: sv.Color,
    radius: int = 16,
    thickness: int = 2,
    text: Optional[List[str]] = None,
    text_color: sv.Color = sv.Color.BLACK,
    text_scale: float = 0.5,
    text_thickness: int = 1,
) -> np.ndarray:
    xy = transformer.transform_points(points=xy)
    for i, (x, y) in enumerate(xy):
        cv2.circle(
            pitch, (int(x), int(y)), radius, face_color.as_bgr(), -1, lineType=ANTI_ALIAS
        )
        cv2.circle(
            pitch, (int(x), int(y)), radius, edge_color.as_bgr(), thickness, lineType=ANTI_ALIAS
        )
        if text and text[i]:
            text_size, _ = cv2.getTextSize(
                text[i], cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness
            )
            text_x = int(x - text_size[0] / 2)
            text_y = int(y + text_size[1] / 2)
            cv2.putText(
                pitch,
                text[i],
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                text_color.as_bgr(),
                text_thickness,
                lineType=ANTI_ALIAS
            )
    return pitch


def draw_highlight_on_pitch(
    pitch: np.ndarray, 
    transformer: ViewTransformer,
    trajectory: list, 
    label: str
) -> np.ndarray:
    
    points_world = np.array(trajectory, dtype=np.float32)
    points_radar = transformer.transform_points(points=points_world)
    
    if len(points_radar) > 1:
        pts = points_radar.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(pitch, [pts], isClosed=False, color=COLOR_HIGHLIGHT.as_bgr(), thickness=1, lineType=ANTI_ALIAS)
        
        text_anchor_point = points_radar[len(points_radar) // 3]
        text_x = int(text_anchor_point[0]) + 5
        text_y = int(text_anchor_point[1]) - 5
        
        cv2.putText(pitch, label.upper() + "!", 
                    (text_x, text_y), 
                    cv2.FONT_HERSHEY_DUPLEX, 
                    0.6, 
                    COLOR_HIGHLIGHT.as_bgr(), 
                    2, 
                    lineType=ANTI_ALIAS)
    return pitch