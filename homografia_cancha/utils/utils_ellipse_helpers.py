"""
Functions for ellipses geometry operations.
(Versión Corregida/Parchada)

Ellipses are assumed to be defined by a*x^2+b*x*y+c*y^2+d*x+e*y+f=0,
lines: k*x+h=y.
"""
import warnings
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from ellipse import LsqEllipse


PITCH_POINTS = {}

PITCH_POINTS['CENTER_MARK'] = np.array([0.0, 0.0, 0.0])
CENTER_CIRCLE_RADIUS = 9.15
PITCH_POINTS['T_TOUCH_AND_HALFWAY_LINES_INTERSECTION'] = np.array([0.0, -34.0, 0.0]) 
PITCH_POINTS['B_TOUCH_AND_HALFWAY_LINES_INTERSECTION'] = np.array([0.0, 34.0, 0.0])  
PITCH_POINTS['L_PENALTY_MARK'] = np.array([-41.5, 0.0, 0.0]) 
PITCH_POINTS['R_PENALTY_MARK'] = np.array([41.5, 0.0, 0.0])
PITCH_POINTS['L_PENALTY_AREA_TL_CORNER'] = np.array([-36.5, -20.16, 0.0]) 
PITCH_POINTS['L_PENALTY_AREA_TR_CORNER'] = np.array([-36.5, 20.16, 0.0])
PITCH_POINTS['L_PENALTY_AREA_BR_CORNER'] = np.array([-52.5, 20.16, 0.0]) 
PITCH_POINTS['R_PENALTY_AREA_TL_CORNER'] = np.array([36.5, -20.16, 0.0])
PITCH_POINTS['R_PENALTY_AREA_BL_CORNER'] = np.array([52.5, -20.16, 0.0])
PITCH_POINTS['L_PENALTY_AREA_BL_CORNER'] = np.array([-52.5, -20.16, 0.0])
PITCH_POINTS['R_PENALTY_AREA_BR_CORNER'] = np.array([36.5, 20.16, 0.0])
PITCH_POINTS['R_PENALTY_AREA_TR_CORNER'] = np.array([52.5, 20.16, 0.0])


def circle_tangent_points(circle_center: Tuple[float, float],
                          radius: float,
                          point: Tuple[float, float]):
    hypotenuse = np.sqrt((point[0] - circle_center[0])**2
                         + (point[1] - circle_center[1])**2)
    
    if hypotenuse <= radius:

        invalid_point = np.array([np.nan, np.nan, 0.0], dtype=float)
        return (invalid_point, invalid_point)
        
    th = np.arccos(radius / hypotenuse)
    d = np.arctan2(point[1] - circle_center[1], point[0] - circle_center[0])
    d1 = d + th
    d2 = d - th
    return (np.array((circle_center[0] + radius * np.cos(d1),
                      circle_center[1] + radius * np.sin(d1), 0), dtype=float),
            np.array((circle_center[0] + radius * np.cos(d2),
                      circle_center[1] + radius * np.sin(d2), 0), dtype=float))


top_tanget = circle_tangent_points(
    PITCH_POINTS['CENTER_MARK'][:2], CENTER_CIRCLE_RADIUS,
    PITCH_POINTS['T_TOUCH_AND_HALFWAY_LINES_INTERSECTION'][:2])

bottom_tanget = circle_tangent_points(
    PITCH_POINTS['CENTER_MARK'][:2], CENTER_CIRCLE_RADIUS,
    PITCH_POINTS['B_TOUCH_AND_HALFWAY_LINES_INTERSECTION'][:2])

PITCH_POINTS['CENTER_CIRCLE_TANGENT_TR'] = top_tanget[0]
PITCH_POINTS['CENTER_CIRCLE_TANGENT_TL'] = top_tanget[1]
PITCH_POINTS['CENTER_CIRCLE_TANGENT_BR'] = bottom_tanget[1]
PITCH_POINTS['CENTER_CIRCLE_TANGENT_BL'] = bottom_tanget[0]

circle_sq = np.sqrt(2.) * CENTER_CIRCLE_RADIUS / 2
PITCH_POINTS['CENTER_CIRCLE_TR'] = np.array([circle_sq, -circle_sq, 0],
                                            dtype=float)
PITCH_POINTS['CENTER_CIRCLE_TL'] = np.array([-circle_sq, -circle_sq, 0],
                                            dtype=float)
PITCH_POINTS['CENTER_CIRCLE_BR'] = np.array([circle_sq, circle_sq, 0],
                                            dtype=float)
PITCH_POINTS['CENTER_CIRCLE_BL'] = np.array([-circle_sq, circle_sq, 0],
                                            dtype=float)

PITCH_POINTS['CENTER_CIRCLE_R'] = np.array([CENTER_CIRCLE_RADIUS, 0, 0],
                                           dtype=float)
PITCH_POINTS['CENTER_CIRCLE_L'] = np.array([-CENTER_CIRCLE_RADIUS, 0, 0],
                                           dtype=float)

PITCH_POINTS['LEFT_CIRCLE_R'] = PITCH_POINTS['L_PENALTY_MARK'] + \
    PITCH_POINTS['CENTER_CIRCLE_R']
PITCH_POINTS['RIGHT_CIRCLE_L'] = PITCH_POINTS['R_PENALTY_MARK'] + \
    PITCH_POINTS['CENTER_CIRCLE_L']

PITCH_POINTS['LEFT_CIRCLE_TANGENT_T'] = circle_tangent_points(
    PITCH_POINTS['L_PENALTY_MARK'][:2], CENTER_CIRCLE_RADIUS,
    PITCH_POINTS['L_PENALTY_AREA_TR_CORNER'][:2])[0]

PITCH_POINTS['LEFT_CIRCLE_TANGENT_B'] = circle_tangent_points(
    PITCH_POINTS['L_PENALTY_MARK'][:2], CENTER_CIRCLE_RADIUS,
    PITCH_POINTS['L_PENALTY_AREA_BR_CORNER'][:2])[1]
PITCH_POINTS['L_MIDDLE_PENALTY'] = PITCH_POINTS['L_PENALTY_AREA_BR_CORNER'].copy()
PITCH_POINTS['L_MIDDLE_PENALTY'][1] = 0.0


PITCH_POINTS['RIGHT_CIRCLE_TANGENT_T'] = circle_tangent_points(
    PITCH_POINTS['R_PENALTY_MARK'][:2], CENTER_CIRCLE_RADIUS,
    PITCH_POINTS['R_PENALTY_AREA_TL_CORNER'][:2])[1]

PITCH_POINTS['RIGHT_CIRCLE_TANGENT_B'] = circle_tangent_points(
    PITCH_POINTS['R_PENALTY_MARK'][:2], CENTER_CIRCLE_RADIUS,
    PITCH_POINTS['R_PENALTY_AREA_BL_CORNER'][:2])[0]
PITCH_POINTS['R_MIDDLE_PENALTY'] = PITCH_POINTS['R_PENALTY_AREA_BL_CORNER'].copy()
PITCH_POINTS['R_MIDDLE_PENALTY'][1] = 0.0


def get_pitch():
    PITCH_POINTS['L_GOAL_TL_POST'] = np.array([-52.5, -3.66, 2.44]) 
    PITCH_POINTS['L_GOAL_TR_POST'] = np.array([-52.5, 3.66, 2.44]) 
    PITCH_POINTS['L_GOAL_BL_POST'] = np.array([-52.5, -3.66, 0.0])
    PITCH_POINTS['L_GOAL_BR_POST'] = np.array([-52.5, 3.66, 0.0])
    PITCH_POINTS['L_GOAL_AREA_BR_CORNER'] = np.array([-47.0, 9.16, 0.0]) 
    PITCH_POINTS['L_GOAL_AREA_TR_CORNER'] = np.array([-47.0, -9.16, 0.0])
    PITCH_POINTS['L_GOAL_AREA_BL_CORNER'] = np.array([-52.5, 9.16, 0.0])
    PITCH_POINTS['L_GOAL_AREA_TL_CORNER'] = np.array([-52.5, -9.16, 0.0])
    PITCH_POINTS['BL_PITCH_CORNER'] = np.array([-52.5, 34.0, 0.0])
    PITCH_POINTS['TL_PITCH_CORNER'] = np.array([-52.5, -34.0, 0.0])
    
    PITCH_POINTS['R_GOAL_TL_POST'] = np.array([52.5, -3.66, 2.44]) 
    PITCH_POINTS['R_GOAL_TR_POST'] = np.array([52.5, 3.66, 2.44]) 
    PITCH_POINTS['R_GOAL_BL_POST'] = np.array([52.5, -3.66, 0.0])
    PITCH_POINTS['R_GOAL_BR_POST'] = np.array([52.5, 3.66, 0.0])
    PITCH_POINTS['R_GOAL_AREA_BL_CORNER'] = np.array([47.0, -9.16, 0.0])
    PITCH_POINTS['R_GOAL_AREA_TL_CORNER'] = np.array([47.0, 9.16, 0.0])
    PITCH_POINTS['R_GOAL_AREA_BR_CORNER'] = np.array([52.5, -9.16, 0.0])
    PITCH_POINTS['R_GOAL_AREA_TR_CORNER'] = np.array([52.5, 9.16, 0.0])
    PITCH_POINTS['BR_PITCH_CORNER'] = np.array([52.5, 34.0, 0.0])
    PITCH_POINTS['TR_PITCH_CORNER'] = np.array([52.5, -34.0, 0.0])

    PITCH_POINTS['T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION'] = np.array([0.0, -9.15, 0.0])
    PITCH_POINTS['B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION'] = np.array([0.0, 9.15, 0.0])
    PITCH_POINTS['BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION'] = np.array([-36.5, 7.86, 0.0]) 
    PITCH_POINTS['TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION'] = np.array([-36.5, -7.86, 0.0]) 
    PITCH_POINTS['BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION'] = np.array([36.5, 7.86, 0.0]) 
    PITCH_POINTS['TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION'] = np.array([36.5, -7.86, 0.0]) 

    return PITCH_POINTS


INTERSECTON_TO_PITCH_POINTS = {
    0: 'L_GOAL_TL_POST',
    1: 'L_GOAL_TR_POST',
    2: 'L_GOAL_BL_POST',
    3: 'L_GOAL_BR_POST',
    4: 'L_GOAL_AREA_BR_CORNER',
    5: 'L_GOAL_AREA_TR_CORNER',
    6: 'L_GOAL_AREA_BL_CORNER',
    7: 'L_GOAL_AREA_TL_CORNER',
    8: 'L_PENALTY_AREA_BR_CORNER',
    9: 'L_PENALTY_AREA_TR_CORNER',
    10: 'L_PENALTY_AREA_BL_CORNER',
    11: 'L_PENALTY_AREA_TL_CORNER',
    12: 'BL_PITCH_CORNER',
    13: 'TL_PITCH_CORNER',
    14: 'B_TOUCH_AND_HALFWAY_LINES_INTERSECTION',
    15: 'T_TOUCH_AND_HALFWAY_LINES_INTERSECTION',
    16: 'R_PENALTY_AREA_BL_CORNER',
    17: 'R_PENALTY_AREA_TL_CORNER',
    18: 'R_PENALTY_AREA_BR_CORNER',
    19: 'R_PENALTY_AREA_TR_CORNER',
    20: 'R_GOAL_AREA_BL_CORNER',
    21: 'R_GOAL_AREA_TL_CORNER',
    22: 'R_GOAL_AREA_BR_CORNER',
    23: 'R_GOAL_AREA_TR_CORNER',
    24: 'R_GOAL_TL_POST',
    25: 'R_GOAL_TR_POST',
    26: 'R_GOAL_BL_POST',
    27: 'R_GOAL_BR_POST',
    28: 'BR_PITCH_CORNER',
    29: 'TR_PITCH_CORNER',
    30: 'CENTER_CIRCLE_TANGENT_TR',
    31: 'CENTER_CIRCLE_TANGENT_TL',
    32: 'CENTER_CIRCLE_TANGENT_BR',
    33: 'CENTER_CIRCLE_TANGENT_BL',
    34: 'CENTER_CIRCLE_TR',
    35: 'CENTER_CIRCLE_TL',
    36: 'CENTER_CIRCLE_BR',
    37: 'CENTER_CIRCLE_BL',
    38: 'CENTER_CIRCLE_R',
    39: 'CENTER_CIRCLE_L',
    40: 'T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION',
    41: 'B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION',
    42: 'CENTER_MARK',
    43: 'LEFT_CIRCLE_R',
    44: 'BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION',
    45: 'TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION',
    46: 'LEFT_CIRCLE_TANGENT_T',
    47: 'LEFT_CIRCLE_TANGENT_B',
    48: 'L_PENALTY_MARK',
    49: 'L_MIDDLE_PENALTY',
    50: 'RIGHT_CIRCLE_L',
    51: 'BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION',
    52: 'TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION',
    53: 'RIGHT_CIRCLE_TANGENT_T',
    54: 'RIGHT_CIRCLE_TANGENT_B',
    55: 'R_PENALTY_MARK',
    56: 'R_MIDDLE_PENALTY'
}

PITCH_POINTS.update(get_pitch())

PERP_LINES: List[Tuple[int, int]] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
    (10, 11),
    (12, 13),
    (14, 15),
    (16, 17),
    (18, 19),
    (20, 21),
    (22, 23),
    (24, 25),
    (26, 27),
    (28, 29),
    (40, 41),
    (44, 45),
    (51, 52)
]


POINTS_LEFT: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 31, 33,
                          35, 37, 39, 43, 44, 45, 46, 47, 48, 49]
POINTS_RIGHT: List[int] = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                           29, 30, 32, 34, 36, 38, 50, 51, 52, 53, 54, 55, 56]

NOT_ON_PLANE = [0, 1, 24, 25]

PITCH_POINTS_TO_INTERSECTON = {v: k for k, v
                               in INTERSECTON_TO_PITCH_POINTS.items() if v in PITCH_POINTS}

CONICS_KEYS = {
    'Circle central': {
        'CENTER_CIRCLE_TANGENT_TR': {
            'type': 'tangent',
            'reference': 'T_TOUCH_AND_HALFWAY_LINES_INTERSECTION',
            'idx': 0
        },
        'CENTER_CIRCLE_TANGENT_TL': {
            'type': 'tangent',
            'reference': 'T_TOUCH_AND_HALFWAY_LINES_INTERSECTION',
            'idx': 1
        },
        'CENTER_CIRCLE_TANGENT_BR': {
            'type': 'tangent',
            'reference': 'B_TOUCH_AND_HALFWAY_LINES_INTERSECTION',
            'idx': 0
        },
        'CENTER_CIRCLE_TANGENT_BL': {
            'type': 'tangent',
            'reference': 'B_TOUCH_AND_HALFWAY_LINES_INTERSECTION',
            'idx': 1
        },
        'T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION': {
            'type': 'intersection',
            'line': 'Middle line',
            'side': 'Top'
        },
        'B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION': {
            'type': 'intersection',
            'line': 'Middle line',
            'side': 'Bottom'
        }
    },
    'Circle left': {
        'BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION': {
            'type': 'intersection',
            'line': 'Big rect. left main',
            'side': 'Bottom',
        },
        'TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION': {
            'type': 'intersection',
            'line': 'Big rect. left main',
            'side': 'Top',
        },
        'LEFT_CIRCLE_TANGENT_T': {
            'type': 'tangent',
            'reference': 'L_PENALTY_AREA_TR_CORNER',
            'idx': 0
        },
        'LEFT_CIRCLE_TANGENT_B': {
            'type': 'tangent',
            'reference': 'L_PENALTY_AREA_BR_CORNER',
            'idx': 1
        }
    },
    'Circle right': {
        'BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION': {
            'type': 'intersection',
            'line': 'Big rect. right main',
            'side': 'Bottom',
        },
        'TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION': {
            'type': 'intersection',
            'line': 'Big rect. right main',
            'side': 'Top',
        },
        'RIGHT_CIRCLE_TANGENT_T': {
            'type': 'tangent',
            'reference': 'R_PENALTY_AREA_TL_CORNER',
            'idx': 1
        },
        'RIGHT_CIRCLE_TANGENT_B': {
            'type': 'tangent',
            'reference': 'R_PENALTY_AREA_BL_CORNER',
            'idx': 0
        }
    }
}

def find_closest_points(line_arr: np.ndarray, x: float, y: float, *args) \
        -> Optional[np.ndarray]:
    """
    Encuentra los 2 puntos más cercanos en un array a un punto (x, y).
    (Re-implementado para eliminar la dependencia de src.datatools.line)
    """
    if line_arr is None or len(line_arr) == 0:
        return None
    distances = np.sqrt(np.sum((line_arr - np.array([x, y]))**2, axis=1))
    closest_indices = np.argsort(distances)[:2]
    if len(closest_indices) < 2:
        return None 
    return line_arr[closest_indices]


def get_m(a, b, c, d, e, f, x0, y0) -> Tuple[float, float]:
    """Helper function for getting the tangent points."""
    A = 4*a*c*x0*y0 + 2*a*e*x0 - b**2*x0*y0 - b*d*x0 - b*e*y0 - 2*b*f\
        + 2*c*d*y0 + d*e
    
    sqrt_term = (-4*a*c*f + a*e**2 + b**2*f - b*d*e + c*d**2) \
                * (a*x0**2 + b*x0*y0 + c*y0**2 + d*x0 + e*y0 + f)
    
    if sqrt_term < 0:
        return (np.nan, np.nan)

    B = 2*np.sqrt(sqrt_term)
    C = 4*a*c*x0**2 - b**2*x0**2 - 2*b*e*x0 + 4*c*d*x0 + 4*c*f - e**2
    
    if C == 0:
        return (np.nan, np.nan)
        
    return ((A - B) / C, (A + B) / C)


def get_x(a, b, c, d, e, m, x0, y0):
    """Helper function for getting the tangent points."""
    C = 2 * (a + b*m + c*m**2)
    if C == 0:
        return np.nan
    A = (b*m*x0 - b*y0 + 2*c*m**2*x0 - 2*c*m*y0 - d - e*m)
    return A / C


def find_tangent_point(ellipse: List[float], point: Tuple[float, float],
                       idx: int = 0) -> Tuple[float, float]:
    """Find the tangent point for an ellipse... (código original)"""
    a, b, c, d, e, f = ellipse
    x0, y0 = point
    m = get_m(a, b, c, d, e, f, x0, y0)
    
    if np.isnan(m[idx]):
        return (np.nan, np.nan) 

    x = get_x(a, b, c, d, e, m[idx], x0, y0)
    
    if np.isnan(x):
       return (np.nan, np.nan)

    y = m[idx]*x+y0-m[idx]*x0
    return (x, y)


def add_conic_points(points: Dict[str, List[Tuple[float, float]]],
                     intersections: Dict[int, Tuple[float, float] | None],
                     img_size: Tuple[int, int] = (960, 540)):
    mask = []

    for conic in CONICS_KEYS:
        ellipse = None
        if conic in points and len(points[conic]) > 4:
            conic_points = np.array(points[conic]) * img_size
            try:
                reg = LsqEllipse().fit(conic_points)
                if len(reg.coefficients) == 6:
                    ellipse = reg.coefficients
            except Exception:
                ellipse = None
                
        if ellipse is not None:
            for point_name in CONICS_KEYS[conic]:
                proc = CONICS_KEYS[conic][point_name]
                
                if point_name not in PITCH_POINTS_TO_INTERSECTON:
                    continue
                inters_id = PITCH_POINTS_TO_INTERSECTON[point_name]
                
                if proc['type'] == 'tangent':
                    ref_point_name = proc['reference']
                    if ref_point_name not in PITCH_POINTS_TO_INTERSECTON:
                        continue
                    ref_inters = PITCH_POINTS_TO_INTERSECTON[ref_point_name]
                    
                    if ref_inters in intersections\
                            and intersections[ref_inters] is not None:
                        coords = find_tangent_point(ellipse,
                                               intersections[ref_inters],
                                               proc['idx'])
                        if not np.isnan(coords[0]):
                            intersections[inters_id] = coords
                            
                elif proc['type'] == 'intersection':
                    line_name = proc['line']
                    if line_name in points and len(points[line_name]) > 1:
                        line_cooords = np.array(points[line_name]) * img_size
                        res = ellipse_line_intersect(ellipse, line_cooords)
                        if res is not None:
                            intersections[inters_id] = select_intersect(
                                res, points, img_size, conic, line_name,
                                proc['side']
                            )

    idxs = [i for i in intersections if intersections[i] is not None
            and i not in NOT_ON_PLANE]
    
    dst_points = []
    src_points_2d = []
    
    for i in idxs:
        point_name = INTERSECTON_TO_PITCH_POINTS.get(i)
        if point_name in PITCH_POINTS:
            dst_points.append(PITCH_POINTS[point_name][:2]) 
            src_points_2d.append(intersections[i]) 

    src = np.array(src_points_2d, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    filled_with_homography = False
    if len(src) >= 4: 
        try:
            hom = get_homography(dst, src) 
            if hom is not None:
                for idx in INTERSECTON_TO_PITCH_POINTS.keys():
                    if idx > 29 and (idx not in intersections
                                     or intersections[idx] is None):
                        
                        point_name = INTERSECTON_TO_PITCH_POINTS.get(idx)
                        if point_name in PITCH_POINTS:
                            world_p = PITCH_POINTS[point_name]
                            
                            img_p = cv2.perspectiveTransform(
                                np.array([[world_p[:2]]], dtype=np.float32), hom
                            )
                            intersections[idx] = img_p[0][0]
                filled_with_homography = True
        except Exception as e:
            warnings.warn(f'No se pudo crear la homografía: {e}')
            
    if not filled_with_homography:
        for idx in INTERSECTON_TO_PITCH_POINTS.keys():
            if idx > 29 and (idx not in intersections
                             or intersections[idx] is None):
                mask.append(idx)

    for i in INTERSECTON_TO_PITCH_POINTS:
        if i not in intersections:
            intersections[i] = None
    return intersections, mask


def select_intersect(res, points: Dict[str, List[Tuple[float, float]]],
                     img_size: Tuple[int, int], circle: str = 'Circle central',
                     line: str = 'Middle line',
                     inters_type: str = 'Top'):
    
    if res is None or len(res) < 2:
        return None 
        
    p1 = res[0].real
    p2 = res[1].real
    p_min_y = min(p1[1], p2[1])
    lr = False

    for line_name in [pn for pn in points.keys()
                      if 'left' in pn.split()[:3]
                      and pn not in (line, circle)]:
        y_list = [p[1]*img_size[1] for p in points[line_name]]
        if any([y > p_min_y for y in y_list]):
            lr = True
            break
    if circle == 'Circle left':
        y_list = [p[1]*img_size[1] for p in points[circle]]
        if any([(p_min_y-y) > 3 for y in y_list]):
            lr = True
    if circle == 'Circle right':
        y_list = [p[1]*img_size[1] for p in points[circle]]
        if any([(y-p_min_y) > 3 for y in y_list]):
            lr = True
    bottom, top = sort_conic_inters(p1, p2, lr)
    if inters_type == 'Bottom':
        return bottom
    elif inters_type == 'Top':
        return top
    return None 


def ellipse_line_intersect(ellipse, line: np.ndarray) -> np.ndarray | None:
    """Find intersection of a line and an ellipse. (Versión Robusta)"""
    conic_coeffs = ellipse
    res = None
    if len(conic_coeffs) == 6:
        line_x = line[:, 0]
        x_mean_val = np.mean(line_x)
        is_x_line = np.all(np.isclose(line_x, x_mean_val, atol=0.5))

        int_x, int_y = [], []
        if is_x_line:
            int_x, int_y = find_conic_y(*conic_coeffs, x_mean_val)
        else:
            finite_idx = np.isfinite(line[:, 0]) & np.isfinite(line[:, 1])
            if np.sum(finite_idx) < 2:
                return None

            try:
                a, b = np.polyfit(line[finite_idx, 0], line[finite_idx, 1], 1)
                int_x, int_y = quadratic_linear_intersection(
                    *conic_coeffs, a, b)
            except np.linalg.LinAlgError:
                 return None 

        if len(int_x) > 0:
            valid_results = []
            for i in range(len(int_x)):
                x = int_x[i]
                y = int_y[i]

                if np.isnan(x) or np.isnan(y):
                    continue

                l_line = find_closest_points(line, x, y)

                if l_line is not None:
                    l_int_x, l_int_y = [], []
                    try:
                        if np.isclose(l_line[0, 0], l_line[1, 0], atol=0.5):
                            l_int_x, l_int_y = find_conic_y(*conic_coeffs, l_line[0, 0])
                        else:
                            l_a, l_b = np.polyfit(
                                l_line[:, 0], l_line[:, 1], 1)
                            l_int_x, l_int_y = quadratic_linear_intersection(
                                *conic_coeffs, l_a, l_b)
                    except Exception:
                        pass 

                    if len(l_int_x) > 0:
                        valid_intersections = [(lx, ly) for lx, ly in zip(l_int_x, l_int_y) if not (np.isnan(lx) or np.isnan(ly))]
                        if not valid_intersections:
                            valid_results.append((x, y)) 
                            continue

                        l_int_x_valid, l_int_y_valid = zip(*valid_intersections)
                        dist = [(lx-x)**2+(ly-y)**2 for lx, ly in valid_intersections]
                        l_idx = np.argmin(dist)
                        valid_results.append((l_int_x_valid[l_idx], l_int_y_valid[l_idx]))
                    else:
                         valid_results.append((x, y))
                else:
                    valid_results.append((x, y)) 
                
            if valid_results:
                res = np.array(valid_results)
    return res


def sort_conic_inters(p1, p2, lr: bool = True):
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    if dy < 1.0 or (dx > 0 and dy > 0 and dx/dy > 10):
        if p1[0] < p2[0]:
            bottom, top = p2, p1
        else:
            bottom, top = p1, p2
        if not lr:
            bottom, top = top, bottom
    else:
        if p1[1] < p2[1]:
            bottom, top = p2, p1
        else:
            bottom, top = p1, p2

    return bottom, top


def get_homography(src, dst, threshold: float = 5.0):
    src_pts = np.array(src).reshape(-1, 1, 2)
    dst_pts = np.array(dst).reshape(-1, 1, 2)
    
    hom, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)
    return hom


def quadratic_linear_intersection(a, b, c, d, e, f, k, h)\
        -> Tuple[List[float], List[float]]:
    """Find intersections of a*x^2+b*x*y+c*y^2+d*x+e*y+f=0 and k*x+h=y.
    """
    def y(x):
        return k*x+h
    
    A = a + b*k + c*k**2
    B = b*h + 2*c*k*h + d + e*k
    C = c*h**2 + h*e + f

    if A == 0: 
        if B != 0:
            x_root = -C / B
            x_intersections = [x_root]
        else:
            x_intersections = []
    else:
        
        x_intersections = list(np.roots([A, B, C]))
    
    x_intersections = [x.real for x in x_intersections if np.isreal(x)]

    y_intersections = [y(x) for x in x_intersections]
    return x_intersections, y_intersections


def find_conic_y(a, b, c, d, e, f, x):
   
    A = c
    B = b*x + e
    C = a*x**2 + d*x + f

    if A == 0:
        if B != 0:
            y_root = -C / B
            y_intersections = [y_root]
        else:
            y_intersections = []
    else:
        y_intersections = list(np.roots([A, B, C]))

    y_intersections = [y.real for y in y_intersections if np.isreal(y)]
    
    x_intersections = [x] * len(y_intersections)
    return x_intersections, y_intersections