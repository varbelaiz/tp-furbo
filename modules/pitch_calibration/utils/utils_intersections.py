from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from numpy.polynomial import polynomial as P

from utils.utils_ellipse_helpers import add_conic_points

EPS = 1e-18

# Maps each keypoint id to the pair of lines whose
# intersection defines it.
LINE_INTERSECTIONS: Dict[int, Tuple[str, str]] = {
    0: ('Goal left crossbar', 'Goal left post left '),
    1: ('Goal left crossbar', 'Goal left post right'),
    2: ('Side line left', 'Goal left post left '),
    3: ('Side line left', 'Goal left post right'),
    4: ('Small rect. left main', 'Small rect. left bottom'),
    5: ('Small rect. left main', 'Small rect. left top'),
    6: ('Side line left', 'Small rect. left bottom'),
    7: ('Side line left', 'Small rect. left top'),
    8: ('Big rect. left main', 'Big rect. left bottom'),
    9: ('Big rect. left main', 'Big rect. left top'),
    10: ('Side line left', 'Big rect. left bottom'),
    11: ('Side line left', 'Big rect. left top'),
    12: ('Side line left', 'Side line bottom'),
    13: ('Side line left', 'Side line top'),
    14: ('Middle line', 'Side line bottom'),
    15: ('Middle line', 'Side line top'),
    16: ('Big rect. right main', 'Big rect. right bottom'),
    17: ('Big rect. right main', 'Big rect. right top'),
    18: ('Side line right', 'Big rect. right bottom'),
    19: ('Side line right', 'Big rect. right top'),
    20: ('Small rect. right main', 'Small rect. right bottom'),
    21: ('Small rect. right main', 'Small rect. right top'),
    22: ('Side line right', 'Small rect. right bottom'),
    23: ('Side line right', 'Small rect. right top'),
    24: ('Goal right crossbar', 'Goal right post left'),
    25: ('Goal right crossbar', 'Goal right post right'),
    26: ('Side line right', 'Goal right post left'),
    27: ('Side line right', 'Goal right post right'),
    28: ('Side line right', 'Side line bottom'),
    29: ('Side line right', 'Side line top'),
}

def point_within_img(point: Optional[Tuple[float, float]],
                     img_size: Tuple[int, int] = (960, 540),
                     within_image: bool = True,
                     margin: float = 0.0) -> Optional[Tuple[float, float]]:
    """
    Check whether a point lies inside the image bounds.
    """
    if point is None:
        return None
    
    x, y = point
    W, H = img_size[0], img_size[1]
    
    if not within_image:
        # When within_image is False the point is returned
        # regardless of where it lies, which is what this pipeline wants.
        return point
    
    # Bounds check (within_image == True)
    if (x >= 0 - margin) and (x <= W + margin) and \
       (y >= 0 - margin) and (y <= H + margin):
        return point
    else:
        return None  # The point lies outside the image

def find_closest_points(line_arr: np.ndarray, x: float, y: float) \
        -> np.ndarray:
    """
    Find the 2 points in an array closest to a given (x, y) point.
    (Re-implementado para eliminar la dependencia de src.datatools.line)
    """
    distances = np.sqrt(np.sum((line_arr - np.array([x, y]))**2, axis=1))
    # Returns the 2 closest points, used by the recursive line fit
    closest_indices = np.argsort(distances)[:2]
    return line_arr[closest_indices]



def intersection(line1_arr: np.ndarray, line2_arr: np.ndarray)\
        -> Optional[Tuple[float, float]]:
    """
    Find the intersection point of two lines.

    Each line is given as a list of (x, y) tuples. 
    Each set of points is fitted to a straight line.

    Args:
        line1_arr (np.ndarray): First line, shape (N, 2).
        line2_arr (np.ndarray): Second line, shape (N, 2).

    Returns:
        Optional[Tuple[float, float]]: The intersection point.
            Nota: el punto puede estar fuera de la imagen.
    """

    x1, y1 = line1_arr[:, 0], line1_arr[:, 1]
    x2, y2 = line2_arr[:, 0], line2_arr[:, 1]
    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)
    
    # Check whether either line is vertical
    is_x1_line = np.all(np.isclose(x1, x1_mean, atol=0.5))
    is_x2_line = np.all(np.isclose(x2, x2_mean, atol=0.5))
    point = None
    
    if is_x1_line:  # Line 1 is vertical (x = constant)
        x = x1_mean
        if is_x2_line:
            return None  # Two parallel vertical lines never meet
        b2, a2 = P.polyfit(x2, y2, 1)
        y = a2 * x + b2
    elif is_x2_line:  # Line 2 is vertical (x = constant)
        x = x2_mean
        b1, a1 = P.polyfit(x1, y1, 1)
        y = a1 * x + b1
    else:  # Standard case: both lines have a slope
        b1, a1 = P.polyfit(x1, y1, 1)
        b2, a2 = P.polyfit(x2, y2, 1)
        x = (b2 - b1) / (a1 - a2 + EPS)  # Numerically stable division
        y = a1 * x + b1
        
    if line1_arr.shape[0] > 2 or line2_arr.shape[0] > 2:
        line1_arr = find_closest_points(line1_arr, x, y)
        line2_arr = find_closest_points(line2_arr, x, y)
        point = intersection(line1_arr, line2_arr)
    else:
        # Base case of the recursion (only 2 points per line)
        point = (x, y)
        
    return point


def get_intersections(points: Dict[str, List[Tuple[float, float]]],
                      img_size: Tuple[int, int] = (960, 540),
                      within_image: bool = True,
                      margin: float = 0.0)\
        -> Tuple[Dict[int, Tuple[float, float] | None], List[int]]:
    """
    Main entry point that returns every keypoint, from lines and conics.
    
    Args:
        points: Diccionario de anotaciones (ej. {"Side line top": [(x,y),...]})
        img_size: Image size in pixels (W, H).
        within_image: Flag para filtrar puntos fuera de la imagen.
        margin: Margen adicional para el filtrado.

    Returns:
        A dict of keypoints {id: (x, y)} and a mask (list[int]).
    """
    res: Dict[int, Tuple[float, float] | None] = {}
    
    # --- Part 1: line intersections (keypoints 0-29) ---
    for i, pair in LINE_INTERSECTIONS.items():
        res[i] = None
        # Check that both lines are annotated
        if pair[0] in points and pair[1] in points:
            if len(points[pair[0]]) > 1 and len(points[pair[1]]) > 1:
                # Compute the intersection
                res[i] = intersection(
                    np.array(points[pair[0]]) * img_size,
                    np.array(points[pair[1]]) * img_size)

    # --- Part 2: conic points (keypoints 30-56) ---
    res, mask = add_conic_points(points, res, img_size)

    res = {i: point_within_img(res[i], img_size, within_image, margin)
           for i in res}
           
    return res, mask