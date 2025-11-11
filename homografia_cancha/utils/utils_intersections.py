from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from numpy.polynomial import polynomial as P

# --- Dependencia Corregida ---
# Apunta a ellipse.py en el mismo directorio
from utils.utils_ellipse_helpers import add_conic_points

EPS = 1e-18

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


LINE_TO_INTERSECTION: Dict[str, List[int]] = defaultdict(list)
for idx, lines in LINE_INTERSECTIONS.items():
    for line in lines:
        LINE_TO_INTERSECTION[line].append(idx)

# --- INICIO: CÓDIGO PARCHADO (reemplaza src.datatools) ---

def point_within_img(point: Optional[Tuple[float, float]],
                     img_size: Tuple[int, int] = (960, 540),
                     within_image: bool = True,
                     margin: float = 0.0) -> Optional[Tuple[float, float]]:
    """
    Comprueba si un punto está dentro de los límites de la imagen.
    (Re-implementado para eliminar la dependencia de src.datatools.geom)
    """
    if point is None:
        return None
    
    x, y = point
    # El script de Falaleev usa (W, H)
    W, H = img_size[0], img_size[1]
    
    if not within_image:
        # Si within_image es Falso, devolvemos el punto
        # sin importar dónde esté (nuestro caso de uso).
        return point
    
    # Comprobar límites (within_image == True)
    if (x >= 0 - margin) and (x <= W + margin) and \
       (y >= 0 - margin) and (y <= H + margin):
        return point
    else:
        return None # El punto está fuera de los límites

def find_closest_points(line_arr: np.ndarray, x: float, y: float, *args) \
        -> np.ndarray:
    """
    Encuentra los 2 puntos más cercanos en un array a un punto (x, y).
    (Re-implementado para eliminar la dependencia de src.datatools.line)
    """
    distances = np.sqrt(np.sum((line_arr - np.array([x, y]))**2, axis=1))
    # Retorna los 2 puntos más cercanos para el ajuste de línea recursivo
    closest_indices = np.argsort(distances)[:2]
    return line_arr[closest_indices]

# --- FIN: CÓDIGO PARCHADO ---


def intersection(line1_arr: np.ndarray, line2_arr: np.ndarray)\
        -> Optional[Tuple[float, float]]:
    """
    Find the intersection point of two lines.

    Each line is represented by a list of (x, y) tuples. The function fit each
    set of points with a line.

    Args:
        line1_arr (np.ndarray): First line: (N, 2).
        line2_arr (np.ndarray): Second line: (N, 2).

    Returns:
        Optional[Tuple[float, float]]: Intersection point. Note: the
            intersection point can be beyond the image.
    """

    x1, y1 = line1_arr[:, 0], line1_arr[:, 1]
    x2, y2 = line2_arr[:, 0], line2_arr[:, 1]
    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)
    is_x1_line = np.all(np.isclose(x1, x1_mean, atol=0.5))
    is_x2_line = np.all(np.isclose(x2, x2_mean, atol=0.5))
    point = None
    if is_x1_line:  # Deal with the case when the line1 is close to x=consts
        x = x1_mean
        if is_x2_line:
            return None
        b2, a2 = P.polyfit(x2, y2, 1)
        y = a2 * x + b2
    elif is_x2_line:  # Deal with the case when the line2 is close to x=consts
        x = x2_mean
        b1, a1 = P.polyfit(x1, y1, 1)
        y = a1 * x + b1
    else:  # Find lines intersecion as intersection of fitted lines
        b1, a1 = P.polyfit(x1, y1, 1)
        b2, a2 = P.polyfit(x2, y2, 1)
        x = (b2 - b1) / (a1 - a2 + EPS)  # Numerical stable division
        y = a1 * x + b1
    if line1_arr.shape[0] > 2 or line2_arr.shape[0] > 2:
        # Recursiver application of the function. It is applied in hope the
        # points closer to the intersection point can represent actual
        # intesection point better (the idea was approved for the dataset).
        line1_arr = find_closest_points(line1_arr, x, y, True)
        line2_arr = find_closest_points(line2_arr, x, y, True)
        point = intersection(line1_arr, line2_arr)
    else:
        point = (x, y)
    return point


def get_intersections(points: Dict[str, List[Tuple[float, float]]],
                      img_size: Tuple[int, int] = (960, 540),
                      within_image: bool = True,
                      margin: float = 0.0)\
        -> Tuple[Dict[int, Tuple[float, float] | None], List[int]]:
    res: Dict[int, Tuple[float, float] | None] = {}
    for i, pair in LINE_INTERSECTIONS.items():
        res[i] = None
        if pair[0] in points and pair[1] in points:
            if len(points[pair[0]]) > 1 and len(points[pair[1]]) > 1:
                # Find intersections and keep only the intersections withing
                # the given margins of the image.
                res[i] = point_within_img(intersection(
                    np.array(points[pair[0]]) * img_size,
                    np.array(points[pair[1]]) * img_size), img_size,
                    within_image, margin)
    
    # Esta función llama a ellipse.py y hace el resto de la magia
    res, mask = add_conic_points(points, res, img_size)

    # --- Corrección del Bug Lógico ---
    # Aplicar el filtrado final respetando el flag 'within_image'
    res = {i: point_within_img(res[i], img_size, within_image, margin)
           for i in res}
    return res, mask