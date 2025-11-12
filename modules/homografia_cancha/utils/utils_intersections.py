from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
from numpy.polynomial import polynomial as P

from utils.utils_ellipse_helpers import add_conic_points

EPS = 1e-18

# Define qué keypoints (por ID) se calculan a partir de la 
# intersección de qué dos líneas (por nombre).
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
    Comprueba si un punto está dentro de los límites de la imagen.
    """
    if point is None:
        return None
    
    x, y = point
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

def find_closest_points(line_arr: np.ndarray, x: float, y: float) \
        -> np.ndarray:
    """
    Encuentra los 2 puntos más cercanos en un array a un punto (x, y).
    (Re-implementado para eliminar la dependencia de src.datatools.line)
    """
    distances = np.sqrt(np.sum((line_arr - np.array([x, y]))**2, axis=1))
    # Retorna los 2 puntos más cercanos para el ajuste de línea recursivo
    closest_indices = np.argsort(distances)[:2]
    return line_arr[closest_indices]



def intersection(line1_arr: np.ndarray, line2_arr: np.ndarray)\
        -> Optional[Tuple[float, float]]:
    """
    Encuentra el punto de intersección de dos líneas.

    Cada línea es representada por una lista de tuplas (x, y). 
    La función ajusta cada conjunto de puntos a una recta.

    Args:
        line1_arr (np.ndarray): Primera línea: (N, 2).
        line2_arr (np.ndarray): Segunda línea: (N, 2).

    Returns:
        Optional[Tuple[float, float]]: Punto de intersección.
            Nota: el punto puede estar fuera de la imagen.
    """

    x1, y1 = line1_arr[:, 0], line1_arr[:, 1]
    x2, y2 = line2_arr[:, 0], line2_arr[:, 1]
    x1_mean = np.mean(x1)
    x2_mean = np.mean(x2)
    
    # Comprobar si las líneas son verticales
    is_x1_line = np.all(np.isclose(x1, x1_mean, atol=0.5))
    is_x2_line = np.all(np.isclose(x2, x2_mean, atol=0.5))
    point = None
    
    if is_x1_line:  # Caso: línea 1 es vertical (x=constante)
        x = x1_mean
        if is_x2_line:
            return None # Dos líneas verticaless paralelas no se cruzan
        b2, a2 = P.polyfit(x2, y2, 1)
        y = a2 * x + b2
    elif is_x2_line:  # Caso: línea 2 es vertical (x=constante)
        x = x2_mean
        b1, a1 = P.polyfit(x1, y1, 1)
        y = a1 * x + b1
    else:  # Caso estándar: ambas líneas tienen pendiente
        b1, a1 = P.polyfit(x1, y1, 1)
        b2, a2 = P.polyfit(x2, y2, 1)
        x = (b2 - b1) / (a1 - a2 + EPS)  # División numéricamente estable
        y = a1 * x + b1
        
    if line1_arr.shape[0] > 2 or line2_arr.shape[0] > 2:
        line1_arr = find_closest_points(line1_arr, x, y)
        line2_arr = find_closest_points(line2_arr, x, y)
        point = intersection(line1_arr, line2_arr)
    else:
        # Caso base de la recursión (solo 2 puntos por línea)
        point = (x, y)
        
    return point


def get_intersections(points: Dict[str, List[Tuple[float, float]]],
                      img_size: Tuple[int, int] = (960, 540),
                      within_image: bool = True,
                      margin: float = 0.0)\
        -> Tuple[Dict[int, Tuple[float, float] | None], List[int]]:
    """
    Función principal para obtener todos los keypoints (líneas y cónicas).
    
    Args:
        points: Diccionario de anotaciones (ej. {"Side line top": [(x,y),...]})
        img_size: Tamaño de la imagen en píxeles (W, H).
        within_image: Flag para filtrar puntos fuera de la imagen.
        margin: Margen adicional para el filtrado.

    Returns:
        Un diccionario de keypoints {ID: (x, y)} y una máscara (list[int]).
    """
    res: Dict[int, Tuple[float, float] | None] = {}
    
    # --- Parte 1: Intersecciones de Líneas (Keypoints 0-29) ---
    for i, pair in LINE_INTERSECTIONS.items():
        res[i] = None
        # Comprobar si tenemos anotaciones para ambas líneas
        if pair[0] in points and pair[1] in points:
            if len(points[pair[0]]) > 1 and len(points[pair[1]]) > 1:
                # Calcular intersección
                res[i] = intersection(
                    np.array(points[pair[0]]) * img_size,
                    np.array(points[pair[1]]) * img_size)

    # --- Parte 2: Puntos de Cónicas (Keypoints 30-56) ---
    res, mask = add_conic_points(points, res, img_size)

    res = {i: point_within_img(res[i], img_size, within_image, margin)
           for i in res}
           
    return res, mask