import cv2
import yaml
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as f

from tqdm import tqdm
from PIL import Image

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l

from utils.utils_calib import FramebyFrameCalib, pan_tilt_roll_to_orientation
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, complete_keypoints, coords_to_dict

import os

DEBUG_SAVE_LIMIT = 50
debug_frame_counter = 0
DEBUG_DIR = "debug_heatmaps"
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR)


def save_heatmap_visualization(tensor_hm, prefix, frame_idx):
    """
    Toma el tensor de heatmaps (1, C, H, W), lo aplasta a (H, W)
    y lo guarda como imagen coloreada.
    """
    # 1. Tomamos el máximo valor a través de los canales (dim 1).
    # Esto combina todos los keypoints en una sola "capa".
    # tensor_hm[0] quita la dimensión del batch.
    hm_max = torch.max(tensor_hm[0], dim=0)[0] 
    
    # 2. Pasar a CPU y Numpy
    hm_np = hm_max.cpu().numpy()
    
    # 3. Normalizar a 0-255 para guardar como imagen
    # Si la red no detecta nada, el max y min serán bajos, así que normalizamos "min-max"
    hm_norm = cv2.normalize(hm_np, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 4. Aplicar mapa de color (Azul = frío/nada, Rojo = caliente/detectado)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    
    # 5. Guardar
    filename = f"{DEBUG_DIR}/{prefix}_{frame_idx:04d}.png"
    cv2.imwrite(filename, hm_color)


lines_coords = [[[0., 54.16, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
                [[16.5, 13.84, 0.], [0., 13.84, 0.]],
                [[88.5, 54.16, 0.], [105., 54.16, 0.]],
                [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
                [[88.5, 13.84, 0.], [105., 13.84, 0.]],
                [[0., 37.66, -2.44], [0., 30.34, -2.44]],
                [[0., 37.66, 0.], [0., 37.66, -2.44]],
                [[0., 30.34, 0.], [0., 30.34, -2.44]],
                [[105., 37.66, -2.44], [105., 30.34, -2.44]],
                [[105., 30.34, 0.], [105., 30.34, -2.44]],
                [[105., 37.66, 0.], [105., 37.66, -2.44]],
                [[52.5, 0., 0.], [52.5, 68, 0.]],
                [[0., 68., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [0., 68., 0.]],
                [[105., 0., 0.], [105., 68., 0.]],
                [[0., 0., 0.], [105., 0., 0.]],
                [[0., 43.16, 0.], [5.5, 43.16, 0.]],
                [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
                [[5.5, 24.84, 0.], [0., 24.84, 0.]],
                [[99.5, 43.16, 0.], [105., 43.16, 0.]],
                [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
                [[99.5, 24.84, 0.], [105., 24.84, 0.]]]


def projection_from_cam_params(final_params_dict):
    cam_params = final_params_dict["cam_params"]
    x_focal_length = cam_params['x_focal_length']
    y_focal_length = cam_params['y_focal_length']
    principal_point = np.array(cam_params['principal_point'])
    position_meters = np.array(cam_params['position_meters'])
    rotation = np.array(cam_params['rotation_matrix'])

    It = np.eye(4)[:-1]
    It[:, -1] = -position_meters
    Q = np.array([[x_focal_length, 0, principal_point[0]],
                  [0, y_focal_length, principal_point[1]],
                  [0, 0, 1]])
    P = Q @ (rotation @ It)

    return P


def inference(cam, frame, model, model_l, kp_threshold, line_threshold, pnl_refine):
    # --- Pre-procesamiento ---
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = f.to_tensor(frame).float().unsqueeze(0)
    frame = frame if frame.size()[-1] == 960 else transform2(frame)
    frame = frame.to(device)
    b, c, h, w = frame.size()

    global debug_frame_counter

    with torch.no_grad():
        heatmaps = model(frame)
        heatmaps_l = model_l(frame)

        # if debug_frame_counter < DEBUG_SAVE_LIMIT:
        #     # Guardamos heatmap de puntos (KP)
        #     save_heatmap_visualization(heatmaps, "KP", debug_frame_counter)
        #     # Guardamos heatmap de líneas (Lines)
        #     save_heatmap_visualization(heatmaps_l, "LINE", debug_frame_counter)
            
        #     print(f"[DEBUG] Guardado heatmap frame {debug_frame_counter}")
        #     debug_frame_counter += 1

    # 1. Obtener coordenadas crudas
    kp_coords_raw = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
    line_coords_raw = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
    
    # 2. Convertir a diccionarios PERO sin filtrar (threshold=0.0)
    kp_dict_all = coords_to_dict(kp_coords_raw, threshold=0.0)[0]
    lines_dict_all = coords_to_dict(line_coords_raw, threshold=0.0)[0]
    
    kp_dict_filtered = {}
    lines_dict_filtered = {}
    kp_dict_debug = {}

    # Filtrar Puntos (Red 1)
    if kp_dict_all:
        for cls_id, coords in kp_dict_all.items():
            score = coords.get('score', 0.0)
            kp_dict_debug[cls_id] = coords # Guardamos todo para debug visual
            
            # Aplicamos el threshold normal
            if score > kp_threshold:
                kp_dict_filtered[cls_id] = coords

    # Filtrar Líneas (Red 2)
    if lines_dict_all:
        for cls_id, coords in lines_dict_all.items():
            score = coords.get('score', 0.0)
            
            # ID 6 suele ser el Círculo Central en SoccerNet
            is_center_circle = (cls_id == 6) 
            
            # Umbral dinámico: 
            # Si es el círculo, aceptamos casi cualquier cosa (0.05). Si no, somos estrictos.
            threshold_to_use = 0.05 if is_center_circle else line_threshold
            
            if score > threshold_to_use:
                lines_dict_filtered[cls_id] = coords
                if is_center_circle:
                    print(f"[DEBUG] Círculo Central detectado con score: {score:.3f}")


    # 3. Completar y Normalizar para PnLCalib
    kp_dict_final, lines_dict_final = complete_keypoints(
        kp_dict_filtered, lines_dict_filtered, w=w, h=h, normalize=True
    )

    cam.update(kp_dict_final, lines_dict_final)
    final_params_dict = cam.heuristic_voting(refine_lines=pnl_refine, th=25.0)

    return final_params_dict, kp_dict_debug


def project(frame, P):

    for line in lines_coords:
        w1 = line[0]
        w2 = line[1]
        i1 = P @ np.array([w1[0]-105/2, w1[1]-68/2, w1[2], 1])
        i2 = P @ np.array([w2[0]-105/2, w2[1]-68/2, w2[2], 1])
        i1 /= i1[-1]
        i2 /= i2[-1]
        frame = cv2.line(frame, (int(i1[0]), int(i1[1])), (int(i2[0]), int(i2[1])), (255, 0, 0), 3)

    r = 9.15
    pts1, pts2, pts3 = [], [], []
    base_pos = np.array([11-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(37, 143, 50):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts1.append([ipos[0], ipos[1]])

    base_pos = np.array([94-105/2, 68/2-68/2, 0., 0.])
    for ang in np.linspace(217, 323, 200):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts2.append([ipos[0], ipos[1]])

    base_pos = np.array([0, 0, 0., 0.])
    for ang in np.linspace(0, 360, 500):
        ang = np.deg2rad(ang)
        pos = base_pos + np.array([r*np.sin(ang), r*np.cos(ang), 0., 1.])
        ipos = P @ pos
        ipos /= ipos[-1]
        pts3.append([ipos[0], ipos[1]])

    XEllipse1 = np.array(pts1, np.int32)
    XEllipse2 = np.array(pts2, np.int32)
    XEllipse3 = np.array(pts3, np.int32)
    frame = cv2.polylines(frame, [XEllipse1], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [XEllipse2], False, (255, 0, 0), 3)
    frame = cv2.polylines(frame, [XEllipse3], False, (255, 0, 0), 3)

    return frame


def process_input(input_path, input_type, model_kp, model_line, kp_threshold, line_threshold, pnl_refine,
                  save_path, display):

    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    cam = FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)

    # La red siempre trabaja internamente a 960x540. 
    # Los puntos que devuelve ya están en esa escala.
    NET_INPUT_W = 960
    NET_INPUT_H = 540
    
    scale_x = frame_width / NET_INPUT_W
    scale_y = frame_height / NET_INPUT_H

    last_P = None 
    missed_frames = 0

    if input_type == 'video':
        if save_path != "":
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        pbar = tqdm(total=total_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            final_params_dict, raw_kps = inference(
                cam, frame, model_kp, model_line, kp_threshold, line_threshold, pnl_refine
            )
            
            if raw_kps:
                for pid, coords in raw_kps.items():
                    score = coords.get('score', 1.0)
                    if score > 0.1: 
                        try:
                            x = int(coords['x'] * scale_x)
                            y = int(coords['y'] * scale_y)
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        except: pass

            # Lógica de Memoria
            current_P = None
            if final_params_dict is not None:
                current_P = projection_from_cam_params(final_params_dict)
                last_P = current_P
                missed_frames = 0
            elif last_P is not None and missed_frames < 5:
                current_P = last_P
                missed_frames += 1

            if current_P is not None:
                projected_frame = project(frame, current_P)
            else:
                projected_frame = frame
                cv2.putText(projected_frame, "CALIB FAIL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
            if save_path != "": out.write(projected_frame)
            if display:
                cv2.imshow('Projected', projected_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            pbar.update(1)

        cap.release()
        if save_path != "": out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process video or image and plot lines on each frame.")
    parser.add_argument("--weights_kp", type=str, help="Path to the model for keypoint inference.")
    parser.add_argument("--weights_line", type=str, help="Path to the model for line projection.")
    parser.add_argument("--kp_threshold", type=float, default=0.3434, help="Threshold for keypoint detection.")
    parser.add_argument("--line_threshold", type=float, default=0.7867, help="Threshold for line detection.")
    parser.add_argument("--pnl_refine", action="store_true", help="Enable PnL refinement module.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CPU or CUDA device index")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input video or image file.")
    parser.add_argument("--input_type", type=str, choices=['video', 'image'], required=True,
                        help="Type of input: 'video' or 'image'.")
    parser.add_argument("--save_path", type=str, default="", help="Path to save the processed video.")
    parser.add_argument("--display", action="store_true", help="Enable real-time display.")
    args = parser.parse_args()


    input_path = args.input_path
    input_type = args.input_type
    model_kp = args.weights_kp
    model_line = args.weights_line
    pnl_refine = args.pnl_refine
    save_path = args.save_path
    device = args.device
    display = args.display and input_type == 'video'
    kp_threshold = args.kp_threshold
    line_threshold = args.line_threshold

    cfg = yaml.safe_load(open("config/hrnetv2_w48.yaml", 'r'))
    cfg_l = yaml.safe_load(open("config/hrnetv2_w48_l.yaml", 'r'))

    loaded_state = torch.load(args.weights_kp, map_location=device)
    model = get_cls_net(cfg)
    model.load_state_dict(loaded_state)
    model.to(device)
    model.eval()

    loaded_state_l = torch.load(args.weights_line, map_location=device)
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(loaded_state_l)
    model_l.to(device)
    model_l.eval()

    transform2 = T.Resize((540, 960))

    process_input(input_path, input_type, model, model_l, kp_threshold, line_threshold, pnl_refine,
                  save_path, display)