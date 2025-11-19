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
from matplotlib.patches import Polygon

from model.cls_hrnet import get_cls_net
from model.cls_hrnet_l import get_cls_net as get_cls_net_l

from utils.utils_calib import FramebyFrameCalib, pan_tilt_roll_to_orientation
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool, get_keypoints_from_heatmap_batch_maxpool_l, \
    complete_keypoints, coords_to_dict


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


def project_2d(frame, H):
    
    W_WORLD, H_WORLD = 105, 68
    W_HALF, H_HALF = W_WORLD / 2, H_WORLD / 2
    img_h, img_w = frame.shape[:2]

    # --- 1. Proyección de Líneas ---
    for i, line in enumerate(lines_coords):
        w1, w2 = line[0], line[1]
        
        # 1. Coordenadas del mundo Centradas y Homogéneas
        i1_homo = H @ np.array([w1[0] - W_HALF, w1[1] - H_HALF, 1])
        i2_homo = H @ np.array([w2[0] - W_HALF, w2[1] - H_HALF, 1])

        # 🛑 DEBUG: Print the raw homogeneous coordinates
        # if i == 0:
            # print(f"\n--- HOMOGENEOUS DEBUG ---")
            # print(f"  i1_homo (Raw Px, Raw Py, Px Factor): {i1_homo}")
            # print(f"  i2_homo (Raw Px, Raw Py, Px Factor): {i2_homo}")
        
        # 2. Normalización Homogénea (Dividir por el tercer componente)
        try:
            # i1_homo[-1] is the perspective divisor (s).
            # If s is zero or near-zero, division fails.
            s1 = i1_homo[-1]
            s2 = i2_homo[-1]

            if abs(s1) < 1e-6 or abs(s2) < 1e-6:
                #  print(f"  [CRITICAL ERROR] Divisor is near zero. s1: {s1}, s2: {s2}")
                 continue

            # i1 es la coordenada de píxel [x', y']
            i1 = (i1_homo / i1_homo[-1])[:2]
            i2 = (i2_homo / i2_homo[-1])[:2]
        except FloatingPointError:
            continue
            
        # 3. Clipping Robusto y Conversión a Enteros
        
        # Usamos np.clip para forzar que los puntos (que están fuera) se dibujen
        # justo en el borde de la imagen, evitando que cv2.line falle.
        x1 = np.clip(i1[0], -200, img_w + 200)
        y1 = np.clip(i1[1], -200, img_h + 200)
        x2 = np.clip(i2[0], -200, img_w + 200)
        y2 = np.clip(i2[1], -200, img_h + 200)

        # 🛑 DEBUG: Print the final pixel coordinates
        # if i == 0:
            # print(f"  Final Px Coords (Clipped): ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")
        
        # Dibujar la línea (OpenCV maneja el clipping final si los puntos están en el margen)
        frame = cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 3) # Amarillo para que resalte

    # --- 2. Proyección de Círculos (Ajuste del arco) ---
    r = 9.15
    pts1, pts2, pts3 = [], [], []

    # Círculos y Elipses
    
    # Círculo de penalización Izquierdo
    center_pos_l = np.array([11. - W_HALF, H_HALF - H_HALF])
    for ang in np.linspace(37, 143, 50):
        ang = np.deg2rad(ang)
        wx = center_pos_l[0] + r * np.sin(ang)
        wy = center_pos_l[1] + r * np.cos(ang)
        ipos_homo = H @ np.array([wx, wy, 1])
        
        if ipos_homo[-1] != 0:
            ipos = (ipos_homo / ipos_homo[-1])[:2]
            # Clip y Añadir a la lista
            x = np.clip(ipos[0], -200, img_w + 200)
            y = np.clip(ipos[1], -200, img_h + 200)
            pts1.append([int(x), int(y)])

    # Círculo de penalización Derecho
    center_pos_r = np.array([94. - W_HALF, H_HALF - H_HALF])
    for ang in np.linspace(217, 323, 200):
        ang = np.deg2rad(ang)
        wx = center_pos_r[0] + r * np.sin(ang)
        wy = center_pos_r[1] + r * np.cos(ang)
        ipos_homo = H @ np.array([wx, wy, 1])
        
        if ipos_homo[-1] != 0:
            ipos = (ipos_homo / ipos_homo[-1])[:2]
            x = np.clip(ipos[0], -200, img_w + 200)
            y = np.clip(ipos[1], -200, img_h + 200)
            pts2.append([int(x), int(y)])

    # Círculo Central
    center_pos_c = np.array([0., 0.])
    for ang in np.linspace(0, 360, 500):
        ang = np.deg2rad(ang)
        wx = center_pos_c[0] + r * np.sin(ang)
        wy = center_pos_c[1] + r * np.cos(ang)
        ipos_homo = H @ np.array([wx, wy, 1])
        
        if ipos_homo[-1] != 0:
            ipos = (ipos_homo / ipos_homo[-1])[:2]
            x = np.clip(ipos[0], -200, img_w + 200)
            y = np.clip(ipos[1], -200, img_h + 200)
            pts3.append([int(x), int(y)])

    # Dibujar polilíneas (Solo si tienen suficientes puntos)
    if len(pts1) > 1:
        XEllipse1 = np.array(pts1, np.int32)
        frame = cv2.polylines(frame, [XEllipse1], False, (0, 255, 255), 3)
    if len(pts2) > 1:
        XEllipse2 = np.array(pts2, np.int32)
        frame = cv2.polylines(frame, [XEllipse2], False, (0, 255, 255), 3)
    if len(pts3) > 1:
        XEllipse3 = np.array(pts3, np.int32)
        frame = cv2.polylines(frame, [XEllipse3], True, (0, 255, 255), 3) # Círculo central se cierra

    return frame


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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    frame = f.to_tensor(frame).float().unsqueeze(0)
    _, _, h_original, w_original = frame.size()
    frame = frame if frame.size()[-1] == 960 else transform2(frame)
    frame = frame.to(device)
    b, c, h, w = frame.size()

    with torch.no_grad():
        heatmaps = model(frame)
        heatmaps_l = model_l(frame)

    kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:,:-1,:,:])
    line_coords = get_keypoints_from_heatmap_batch_maxpool_l(heatmaps_l[:,:-1,:,:])
    kp_dict = coords_to_dict(kp_coords, threshold=kp_threshold)
    lines_dict = coords_to_dict(line_coords, threshold=line_threshold)
    kp_dict, lines_dict = complete_keypoints(kp_dict[0], lines_dict[0], w=w, h=h, normalize=True)

    cam.update(kp_dict, lines_dict)

    # print("len(kp_dict):", len(kp_dict))


    # homography_result = cam.heuristic_voting(refine_lines=pnl_refine,th=1.5)

    homography_result = cam.heuristic_voting_ground(refine_lines=pnl_refine,th=15.0)

    return homography_result


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

    if input_type == 'video':
        cap = cv2.VideoCapture(input_path)
        if save_path != "":
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        pbar = tqdm(total=total_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            final_params_dict = inference(cam, frame, model, model_l, kp_threshold, line_threshold, pnl_refine)
            if final_params_dict is not None:
                    
                # P = projection_from_cam_params(final_params_dict)
                # projected_frame = project(frame, P)

                H = final_params_dict['homography']

                TX_CORRECTION = 600  # Mover la proyección 600 píxeles a la derecha
                TY_CORRECTION = 300  # Mover la proyección 300 píxeles hacia abajo

                T = np.array([
                    [1, 0, TX_CORRECTION],
                    [0, 1, TY_CORRECTION],
                    [0, 0, 1]
                ], dtype=np.float64)

                # 🛑 APLICAR LA CORRECCIÓN:
                H = T @ H

                # print("Estimated Homography:\n", H)

                projected_frame = project_2d(frame, H)
            else:
                projected_frame = frame
                
            if save_path != "":
                out.write(projected_frame)
    
            if display:
                cv2.imshow('Projected Frame', projected_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            pbar.update(1)

        cap.release()
        if save_path != "":
            out.release()
        cv2.destroyAllWindows()

    elif input_type == 'image':
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Unable to read the image {input_path}")
            return

        final_params_dict = inference(cam, frame, model, model_l, kp_threshold, line_threshold, pnl_refine)
        if final_params_dict is not None:
            P = projection_from_cam_params(final_params_dict)
            projected_frame = project(frame, P)
        else:
            projected_frame = frame

        if save_path != "":
            cv2.imwrite(save_path, projected_frame)
        else:
            plt.imshow(cv2.cvtColor(projected_frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()

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

    process_input(input_path, input_type, model_kp, model_line, kp_threshold, line_threshold, pnl_refine,
                  save_path, display)
