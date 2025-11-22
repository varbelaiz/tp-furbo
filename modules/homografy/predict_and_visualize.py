from ultralytics import YOLO
import cv2
import numpy as np

# --- CONFIGURACIÓN ---
MODEL_PATH = 'best.pt' 
VIDEO_PATH = '121364_0.mp4'  
OUTPUT_PATH = 'resultado_cancha.mp4'
CONF_THRESHOLD = 0.5 

# --- DEFINICIÓN DEL ESQUELETO (QUÉ PUNTOS CONECTAN CON CUÁLES) ---
SKELETON_CONNECTIONS = [
    # Perímetro
    (0, 26), (0, 5), (5, 31), (26, 31), 
    # Línea central
    (14, 17), 
    # Área Grande Izquierda
    (1, 9), (9, 12), (12, 4), (1, 4),
    # Área Chica Izquierda
    (2, 6), (6, 7), (7, 3), (2, 3),
    # Área Grande Derecha
    (27, 19), (19, 22), (22, 30), (27, 30),
    # Área Chica Derecha
    (28, 24), (24, 25), (25, 29), (28, 29)
]

def main():
    # 1. Cargar Modelo
    print(f"Cargando modelo desde {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print("No se encuentra el modelo. ¿Ya terminaste de entrenar? Usa 'yolov8x-pose.pt' para probar si no.")
        return

    # 2. Abrir Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error al abrir el video: {VIDEO_PATH}")
        return

    # Configurar guardado de video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("Procesando video... (Presiona 'q' para salir antes)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Inferencia
        results = model.predict(frame, conf=0.3, verbose=False)
        
        # Obtener keypoints (xy coords)
        # shape: (1, 32, 2) -> Batch 1, 32 puntos, x/y
        keypoints = results[0].keypoints.xy.cpu().numpy()[0]
        confs = results[0].keypoints.conf.cpu().numpy()[0] if results[0].keypoints.conf is not None else [1]*32

        # 4. Dibujo Personalizado (OpenCV)
        # Dibujamos las líneas primero (fondo)
        for p1_idx, p2_idx in SKELETON_CONNECTIONS:
            if p1_idx < len(keypoints) and p2_idx < len(keypoints):
                # Verificar confianza de ambos puntos
                if confs[p1_idx] > CONF_THRESHOLD and confs[p2_idx] > CONF_THRESHOLD:
                    pt1 = (int(keypoints[p1_idx][0]), int(keypoints[p1_idx][1]))
                    pt2 = (int(keypoints[p2_idx][0]), int(keypoints[p2_idx][1]))
                    
                    # Dibujar línea verde semi-transparente o solida
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

        # Dibujamos los puntos encima
        for i, (x, y) in enumerate(keypoints):
            if confs[i] > CONF_THRESHOLD:
                # Puntos en Cian con borde azul
                cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 0), -1)
                cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), 1) # Borde
                
                # Número del índice para debug
                cv2.putText(frame, str(i), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Guardar y mostrar
        out.write(frame)
        
        # Redimensionar para mostrar en pantalla si es 4k
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('Prediccion Cancha YOLOv8', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video guardado en: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()