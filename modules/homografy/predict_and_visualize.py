import cv2
import supervision as sv
from ultralytics import YOLO

# --- 1. DEFINE SKELETON EXPLICITLY ---
# This matches your 32-point model structure
SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (6, 7),
    (9, 10), (10, 11), (11, 12), (13, 14), (14, 15),
    (15, 16), (16, 17), (18, 19), (19, 20), (20, 21),
    (22, 23), (24, 25), (25, 26), (26, 27), (27, 28),
    (28, 29), (0, 13), (1, 9), (2, 6), (3, 7),
    (4, 12), (5, 17), (13, 24), (18, 25), (22, 26),
    (23, 27), (20, 28), (16, 29)
]

def main():
    # 2. Load Model
    # Ensure this is the model trained with the NEW 32-point dataset
    model = YOLO('runs/pose/train/weights/best.pt')

    # 3. Setup Annotators with the Skeleton
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        radius=4
    )

    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.from_hex('#FF1493'),
        thickness=2,
        edges=SKELETON  # <--- PASS THE PYTHON LIST HERE
    )

    # 4. Run Inference
    image_path = "test/images/some_image.jpg" # Change this
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read image.")
        return

    results = model(frame)[0]
    
    # 5. Visualize
    key_points = sv.KeyPoints.from_ultralytics(results)

    annotated_frame = frame.copy()
    annotated_frame = edge_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points
    )
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points
    )

    # 6. Save or Plot
    sv.plot_image(annotated_frame)
    # cv2.imwrite("output_vis.jpg", annotated_frame)

if __name__ == "__main__":
    main()