"""
Unified SoccerNet processing pipeline for 57 KEYPOINTS.
Generates the dataset in 'unified_output_57'.

Run it from the repository root as a module:
    python -m src.data_prep.keypoints.datatools_57.process_images_57
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import tqdm

# Directory helpers and the pitch detector are shared with the 29-point pipeline
from src.data_prep.keypoints.constants import calibration_dir
from src.data_prep.keypoints.datatools_29.get_pitch_object import PitchDetector

# Geometry helpers ported from the Sportlight calibration repository
from src.data_prep.keypoints.datatools_57.reader import read_json, decode_annot
from src.data_prep.keypoints.datatools_57.intersections import get_intersections
from src.data_prep.keypoints.datatools_57.ellipse_utils import INTERSECTON_TO_PITCH_POINTS

# Total number of keypoints in the SoccerNet v3 standard layout
TOTAL_KEYPOINTS = 57


def create_ultralytics_annotation_57(
    pitch_data: Dict,
    keypoints: Dict[int, Tuple[float, float]],
    image_shape: Tuple[int, int]
) -> str:
    """
    Build the YOLO pose annotation line for the 57-point layout.
    Pixel coordinates are normalised to [0, 1].
    """
    img_w, img_h = image_shape

    # Class 0 is 'pitch', followed by the pitch bounding box
    annotation_parts = [
        "0",  # Class index
        f"{pitch_data['center_x']:.6f}",
        f"{pitch_data['center_y']:.6f}",
        f"{pitch_data['width']:.6f}",
        f"{pitch_data['height']:.6f}"
    ]

    # Always iterate 0..56 so the vector keeps a fixed ordering
    for i in range(TOTAL_KEYPOINTS):
        # Check whether point 'i' was computed successfully
        if i in keypoints and keypoints[i] is not None:
            px, py = keypoints[i]

            # Normalisation for YOLO (x / width, y / height)
            nx = px / img_w
            ny = py / img_h

            # Check visibility (inside the image).
            # The geometry sometimes projects points outside the frame; for
            # training only the visible ones are flagged as such.
            if 0 <= nx <= 1 and 0 <= ny <= 1:
                # Format: x y visibility (2 = visible)
                annotation_parts.extend([f"{nx:.6f}", f"{ny:.6f}", "2"])
            else:
                # Computed but off-frame; 0 is the safe visibility flag here
                annotation_parts.extend(["0.0", "0.0", "0"])
        else:
            # Point missing or not found
            annotation_parts.extend(["0.0", "0.0", "0"])

    return " ".join(annotation_parts)


def create_visualization_57(
    image_path: str,
    pitch_data: Dict,
    keypoints: Dict[int, Tuple[float, float]],
    output_path: str
) -> None:
    """Draw the 57 points over the image for visual validation."""
    image = cv2.imread(image_path)
    if image is None:
        return

    height, width = image.shape[:2]

    # Draw the pitch bounding box
    x_min = int(pitch_data['x_min'] * width)
    y_min = int(pitch_data['y_min'] * height)
    x_max = int(pitch_data['x_max'] * width)
    y_max = int(pitch_data['y_max'] * height)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(image, "Pitch", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw the keypoints
    for idx, coords in keypoints.items():
        if coords is not None:
            pt = (int(coords[0]), int(coords[1]))

            # Only draw the points that fall inside the image
            if 0 <= pt[0] < width and 0 <= pt[1] < height:
                # Red dot
                cv2.circle(image, pt, 4, (0, 0, 255), -1)
                # Small white index
                cv2.putText(image, str(idx), (pt[0] + 5, pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cv2.imwrite(output_path, image)


def process_dataset_57() -> None:
    """Main pipeline that generates the 57-point dataset."""
    print("Starting the 57-keypoint processing pipeline...")

    pitch_detector = PitchDetector()

    # Written to 'unified_output_57' so the 29-point dataset stays untouched
    base_output_dir = calibration_dir / 'unified_output_57'
    json_dir = base_output_dir / 'annotations_json'
    images_dir = base_output_dir / 'processed_images'
    yolo_labels_dir = base_output_dir / 'yolo_labels'

    # Create the output folders
    for dir_path in [base_output_dir, json_dir, images_dir, yolo_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    splits = ['train', 'test', 'valid']

    for dataset_type in splits:
        if not (calibration_dir / 'images' / dataset_type).exists():
            continue

        print(f"\nProcessing split: {dataset_type}")
        annotations_path = calibration_dir / 'soccernet_calibration_annotations' / dataset_type
        images_path = calibration_dir / 'images' / dataset_type

        # One subfolder per split
        for output_dir in [json_dir, images_dir, yolo_labels_dir]:
            (output_dir / dataset_type).mkdir(parents=True, exist_ok=True)

        json_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]

        for json_file in tqdm.tqdm(json_files, desc=f"Split {dataset_type}"):
            json_path = annotations_path / json_file
            image_path = images_path / json_file.replace('.json', '.jpg')
            base_name = json_file.replace('.json', '')

            if not image_path.exists():
                continue

            try:
                # 1. Read the image to get its dimensions (needed by the geometry)
                img_temp = cv2.imread(str(image_path))
                if img_temp is None:
                    continue
                h, w = img_temp.shape[:2]
                img_size = (w, h)

                # 2. Read the JSON and extract the annotated lines
                annot_data = read_json(str(json_path))
                lines_dict = decode_annot(annot_data)  # JSON -> dictionary of lines

                # 3. Compute the 57 points with the Sportlight geometry.
                # keypoints_pix is a dict {0: (x, y), ...} in absolute pixel coordinates.
                keypoints_pix, _ = get_intersections(lines_dict, img_size=img_size)

                # 4. Detect the pitch bounding box
                pitch_result = pitch_detector.detect_pitch_from_image(str(image_path))

                if not pitch_result:
                    # Fall back to the whole image when the pitch is not detected
                    pitch_data = {'center_x': 0.5, 'center_y': 0.5, 'width': 1.0, 'height': 1.0,
                                  'x_min': 0, 'y_min': 0, 'x_max': 1, 'y_max': 1}
                else:
                    pitch_data = pitch_result['pitch_detection']

                # 5. Write the outputs

                # A) Unified JSON. Tuples become lists so the dict is serialisable.
                serializable_kpts = {k: list(v) if v is not None else None for k, v in keypoints_pix.items()}

                unified_annotation = {
                    'image_info': {'file_name': image_path.name, 'width': w, 'height': h},
                    'pitch_object': pitch_data,
                    'keypoints': serializable_kpts,
                    'format': 'SoccerNet_57_points'
                }

                with open(json_dir / dataset_type / f"{base_name}.json", 'w', encoding='utf-8') as f:
                    json.dump(unified_annotation, f, indent=2)

                # B) YOLO TXT label
                yolo_annotation = create_ultralytics_annotation_57(
                    pitch_data, keypoints_pix, img_size
                )
                with open(yolo_labels_dir / dataset_type / f"{base_name}.txt", 'w', encoding='utf-8') as f:
                    f.write(yolo_annotation + '\n')

                # C) JPG visualization
                create_visualization_57(
                    str(image_path), pitch_data, keypoints_pix,
                    str(images_dir / dataset_type / f"{base_name}_annotated.jpg")
                )

            except Exception as e:
                print(f"Error on {json_file}: {e}")
                continue

    # Write the final YAML into the new folder
    create_dataset_yaml_57(base_output_dir)
    print(f"\nDone. Dataset written to: {base_output_dir}")


def create_dataset_yaml_57(base_output_dir: Path) -> None:
    """Generate the dataset.yaml specific to the 57-point layout."""

    # Build the keypoint name list automatically
    names_yaml = ""
    for idx in range(TOTAL_KEYPOINTS):
        # INTERSECTON_TO_PITCH_POINTS holds the canonical point names
        name = INTERSECTON_TO_PITCH_POINTS.get(idx, f"kp_{idx}")
        names_yaml += f"  {idx}: {name}\n"

    yaml_content = f"""# SoccerNet 57-Keypoints Dataset Configuration

path: {base_output_dir.absolute()}
train: yolo_labels/train
val: yolo_labels/valid
test: yolo_labels/test

# Keypoints setup
kpt_shape: [{TOTAL_KEYPOINTS}, 3]  # 57 points, each (x, y, visibility)
flip_idx: [] # Optional: define the symmetry here to use flip augmentation

# Classes
names:
  0: pitch

# Keypoint Names (Mapping 0-56)
keypoint_names:
{names_yaml}
"""

    yaml_path = base_output_dir / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


if __name__ == "__main__":
    process_dataset_57()
