#!/bin/bash
#
# Build the 29-keypoint dataset from the raw SoccerNet calibration data. Run it from the repository root:
#
#     bash src/data_prep/run_dataload.sh
#

# Stop the script on any error
set -e

echo "STARTING THE 29 KP PIPELINE..."


# 0. PATH CONFIGURATION

PROJECT_ROOT=$(pwd)

# Data locations (images and JSON annotations)
BASE_DATA_DIR="$PROJECT_ROOT/data/calibration"
SOURCE_ZIPS="$BASE_DATA_DIR"
DEST_IMAGES="$BASE_DATA_DIR/images"
DEST_ANNOTATIONS="$BASE_DATA_DIR/soccernet_calibration_annotations"


# 1. DATA PREPARATION
echo "---------------------------------------"
echo "1. Data preparation"
echo "---------------------------------------"

# 1.1 Download, only when needed
if [ ! -f "$SOURCE_ZIPS/train.zip" ]; then
    echo "Downloading data..."
    python -m src.data_prep.keypoints.datatools_29.downloader
else
    echo "Zips found in $SOURCE_ZIPS"
fi

# 1.2 Extraction and organisation, only when the folders are not there yet
if [ ! -d "$DEST_IMAGES/train" ] || [ ! -d "$DEST_ANNOTATIONS/train" ]; then
    echo "Extracting and organising the raw files..."

    # Clear the destinations so nothing gets mixed up
    rm -rf "$DEST_IMAGES" "$DEST_ANNOTATIONS"
    mkdir -p "$DEST_IMAGES" "$DEST_ANNOTATIONS"

    cd "$SOURCE_ZIPS"
    for split in train valid test; do
        echo "   -> Unzipping $split..."
        if [ ! -f "${split}.zip" ]; then echo "Missing ${split}.zip"; exit 1; fi

        unzip -q -o "${split}.zip" -d "temp_$split"

        # Flatten the nesting when the archive contains a top-level folder
        if [ -d "temp_$split/$split" ]; then
            mv "temp_$split/$split"/* "temp_$split/"
            rmdir "temp_$split/$split"
        fi

        # Split the JSON annotations off from the images
        mkdir -p "$DEST_ANNOTATIONS/$split"
        find "temp_$split" -name "*.json" -exec mv {} "$DEST_ANNOTATIONS/$split/" \;

        mv "temp_$split" "$DEST_IMAGES/$split"
        rm -rf "temp_$split"
    done
    cd "$PROJECT_ROOT"
else
    echo "Raw images and annotations are already in place."
fi

# Post-process a generated dataset: move the folders around and fix the YAML
finalize_dataset() {
    local OUTPUT_DIR=$1
    local KP_COUNT=$2

    echo "   Finalising the YOLO layout ($KP_COUNT points)..."

    # 1. Create the folder that holds the clean images
    mkdir -p "$OUTPUT_DIR/images"

    # 2. Copy the raw (unannotated) images used for training
    echo "      -> Copying raw images..."
    cp -rn "$DEST_IMAGES/train" "$OUTPUT_DIR/images/"
    cp -rn "$DEST_IMAGES/valid" "$OUTPUT_DIR/images/"
    cp -rn "$DEST_IMAGES/test" "$OUTPUT_DIR/images/"

    # 3. Rename 'yolo_labels' to 'labels' (the YOLO convention)
    if [ -d "$OUTPUT_DIR/yolo_labels" ]; then
        rm -rf "$OUTPUT_DIR/labels"
        mv "$OUTPUT_DIR/yolo_labels" "$OUTPUT_DIR/labels"
    fi

    # 4. Fix dataset.yaml (point the split paths at images/ instead of yolo_labels/)
    local YAML_FILE="$OUTPUT_DIR/dataset.yaml"
    if [ -f "$YAML_FILE" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' 's/yolo_labels/images/g' "$YAML_FILE"
        else
            sed -i 's/yolo_labels/images/g' "$YAML_FILE"
        fi
    fi
    echo "      Dataset with $KP_COUNT kp ready in: $OUTPUT_DIR"
}

# 2. GENERATE THE 29-POINT DATASET

echo "---------------------------------------"
echo "Generating the 29 KP dataset"
echo "---------------------------------------"
echo "Running process_images.py..."

python -m src.data_prep.keypoints.datatools_29.process_images

# Post-process the output (which defaults to unified_output)
finalize_dataset "$BASE_DATA_DIR/unified_output" "29"


# 4. DONE
echo "---------------------------------------"
echo "PIPELINE COMPLETE"
echo "---------------------------------------"
echo "The dataset is ready in:"
echo "   29 KP: $BASE_DATA_DIR/unified_output"
