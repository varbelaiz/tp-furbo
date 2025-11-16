# Tracking Dataset

This directory holds the data used to train and evaluate the detection, ball
detection and tracking baseline models. Nothing in it is tracked in git.

## Layout

```
data/tracking/
│
├── raw/
│   ├── train/        raw clips from the dataset (SNMOT)
│   ├── test/         test clips
│   └── test.zip      original download (optional)
│
├── YOLO_ball/
│   ├── images/       frames used to train the ball detector
│   ├── labels/       YOLO labels for the 'ball' class
│   └── ball.yaml     dataset file used for training
│
└── YOLO_baseline/
    ├── images/       annotated frames
    ├── labels/       YOLO labels for players and referees
    └── tracker.yaml  dataset file used to train the baseline
```

## What each folder holds

### `raw/`
The original clips from the **SoccerNet Tracking (SNMOT)** dataset. They are the
source for the extracted frames and labels, and are what
`src/evaluation/tracking/eval.py` reads to score the detectors against the
ground truth in each clip's `gt/gt.txt`.

### `YOLO_ball/`
Dataset for the dedicated ball detector:

- `images/`: extracted frames
- `labels/`: YOLO bounding boxes for the ball
- `ball.yaml`: image and label paths plus the YOLO configuration

### `YOLO_baseline/`
Dataset for the general detector (players and referees):

- `images/`: annotated frames
- `labels/`: YOLO bounding boxes
- `tracker.yaml`: Ultralytics-compatible dataset file

The conversion from the raw SNMOT clips into these YOLO folders is done in
`src/train/tracking/transformation.ipynb` (players and referees) and
`src/train/tracking/ball/transformation.ipynb` (ball).
