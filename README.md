# Experiment: 32-point pitch homography with YOLO pose

Dropped experiment. Kept as a record of an approach that was superseded. The
pipeline that shipped is on `main`.

## What this was

The first attempt at estimating the pitch homography with a YOLO pose model,
built on Roboflow's `sports` library and its football field detection dataset.

The pitch was represented as a **32-keypoint** layout, the model was
`yolov8x-pose`, and the homography was solved from whichever of those 32 points
the model found in a frame.

```
modules/yolo_pose/
├── preprocess_dataset.py   SoccerNet line annotations -> 32-point YOLO pose labels
├── soccernet.yaml          dataset definition, kpt_shape [32, 3]
├── train.py                yolov8x-pose training, 100 epochs, logged to W&B
├── roboflow_demo.ipynb     the Roboflow reference notebook this started from
└── sports/                 Roboflow sports library, vendored upstream
```

## Why it was superseded

Nothing here was broken; it was replaced by a better version of the same idea.

- The 32-point layout was inherited from the Roboflow demo rather than derived
  from the SoccerNet annotations, so it did not line up cleanly with what the
  calibration data actually labels.
- `main` reuses the same core idea, a YOLO pose model plus a RANSAC homography,
  but with keypoint layouts derived directly from the SoccerNet line annotations:
  29 points, and later 57. That made the labels denser and better grounded.
- Vendoring `sports` into the repository turned out to be unnecessary. On `main`
  it is a normal dependency pinned in `pyproject.toml`.

The `flip_idx` in `soccernet.yaml` is marked as approximate, which is a fair
summary of the whole layout: usable, but never nailed down.

## Credits

- [roboflow/sports](https://github.com/roboflow/sports), vendored under
  `modules/yolo_pose/sports/`
- SoccerNet calibration data
