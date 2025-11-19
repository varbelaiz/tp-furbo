# Experiment: pitch calibration with HRNetV2

Dropped experiment. Kept as a record of an approach that was tried and ruled
out. The pipeline that shipped is on `main`.

## What this was

Camera calibration on SoccerNet using the HRNetV2 baseline from the SoccerNet
Calibration Challenge, rather than the YOLO pose approach that ended up on
`main`.

The idea was to predict the pitch geometry as **heatmaps** instead of as a fixed
set of keypoints, using two networks:

- **Network 1 (keypoints)**: 57 keypoint heatmaps plus a background channel,
  following Falaleev's calibration work.
- **Network 2 (lines)**: 23 pitch line heatmaps plus a background channel,
  following PnLCalib.

The predicted heatmaps were then turned into keypoints, and the camera
parameters recovered from them with `cv2.calibrateCamera`, giving a full camera
model rather than a plane-to-plane homography.

## Why it was dropped

It never produced a calibration stable enough to build on:

- The camera solve failed on a large fraction of frames. Whenever the recovered
  points were close to collinear, `cv2.calibrateCamera` either threw or returned
  NaN, and those frames had to be skipped (see the guards in
  `modules/pitch_calibration/utils/utils_calib.py`).
- Training the two HRNet heads was slow and expensive. It needed a GPU VM, a GCS
  bucket for the checkpoints, and precomputed heatmaps on disk just to keep the
  dataloader from starving the GPU.
- Even where it converged, the resulting projection was noisier frame to frame
  than the direct homography from detected keypoints.

The YOLO pose approach on `main` solves a smaller problem, a plane-to-plane
homography rather than a full camera model, and does it well enough for the
tactical minimap at a fraction of the cost.

## Layout

```
docs/assignment-brief.pdf     the coursework brief
radar_prototype/              tactical radar overlay, developed against mock data
├── main.py                   radar overlay prototype, driven by mock JSON
├── generate_mocks.py         generates the mock calibration/detections/actions
├── pitch_annotator.py        pitch drawing helpers for the radar
└── *_MOCK.json               mock inputs for the prototype
modules/pitch_calibration/    the HRNetV2 calibration work
├── model/                    HRNetV2 architecture and dataloaders
├── model_config/             HRNetV2 architecture configs
├── train_config/             training configs for both networks
├── scripts/                  dataset download, heatmap generation, training, checks
├── utils/                    geometry, heatmaps, camera calibration
├── sn_calibration/           SoccerNet calibration baseline, vendored upstream
├── inference.py              heatmaps -> keypoints -> camera parameters
├── crop_video.py             trims a clip for quick tests
└── README.md                 how to run the pipeline
```

The calibration module is the point of this branch; `radar_prototype/` is a
small overlay demo that was built against mock JSON while the calibration was
still unreliable, so the two are kept apart.

`modules/pitch_calibration/sn_calibration/` is vendored from the SoccerNet
calibration development kit and is left exactly as upstream published it.

## Credits

- [SoccerNet calibration development kit](https://github.com/SoccerNet/sn-calibration)
- HRNetV2, from the SoccerNet Calibration Challenge baseline
- [PnLCalib](https://github.com/mguti97/PnLCalib) for the line heatmap formulation
