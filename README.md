# FULBO - SoccerNet Analysis Pipeline

FULBO is a computer vision project that analyses football matches. It detects
players, referees and the ball, estimates the pitch keypoints, tracks people
across a clip, assigns each player to a team, and projects everything onto a
tactical minimap.

The models are YOLO detectors and YOLO pose estimators trained on
[SoccerNet](https://www.soccer-net.org/) data.

## Pipeline

```
frame
  |
  +-- baseline detector (players, referees)  --> ByteTrack --> track ids
  |                                                              |
  +-- ball detector (sliced inference)                           |
  |                                                              v
  +-- keypoint model (29 or 57 points) --> homography     DINOv2 embeddings
                                              |            + UMAP + KMeans
                                              |                  |
                                              v                  v
                                     pitch coordinates      team assignment
                                              |                  |
                                              +--------+---------+
                                                       v
                                            annotated frame + minimap
```

The two passes are implemented in `main.py`: pass 1 runs detection, tracking and
embedding collection; pass 2 fits the team clusters, computes the homography per
frame and renders the output.

## Repository layout

| Path | Contents |
|------|----------|
| `main.py` | Entry point that runs the full pipeline over a video, an image sequence or a single image. |
| `src/data_prep/` | Turns raw SoccerNet calibration annotations into YOLO pose datasets (29 and 57 keypoints). |
| `src/train/` | Training scripts for the keypoint models and the detection models. |
| `src/inference/` | Keypoint detection, homography, tracking and minimap rendering. |
| `src/evaluation/` | Evaluation scripts and stored metrics for keypoints and tracking. |
| `data/` | Where the datasets live once downloaded and prepared. Only the READMEs are tracked. |
| `models/` | Training curves and metrics for every run that was kept. |
| `outputs/` | Generated videos and figures. |
| `pyproject.toml`, `uv.lock` | Environment and dependency definitions. |

Trained weights (`*.pt`) and the datasets are not tracked in the repository, as
they are too large. See `data/README.md` for how to regenerate the datasets, and
`outputs/link_to_videos.txt` for the rendered result videos.

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and targets Python 3.12.

```bash
uv sync
```

Segment Anything 2 is optional and only needed for the segmentation notebook:

```bash
uv pip install -q git+https://github.com/facebookresearch/segment-anything-2.git
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

## Usage

Prepare the keypoint datasets (downloads the SoccerNet calibration data first):

```bash
bash src/data_prep/run_dataload.sh
```

Train a keypoint model:

```bash
python -m src.train.keypoints.main --action train --epochs 50
```

Train the detection baseline and the ball detector:

```bash
python -m src.train.tracking.baseline_train
python -m src.train.tracking.ball.fine_tuning
```

Run the full pipeline (edit the configuration block at the top of `main.py` to
choose the source and the keypoint model):

```bash
python main.py
```

Evaluate:

```bash
python -m src.evaluation.keypoints.execute_evaluation
python -m src.evaluation.tracking.eval
```

## Results

Detection, on the SoccerNet tracking data:

| Run | Model | box mAP50 | box mAP50-95 |
|-----|-------|-----------|--------------|
| `detect/train` | YOLO11m baseline | 0.431 | 0.245 |
| `detect/train2` | baseline, retrained | 0.473 | 0.269 |
| `detect/train3` | baseline, best run | **0.704** | **0.419** |
| `detect/train4` | baseline, longer run | 0.573 | 0.347 |
| `ball` | ball-only fine-tune | 0.587 | 0.244 |

`detect/train3` is the detector the pipeline uses. The ball is detected by the
dedicated `ball` model with sliced inference, which recovers the small objects
the baseline misses.

Keypoints, on the SoccerNet calibration test split:

| Model | pose mAP50 | pose mAP50-95 | median RMSE | visibility P / R |
|-------|-----------|---------------|-------------|------------------|
| 29 keypoints | **0.735** | **0.622** | **59.9 px** | 0.805 / 0.856 |
| 57 keypoints | 0.599 | 0.465 | 74.4 px | 0.779 / 0.803 |

The extended 57-point layout was expected to give a better-conditioned
homography, since it adds the centre circle tangents and the penalty arcs. It
did not: every metric came out worse than the 29-point model, most likely
because the same amount of training data has to cover almost twice as many
points. The pipeline therefore defaults to the 29-point model
(`USE_57_POINTS = False` in `main.py`), and the 57-point path is kept because it
is a real result worth reporting.

Per-keypoint RMSE, per-frame errors and confidences are stored under
`src/evaluation/keypoints/evaluation_outputs/`.

## Approaches that were tried and dropped

Three lines of work did not make it into the pipeline. They are kept on their own
branches rather than deleted, because they document what was ruled out:

- `experiment/hrnet-calibration` - camera calibration with the HRNetV2 baseline
  from the SoccerNet calibration challenge, predicting line heatmaps and fitting
  the camera from them. It never produced a homography stable enough to use.
- `experiment/yolo-pose-homography` - an earlier YOLO pose attempt built on
  Roboflow's `sports` library with a 32-point pitch layout. Superseded by the
  29 and 57 point pipelines on `main`.
- `experiment/ball-action-spotting` - ball action spotting with T-DEED on
  SoccerNet Ball Action Spotting. A different task from the rest of the project,
  abandoned to keep the scope on detection, tracking and calibration.

## Data sources and credits

- Data: [SoccerNet](https://www.soccer-net.org/) calibration and tracking splits.
- The 29-keypoint pipeline is adapted from
  [Adit-jain/Soccer_Analysis](https://github.com/Adit-jain/Soccer_Analysis).
- The 57-keypoint geometry is adapted from
  [NikolasEnt/soccernet-calibration-sportlight](https://github.com/NikolasEnt/soccernet-calibration-sportlight).
- Pitch rendering and configuration come from
  [roboflow/sports](https://github.com/roboflow/sports).

## Authors

[Valentino Arbelaiz](https://github.com/varbelaiz),
[Franco Amato de Lusarreta](https://github.com/famatodlr) and
[Ana Paula Tissera](https://github.com/anatissera).
