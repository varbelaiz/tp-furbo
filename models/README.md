# Trained models

Archived artefacts for every training run that was kept: the training curves,
the batch previews and the per-epoch metrics.

The weights themselves (`best.pt`, `last.pt`) are **not** tracked, since they are
far too large for the repository. Training writes them to `runs/`, which is
ignored by git; the folders here are the record of what each run achieved.

```
models/
├── ball/          ball-only detector, fine-tuned from detect/train4
├── detect/        general detector for players and referees
│   ├── train/     YOLO11m baseline, 15 epochs
│   ├── train2/    retrained baseline, 10 epochs
│   ├── train3/    best detection run, used by the pipeline
│   └── train4/    longer run, 41 epochs
└── keypoints/
    ├── run_1/     29 keypoint pose model
    └── 57_kp/     57 keypoint pose model
```

Each folder holds:

- `results.csv` - per-epoch losses and metrics
- `results.png` - the same, plotted
- `args.yaml` - the exact hyperparameters of the run
- `BoxP_curve.png`, `BoxR_curve.png`, `BoxF1_curve.png`, `BoxPR_curve.png` -
  detection curves (plus `Pose*` equivalents for the keypoint runs)
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `train_batch*.jpg`, `val_batch*_labels.jpg`, `val_batch*_pred.jpg` - sample
  batches with labels and predictions

Headline numbers for each run are in the results table in the
[top-level README](../README.md).

Note that `detect/train4/results.csv` has a truncated final row, from a run that
was interrupted while writing. Its last complete epoch is 40.
