# SoccerNet Calibration Dataset

This directory holds the camera calibration data used to build the 29 and 57
keypoint datasets. Everything in it is generated; nothing is tracked in git.

```
data/calibration/
├── train.zip, valid.zip, test.zip     raw downloads
├── images/                            extracted frames, per split
├── soccernet_calibration_annotations/ line annotations as JSON, per split
├── unified_output/                    final 29 keypoint dataset
└── unified_output_57/                 final 57 keypoint dataset
```

Both `unified_output*` folders follow the layout Ultralytics expects:

```
unified_output/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/
    ├── train/
    ├── valid/
    └── test/
```

## Generating the data

From the repository root:

```bash
export SOCCERNET_PASSWORD='...'
bash src/data_prep/run_dataload.sh
```

The script:

1. Downloads the calibration zips into this directory.
2. Extracts and separates images from JSON annotations.
3. Generates `unified_output/` (29 keypoints) and `unified_output_57/`
   (57 keypoints).

It finishes with:

```
PIPELINE COMPLETE
   1. 29 KP: data/calibration/unified_output
   2. 57 KP: data/calibration/unified_output_57
```

See [src/data_prep/README.md](../../src/data_prep/README.md) for what happens in
each step.
