# Data

None of the datasets are tracked in the repository. This directory holds the
READMEs that describe the expected layout and how to regenerate each dataset.

```
data/
├── calibration/    SoccerNet calibration data -> 29 and 57 keypoint datasets
└── tracking/       SoccerNet tracking data -> detection and ball datasets
```

- `calibration/` is produced automatically by `src/data_prep/run_dataload.sh`.
  See [calibration/README.md](calibration/README.md).
- `tracking/` has to be downloaded and converted to the YOLO layout described in
  [tracking/README.md](tracking/README.md).

Downloading SoccerNet requires the dataset password, which is under an NDA and is
therefore not committed. Export it before running the downloader:

```bash
export SOCCERNET_PASSWORD='...'
```
