# Experiment: ball action spotting with T-DEED

Dropped experiment. Kept as a record of a direction that was explored and then
abandoned. The pipeline that shipped is on `main`.

## What this was

Action spotting on SoccerNet: given a match video, predict *when* events happen
(pass, drive, shot, cross, and so on) rather than where the players are.

The approach was [T-DEED](https://github.com/arturxe2/T-DEED), a
temporally-discriminative encoder-decoder built on a GSF/GSM feature extractor,
trained on **SoccerNet Ball Action Spotting**.

```
config/                 dataset and training configurations
data/                   class lists and train/val/test splits
dataset/                frame and clip datasets
model/                  T-DEED, GSF and GSM implementations
util/                   evaluation, scoring and IO helpers
download_data.py        fetches the SoccerNet videos and labels
extract_frames_sn.py    extracts frames for SoccerNet Action Spotting
extract_frames_snb.py   extracts frames for SoccerNet Ball Action Spotting
train_tdeed.py          training entry point
train_tdeed_bas.py      training entry point for Ball Action Spotting
evaluate_tdeed_challenge.py
```

Everything except the extraction scripts, the download helper and the configs is
vendored from the T-DEED repository and left as upstream published it.

## Why it was dropped

Not because it failed, but because it was the wrong scope.

- Action spotting is a different task from the rest of the project. It answers
  "when did a pass happen", while the rest of the work answers "where is
  everyone on the pitch right now". They share the dataset and nothing else.
- It needed the full SoccerNet video set decoded to frames, which is an order of
  magnitude more data than the detection and calibration work, for a result that
  would not have fed into the tactical minimap.
- With limited time, keeping detection, tracking and calibration good was worth
  more than adding a fourth, unrelated model.

Work stopped here on 26 November. The team put the remaining time into the
detection, tracking and homography pipeline that is on `main`.

## Credits

- [T-DEED](https://github.com/arturxe2/T-DEED)
- [SoccerNet action spotting](https://github.com/SoccerNet/sn-spotting)
