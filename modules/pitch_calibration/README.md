# Pitch calibration with HRNetV2

Quick start for running the full training and inference pipeline from scratch on
a clean VM.

## 1. One-time setup

1. Clone the repository:
   ```bash
   git clone <REPO_URL>
   cd tp-furbo/modules/pitch_calibration
   ```

2. Install the dependencies:
   ```bash
   # Project dependencies
   pip install -r requirements.txt

   # Tooling
   pip install google-cloud-storage
   pip install tensorboard
   sudo apt install tmux
   ```

## 2. Running the pipeline

Follow the steps in order.

### Step 0: download the dataset

`download_dataset.py` fetches the `.zip` files into `temp/` and extracts them
into `dataset/`.

```bash
python -m scripts.download_dataset
```

### Step 1: precompute the heatmaps

Generates every heatmap and mask ahead of time. This matters: without it the
dataloader is far too slow to keep the GPU busy.

```bash
python -m scripts.generate_pn_keypoints --dataset_path ./dataset
```

Check the result before training on it:

```bash
python -m scripts.inspect_heatmaps
python -m scripts.inspect_npz
```

### Step 2: train the keypoint network (the long part)

Run it under `tmux` so the job survives a dropped SSH connection.

1. Start a detachable session:

   ```bash
   tmux new -s calib
   ```

2. Launch the training from inside `tmux`. This trains for 50 epochs, compiles
   the model, writes local checkpoints into `models/` and uploads the best one
   to GCS:

   ```bash
   python -m scripts.train \
       --SoccerNet_path <PATH_TO_REPO>/modules/pitch_calibration \
       --epochs 50 \
       --batch_size 16 \
       --output_folder ./models/run_final_50epochs \
       --gcs_bucket bucket-homo
   ```

3. Detach and leave it running: once `tqdm` starts, press `Ctrl` + `B`, release,
   then `D`.

The line network is trained the same way with `scripts.train_l`.

### Step 3: inference

Once training finishes, use `checkpoint_best.pth` to produce the JSON
predictions.

1. Make the helper scripts executable (first time only):

   ```bash
   chmod +x run_inference.sh
   chmod +x run_evaluation.sh
   ```

2. Run the inference. The script creates the output folder under `outputs/`:

   ```bash
   ./run_inference.sh ./models/run_final_50epochs/checkpoint_best.pth
   ```

### Step 4: evaluation

```bash
./run_evaluation.sh ./outputs/checkpoint_best
```

Look for `accuracy mean value` in the `evaluate_camera.py` log.

## 3. Useful `tmux` commands

* List active sessions:
  ```bash
  tmux ls
  ```
* Reattach to a session:
  ```bash
  tmux attach -t calib
  ```
* Kill a stuck session:
  ```bash
  tmux kill-session -t calib
  ```
* Kill the `tmux` server entirely:
  ```bash
  tmux kill-server
  ```
