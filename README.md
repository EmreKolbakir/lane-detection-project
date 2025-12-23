# Lane Detection Project

This repo compares two approaches for lane detection:

- Classical computer vision: Canny edge detection + Hough Transform.
- Deep learning: U-Net semantic segmentation (to be added).

The goal is to evaluate both methods on normal day, normal night, and rainy
driving videos, and compare visual quality and FPS.

## Folder Layout

- `classical_approach/`: Classical pipeline implementation.
- `deep_learning_approach/`: Deep learning training/inference (coming next).
- `notebooks/`: Scratch notebooks.
- `outputs/`: Generated videos (ignored by git).
- `data/`: Datasets (ignored by git).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

Training (U-Net):
- BDD100K (Kaggle): https://www.kaggle.com/datasets/marquis03/bdd100k

Testing (video comparison):
- Driving video for lane detection (Various weather):
  https://www.kaggle.com/datasets/ashikadnan/driving-video-for-lane-detection-variousweather

## Automatic Download (Kaggle)

You need Kaggle credentials. Use one of these options:

```bash
export KAGGLE_API_TOKEN="your_kaggle_token"
export KAGGLE_USERNAME="your_kaggle_username"
bash scripts/download_datasets.sh
```

Or place `kaggle.json` at `~/.kaggle/kaggle.json` (chmod 600), then run:

```bash
bash scripts/download_datasets.sh
```

Legacy method also works:

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_key"
bash scripts/download_datasets.sh
```

To also download the (very large) BDD100K dataset:

```bash
DOWNLOAD_BDD100K=1 bash scripts/download_datasets.sh
```

Suggested local structure:

```
data/
  videos/
    normal_day.mp4
    normal_night.mp4
    rainy_day.mp4
```

## Classical Approach (Canny + Hough)

Run on a single video:

```bash
python classical_approach/lane_detection_hough.py \
  --input data/videos/normal_day.mp4 \
  --output outputs/normal_day_classical.mp4 \
  --show
```

The script writes a video with a green lane polygon overlay and prints the
average FPS.
