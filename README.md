# Real-Time Road Object Detection and Tracking using YOLOv8

This repository contains a road-scene perception pipeline built around YOLOv8.
It includes KITTI-to-YOLO conversion, label visualization, quick inference tests,
FPS benchmarking, and a Slurm job script for Linux/HPC training.

## Goal

Detect and track road users such as cars, pedestrians, and cyclists, then
evaluate the model with detection metrics and speed benchmarks.

## Project Layout

- `scripts/convert_kitti_to_yolo.py` converts KITTI labels into YOLO format.
- `scripts/visualize_yolo_labels.py` draws YOLO boxes for a quick dataset check.
- `scripts/test_inference.py` runs a small pretrained inference smoke test.
- `scripts/benchmark_fps.py` measures inference throughput on a folder of images.
- `scripts/train_hpc.sbatch` submits a YOLO training job to Slurm.

## Dataset

Target classes:

```text
0: car
1: pedestrian
2: cyclist
```

Expected structure:

```text
data/
└── road_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
```

Example `data.yaml`:

```yaml
path: data/road_dataset
train: images/train
val: images/val

names:
  0: car
  1: pedestrian
  2: cyclist
```

## Setup

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux or HPC shell:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If your target machine needs a specific PyTorch build, install that before or
separately from `requirements.txt` so the wheel matches the local Python/CUDA
stack.

## Common Commands

```bash
python scripts/convert_kitti_to_yolo.py
python scripts/visualize_yolo_labels.py
python scripts/test_inference.py
python scripts/benchmark_fps.py --source data/road_dataset/images/val
```

## HPC Training

Submit the default Slurm job from the repo root:

```bash
sbatch scripts/train_hpc.sbatch
```

Override parameters with environment variables if needed:

```bash
MODEL=yolov8s.pt EPOCHS=75 BATCH=8 sbatch scripts/train_hpc.sbatch
```

## Notes

- `data/kitti_raw/` and `data/road_dataset/images` or `labels` are intentionally
  not tracked by git. Only `data/road_dataset/data.yaml` is expected to be
  committed.
- `models/*.pt` is ignored on purpose so trained weights stay local unless you
  copy them into the repo manually.
- The repository is meant to be usable on Windows for debugging and on Linux or
  RWTH HPC for actual training.