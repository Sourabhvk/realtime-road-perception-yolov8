# HPC Training Experiment Plan

Created: 2026-05-20  
Project: Real-Time Road Object Detection and Tracking using YOLOv8 + ByteTrack

## Completed Baseline Run

| Field | Value |
|---|---|
| Run label | HPC Run 1 |
| Model | YOLOv8n |
| Base weights | `yolov8n.pt` |
| Dataset | `data/road_dataset/data.yaml` |
| Epochs | 50 |
| Image size | 640 |
| Batch size | 16 |
| Workers | 4 |
| LR schedule | default YOLO schedule |
| `cos_lr` | `False` |
| `lr0` | `0.01` |
| `lrf` | `0.01` |
| Optimizer | `auto` |
| AMP | `True` |
| HPC GPU observed | NVIDIA H100 |
| Runtime | 34.85 min |
| Output folder | `runs/yolov8n_road_hpc run 1` |
| Best model | `runs/yolov8n_road_hpc run 1/weights/best.pt` |

Final validation metrics:

| Metric | Value |
|---|---:|
| Precision | 0.88290 |
| Recall | 0.77891 |
| mAP50 | 0.86590 |
| mAP50-95 | 0.59404 |

Per-class final metrics:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| car | 0.926 | 0.910 | 0.960 | 0.769 |
| pedestrian | 0.858 | 0.646 | 0.781 | 0.442 |
| cyclist | 0.862 | 0.778 | 0.856 | 0.571 |

## Planned Next Runs

Goal: compare model size and schedule/batch changes while keeping the dataset fixed.

| Run label | Model | Epochs | Batch | LR schedule | `cos_lr` | `lr0` | `lrf` | Purpose |
|---|---|---:|---:|---|---|---:|---:|---|
| HPC Run 2 | YOLOv8n | 50 | 32 | cosine | `True` | 0.01 | 0.001 | Test whether larger batch + cosine improves YOLOv8n |
| HPC Run 3 | YOLOv8s | 50 | 16 | default | `False` | 0.01 | 0.01 | Baseline YOLOv8s comparison against YOLOv8n |
| HPC Run 4 | YOLOv8s | 50 | 32 | cosine | `True` | 0.01 | 0.001 | Test whether larger YOLOv8s + cosine improves accuracy |

## Submit Commands

Run 2: YOLOv8n, batch 32, cosine LR:

```bash
MODEL=yolov8n.pt RUN_NAME=yolov8n_b32_cos EPOCHS=50 BATCH=32 COS_LR=True LR0=0.01 LRF=0.001 sbatch scripts/train_hpc.sbatch
```

Run 3: YOLOv8s, old/default settings:

```bash
MODEL=yolov8s.pt RUN_NAME=yolov8s_b16_default EPOCHS=50 BATCH=16 COS_LR=False LR0=0.01 LRF=0.01 sbatch scripts/train_hpc.sbatch
```

Run 4: YOLOv8s, batch 32, cosine LR:

```bash
MODEL=yolov8s.pt RUN_NAME=yolov8s_b32_cos EPOCHS=50 BATCH=32 COS_LR=True LR0=0.01 LRF=0.001 sbatch scripts/train_hpc.sbatch
```

## Pre-Run Checks

Before submitting, verify weights and dataset exist on HPC:

```bash
cd /hpcwork/niy86040/road-scene-yolov8
ls -lh yolov8n.pt yolov8s.pt
find data/road_dataset/images/train -type f | wc -l
find data/road_dataset/images/val -type f | wc -l
find data/road_dataset/labels/train -type f | wc -l
find data/road_dataset/labels/val -type f | wc -l
```

Expected dataset counts:

| Folder | Count |
|---|---:|
| `images/train` | 5984 |
| `images/val` | 1497 |
| `labels/train` | 5984 |
| `labels/val` | 1497 |

## Evaluation Fields To Record

For each run, record:

- run folder
- training runtime
- best epoch
- final precision
- final recall
- final mAP50
- final mAP50-95
- per-class precision/recall/mAP
- local FPS on GTX 1050 Ti
- ByteTrack demo output path
- failure cases

## Decision Criteria

Best CV-ready model is not necessarily the highest mAP only.

Use:

- mAP50-95 for detection quality
- recall for pedestrian/cyclist safety relevance
- local FPS for real-time practicality
- ByteTrack visual stability for demo quality

