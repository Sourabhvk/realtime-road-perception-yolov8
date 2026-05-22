# HPC Training Experiment Log

Updated: 2026-05-22  
Project: Real-Time Road Object Detection and Tracking using YOLOv8 + ByteTrack

This file records what was planned, what actually ran, what went wrong, and which model results should be used in the final project.

## Dataset

Dataset: KITTI Object Detection converted to YOLO format.

HPC dataset path:

```text
/hpcwork/niy86040/road-scene-yolov8/data/road_dataset
```

Expected counts:

| Folder | Count |
|---|---:|
| `images/train` | 5984 |
| `images/val` | 1497 |
| `labels/train` | 5984 |
| `labels/val` | 1497 |

Classes:

| ID | Class |
|---:|---|
| 0 | car |
| 1 | pedestrian |
| 2 | cyclist |

## Final Valid Experiments

These are the runs that should be used for the README and final benchmark comparison.

| Run | Model | Batch | LR Schedule | Epochs | Runtime | Precision | Recall | mAP50 | mAP50-95 | Use |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `yolov8n_b16_default` | YOLOv8n | 16 | default | 50 | 34.85 min | 0.88290 | 0.77891 | 0.86590 | 0.59404 | lightweight baseline |
| `yolov8s_b16_default` | YOLOv8s | 16 | default | 50 | 51.50 min | 0.90265 | 0.85003 | 0.90572 | 0.64728 | best accuracy |
| `yolov8s_b32_cos` | YOLOv8s | 32 | cosine | 50 | 35.94 min | 0.91500 | 0.83952 | 0.89980 | 0.64136 | best speed/accuracy tradeoff |

## Final Recommendation

Use `yolov8s_b16_default` as the best accuracy model.

Use `yolov8n_b16_default` for local speed/FPS comparison.

Use `yolov8s_b32_cos` to discuss an efficient training configuration: it is close to the best model while training much faster than YOLOv8s batch16.

## Run 1: YOLOv8n Baseline

Purpose: establish lightweight baseline.

Settings:

| Field | Value |
|---|---|
| Model | `yolov8n.pt` |
| Epochs | 50 |
| Image size | 640 |
| Batch size | 16 |
| Workers | 4 |
| LR schedule | default |
| `cos_lr` | `False` |
| `lr0` | `0.01` |
| `lrf` | `0.01` |
| Optimizer | `auto` |
| AMP | `True` |
| HPC GPU | NVIDIA H100 |

Result:

| Metric | Value |
|---|---:|
| Precision | 0.88290 |
| Recall | 0.77891 |
| mAP50 | 0.86590 |
| mAP50-95 | 0.59404 |
| Runtime | 34.85 min |

Per-class metrics:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| car | 0.926 | 0.910 | 0.960 | 0.769 |
| pedestrian | 0.858 | 0.646 | 0.781 | 0.442 |
| cyclist | 0.862 | 0.778 | 0.856 | 0.571 |

## Run 2: YOLOv8s Batch16 Default

Purpose: test larger YOLOv8s model against YOLOv8n baseline using default training settings.

Safe submit command used:

```bash
sbatch --job-name=yolov8s_b16 --output=/hpcwork/niy86040/road-scene-yolov8/runs/slurm/%x-%j.out --error=/hpcwork/niy86040/road-scene-yolov8/runs/slurm/%x-%j.err --time=12:00:00 --partition=c23g --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:1 --chdir=/hpcwork/niy86040/road-scene-yolov8 --wrap="MODEL=yolov8s.pt RUN_NAME=yolov8s_b16_default EPOCHS=50 BATCH=16 COS_LR=False LR0=0.01 LRF=0.01 bash scripts/train_hpc.sbatch"
```

Result:

| Metric | Value |
|---|---:|
| Precision | 0.90265 |
| Recall | 0.85003 |
| mAP50 | 0.90572 |
| mAP50-95 | 0.64728 |
| Runtime | 51.50 min |

This was the best overall accuracy run.

## Run 3: YOLOv8s Batch32 Cosine

Purpose: test whether larger batch and cosine learning-rate schedule improve training efficiency while keeping accuracy close.

Safe submit command used:

```bash
sbatch --job-name=yolov8s_b32 --output=/hpcwork/niy86040/road-scene-yolov8/runs/slurm/%x-%j.out --error=/hpcwork/niy86040/road-scene-yolov8/runs/slurm/%x-%j.err --time=12:00:00 --partition=c23g --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:1 --chdir=/hpcwork/niy86040/road-scene-yolov8 --wrap="MODEL=yolov8s.pt RUN_NAME=yolov8s_b32_cos EPOCHS=50 BATCH=32 COS_LR=True LR0=0.01 LRF=0.001 bash scripts/train_hpc.sbatch"
```

Result:

| Metric | Value |
|---|---:|
| Precision | 0.91500 |
| Recall | 0.83952 |
| mAP50 | 0.89980 |
| mAP50-95 | 0.64136 |
| Runtime | 35.94 min |

This was the best speed/accuracy tradeoff run.

## Accidental Duplicate Runs

Three jobs were accidentally submitted with the intention of running:

1. YOLOv8n batch32 cosine
2. YOLOv8s batch16 default
3. YOLOv8s batch32 cosine

But they actually all ran the default script settings:

```text
model=yolov8n.pt
batch=16
cos_lr=False
lr0=0.01
lrf=0.01
run_name=yolov8n_road
```

Cause:

```bash
MODEL=yolov8s.pt RUN_NAME=yolov8s_b16_default BATCH=16 sbatch scripts/train_hpc.sbatch
```

did not reliably pass variables into the Slurm job environment.

Result: all three jobs became duplicate YOLOv8n baseline runs and are excluded from the comparison table.

Approximate wasted compute:

| Job | Runtime | Billing TRES | Approx Core-Hours |
|---:|---:|---|---:|
| 521679 | 52.4 min | billing=24 | 20.96 |
| 521680 | 38.1 min | billing=24 | 15.23 |
| 521685 | 34.5 min | billing=24 | 13.80 |
| total | - | - | 49.99 |

## Slurm Lessons

Do not use this for custom experiments:

```bash
MODEL=yolov8s.pt RUN_NAME=yolov8s_b16_default BATCH=16 sbatch scripts/train_hpc.sbatch
```

Use `--wrap` instead:

```bash
sbatch [slurm resources] --wrap="MODEL=... RUN_NAME=... BATCH=... bash scripts/train_hpc.sbatch"
```

Why: with `--wrap`, the variables are part of the command that actually runs inside the Slurm job.

## Pre-Run Checks

Before submitting future jobs:

```bash
cd /hpcwork/niy86040/road-scene-yolov8
git pull origin main
ls -lh yolov8n.pt yolov8s.pt
find data/road_dataset/images/train -type f | wc -l
find data/road_dataset/images/val -type f | wc -l
find data/road_dataset/labels/train -type f | wc -l
find data/road_dataset/labels/val -type f | wc -l
```

After a job starts, verify the actual config:

```bash
grep -H "Training config" runs/slurm/<job-log>.out
```

## Output Files

Structured summaries:

- `Documents/model_metrics_comparison.md`
- `Documents/model_metrics_summary.csv`
- `Documents/model_metrics_artifact_inventory.csv`

Curated output folders:

- `outputs/yolov8n_b16_default`
- `outputs/yolov8s_b16_default`
- `outputs/yolov8s_b32_cos`

Raw Ultralytics folders should remain under `runs/` and do not need to be committed.

## Remaining Local Work

- Run local FPS benchmarks for each final model.
- Pick best visual examples for README.
- Add ByteTrack output video/GIF or link.
- Update README with final metrics table.
- Add short failure-case analysis.

