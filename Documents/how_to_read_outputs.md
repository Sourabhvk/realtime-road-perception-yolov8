# How To Read Outputs

This project separates raw experiment files from curated portfolio files.

## Folder Meaning

```text
runs/
```

Raw Ultralytics output. Treat this like a lab notebook. It can contain repeated experiments, temporary predictions, checkpoints, and large files.

```text
outputs/
```

Curated results for GitHub, README, and resume discussion. These are the files worth showing.

## Current Curated Runs

```text
outputs/yolov8n_b16_default/
outputs/yolov8s_b16_default/
outputs/yolov8s_b32_cos/
```

Each run should contain:

```text
args.yaml
Metrics/
Images/
```

`weights/` may exist locally, but model weights are ignored by Git unless explicitly allowed.

## Important Metric Files

Inside each `Metrics/` folder:

| File | Meaning | Why It Matters |
|---|---|---|
| `*_results.csv` | Per-epoch metrics and losses | Main source for benchmark tables |
| `*_results.png` | Training curves | Quick visual summary of training stability |
| `*_confusion_matrix.png` | Raw confusion matrix | Shows which classes are confused |
| `*_confusion_matrix_normalized.png` | Normalized confusion matrix | Easier class-level comparison |
| `*_BoxPR_curve.png` | Precision-recall curve | Shows detection quality across thresholds |
| `*_BoxF1_curve.png` | F1-confidence curve | Helps choose confidence threshold |
| `*_BoxP_curve.png` | Precision-confidence curve | Shows false-positive behavior |
| `*_BoxR_curve.png` | Recall-confidence curve | Shows missed-detection behavior |
| `*_labels.jpg` | Label distribution | Shows class imbalance in the dataset |

## Important Image Files

Inside each `Images/` folder:

| File Type | Meaning | Use |
|---|---|---|
| `train_batch*.jpg` | Training batch visualizations | Dataset sanity check |
| `val_batch*_labels.jpg` | Ground-truth validation labels | Compare labels vs predictions |
| `val_batch*_pred.jpg` | Model predictions | Best visual proof for README |

For README, prefer `val_batch*_pred.jpg` over training batch images.

## What Is Desirable

For this project, desirable metrics are:

| Metric | Good Direction | Why |
|---|---|---|
| Precision | higher | Fewer false detections |
| Recall | higher | Fewer missed cars/pedestrians/cyclists |
| mAP50 | higher | General object detection quality |
| mAP50-95 | higher | Stricter localization quality |
| Validation loss | lower | Better validation fit |
| Training time | lower | More efficient model/config |
| Local FPS | higher | Better real-time practicality |

## Current Model Ranking

Based on parsed metrics:

| Rank | Model Run | Reason |
|---:|---|---|
| 1 | `yolov8s_b16_default` | Best mAP50-95 and recall |
| 2 | `yolov8s_b32_cos` | Very close accuracy with much faster training |
| 3 | `yolov8n_b16_default` | Lightweight baseline, best for local speed comparison |

## Current Summary

| Run | Precision | Recall | mAP50 | mAP50-95 | Train Time |
|---|---:|---:|---:|---:|---:|
| `yolov8n_b16_default` | 0.88290 | 0.77891 | 0.86590 | 0.59404 | 34.85 min |
| `yolov8s_b16_default` | 0.90265 | 0.85003 | 0.90572 | 0.64728 | 51.50 min |
| `yolov8s_b32_cos` | 0.91500 | 0.83952 | 0.89980 | 0.64136 | 35.94 min |

## How To Use These In README

Use:

- `model_metrics_comparison.md` for result interpretation.
- `model_metrics_summary.csv` for the benchmark table.
- `*_results.png` for training curve screenshots.
- `*_confusion_matrix_normalized.png` for class-level evaluation.
- `val_batch*_pred.jpg` for visual examples.
- ByteTrack output video or GIF for the tracking demo.