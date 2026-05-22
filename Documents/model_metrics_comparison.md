# Model Metrics Comparison

Generated from the three curated metric folders under `outputs/`.

Metric folders parsed:

- `outputs/yolov8n_b16_default/Metrics`
- `outputs/yolov8s_b16_default/Metrics`
- `outputs/yolov8s_b32_cos/Metrics`

## Executive Summary

Best overall model by validation mAP50-95:

```text
YOLOv8s batch 16 default
```

It gives the strongest detection accuracy:

- Precision: `0.90265`
- Recall: `0.85003`
- mAP50: `0.90572`
- mAP50-95: `0.64728`

Fastest training run:

```text
YOLOv8n batch 16 default
```

But the best accuracy-speed tradeoff among YOLOv8s runs is:

```text
YOLOv8s batch 32 cosine
```

It is much faster to train than YOLOv8s batch 16 while losing only a small amount of accuracy.

## Final Metrics

| Run | Model | Batch | Schedule | Runtime min | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| `yolov8n_b16_default` | YOLOv8n | 16 | default | 34.85 | 0.88290 | 0.77891 | 0.86590 | 0.59404 |
| `yolov8s_b16_default` | YOLOv8s | 16 | default | 51.50 | 0.90265 | 0.85003 | 0.90572 | 0.64728 |
| `yolov8s_b32_cos` | YOLOv8s | 32 | cosine | 35.94 | 0.91500 | 0.83952 | 0.89980 | 0.64136 |

## Best Epochs

| Run | Best mAP50 Epoch | Best mAP50 | Best mAP50-95 Epoch | Best mAP50-95 | Best Precision Epoch | Best Precision | Best Recall Epoch | Best Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `yolov8n_b16_default` | 50 | 0.86590 | 50 | 0.59404 | 27 | 0.90199 | 49 | 0.78521 |
| `yolov8s_b16_default` | 50 | 0.90572 | 50 | 0.64728 | 47 | 0.92077 | 50 | 0.85003 |
| `yolov8s_b32_cos` | 48 | 0.90178 | 49 | 0.64188 | 46 | 0.91887 | 49 | 0.84286 |

## Final Losses

| Run | Train Box | Train Cls | Train DFL | Val Box | Val Cls | Val DFL |
|---|---:|---:|---:|---:|---:|---:|
| `yolov8n_b16_default` | 0.74500 | 0.44305 | 0.87818 | 0.78592 | 0.45578 | 0.90612 |
| `yolov8s_b16_default` | 0.61864 | 0.35701 | 0.84578 | 0.68803 | 0.38879 | 0.88422 |
| `yolov8s_b32_cos` | 0.61883 | 0.35521 | 0.84761 | 0.69026 | 0.38486 | 0.88731 |

## Improvements Over YOLOv8n Baseline

Compared to `yolov8n_b16_default`:

| Run | Precision Change | Recall Change | mAP50 Change | mAP50-95 Change | Runtime Change |
|---|---:|---:|---:|---:|---:|
| `yolov8s_b16_default` | +0.01975 | +0.07112 | +0.03982 | +0.05324 | +16.65 min |
| `yolov8s_b32_cos` | +0.03210 | +0.06061 | +0.03390 | +0.04732 | +1.09 min |

## Interpretation

`yolov8s_b16_default` is the strongest model for final reported accuracy.

`yolov8s_b32_cos` is close in accuracy and much faster to train, so it is useful as an efficient training configuration. It improves over YOLOv8n baseline while keeping runtime nearly the same as YOLOv8n.

`yolov8n_b16_default` remains useful for lightweight local inference and FPS comparison.

## Recommended README Table

| Model | Batch | LR Schedule | Precision | Recall | mAP50 | mAP50-95 | HPC Train Time |
|---|---:|---|---:|---:|---:|---:|---:|
| YOLOv8n | 16 | default | 0.88290 | 0.77891 | 0.86590 | 0.59404 | 34.85 min |
| YOLOv8s | 16 | default | 0.90265 | 0.85003 | 0.90572 | 0.64728 | 51.50 min |
| YOLOv8s | 32 | cosine | 0.91500 | 0.83952 | 0.89980 | 0.64136 | 35.94 min |

## Structured CSV Files

Machine-readable summaries were generated here:

- `Documents/model_metrics_summary.csv`
- `Documents/model_metrics_artifact_inventory.csv`

