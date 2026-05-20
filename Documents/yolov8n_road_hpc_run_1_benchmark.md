# YOLOv8n Road HPC Run 1 Benchmark

Generated: 2026-05-20  
Local run folder: `S:\Project\yolo\runs\yolov8n_road_hpc run 1`  
HPC save directory from `args.yaml`: `/rwthfs/rz/cluster/hpcwork/niy86040/road-scene-yolov8/runs/yolov8n_road`

## Run Summary

This run trained YOLOv8n for road-object detection on the KITTI-derived YOLO dataset.

Classes:

| ID | Class |
|---:|---|
| 0 | car |
| 1 | pedestrian |
| 2 | cyclist |

Training completed successfully for 50 epochs.

| Field | Value |
|---|---:|
| Model | YOLOv8n |
| Base weights | `yolov8n.pt` |
| Task | detect |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 |
| Workers | 4 |
| Device | CUDA GPU |
| HPC GPU observed in training log | NVIDIA H100 |
| Save period | every 5 epochs |
| Runtime from `results.csv` | 2090.81 sec / 34.85 min |
| Final checkpoint | `weights/last.pt` |
| Best checkpoint | `weights/best.pt` |

## Dataset Snapshot

Counts exclude `.gitkeep` files.

| Split | Images | Label Files | Objects |
|---|---:|---:|---:|
| train | 5984 | 5984 | 27974 |
| val | 1497 | 1497 | 6882 |

Object distribution:

| Split | car | pedestrian | cyclist |
|---|---:|---:|---:|
| train | 23062 | 3591 | 1321 |
| val | 5680 | 896 | 306 |

## Final Validation Metrics

Final epoch metrics from `results.csv`:

| Metric | Value |
|---|---:|
| Precision(B) | 0.88290 |
| Recall(B) | 0.77891 |
| mAP50(B) | 0.86590 |
| mAP50-95(B) | 0.59404 |
| train box loss | 0.74500 |
| train cls loss | 0.44305 |
| train dfl loss | 0.87818 |
| val box loss | 0.78592 |
| val cls loss | 0.45578 |
| val dfl loss | 0.90612 |

Best metric epochs from `results.csv`:

| Metric | Best Epoch | Value |
|---|---:|---:|
| mAP50(B) | 50 | 0.86590 |
| mAP50-95(B) | 50 | 0.59404 |
| Precision(B) | 27 | 0.90199 |
| Recall(B) | 49 | 0.78521 |

Per-class validation metrics from the training completion log:

| Class | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| all | 1497 | 6882 | 0.882 | 0.778 | 0.866 | 0.594 |
| car | 1333 | 5680 | 0.926 | 0.910 | 0.960 | 0.769 |
| pedestrian | 357 | 896 | 0.858 | 0.646 | 0.781 | 0.442 |
| cyclist | 222 | 306 | 0.862 | 0.778 | 0.856 | 0.571 |

## Interpretation

The run is strong for a first CV-ready baseline:

- Cars are very strong: high recall and high mAP50-95.
- Cyclists are usable but limited by fewer validation examples.
- Pedestrian recall is the weakest area and is the main accuracy gap to discuss in the README.
- Final epoch is also the best epoch for mAP50 and mAP50-95, so `best.pt` is the right model to keep.
- Training was fast on HPC: about 35 minutes for 50 epochs.

## Run Configuration

Key values parsed from `args.yaml`:

| Setting | Value |
|---|---|
| task | detect |
| mode | train |
| model | `/hpcwork/niy86040/road-scene-yolov8/yolov8n.pt` |
| data | `/hpcwork/niy86040/road-scene-yolov8/data/road_dataset/data.yaml` |
| epochs | 50 |
| patience | 100 |
| batch | 16 |
| imgsz | 640 |
| save | true |
| save_period | 5 |
| cache | false |
| device | `0` |
| workers | 4 |
| project | `/hpcwork/niy86040/road-scene-yolov8/runs` |
| name | `yolov8n_road` |
| pretrained | true |
| optimizer | auto |
| seed | 0 |
| deterministic | true |
| amp | true |
| fraction | 1.0 |
| val | true |
| split | val |
| iou | 0.7 |
| max_det | 300 |
| lr0 | 0.01 |
| lrf | 0.01 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| warmup_epochs | 3.0 |
| box | 7.5 |
| cls | 0.5 |
| dfl | 1.5 |
| hsv_h | 0.015 |
| hsv_s | 0.7 |
| hsv_v | 0.4 |
| translate | 0.1 |
| scale | 0.5 |
| fliplr | 0.5 |
| mosaic | 1.0 |
| close_mosaic | 10 |
| erasing | 0.4 |
| tracker | botsort.yaml |

## Artifact Inventory

Top-level run files:

| File | Size | Purpose |
|---|---:|---|
| `args.yaml` | 1.7 KB | Full YOLO training configuration |
| `results.csv` | 6.3 KB | Per-epoch metrics and losses |
| `results.png` | 294.8 KB | Training curves overview |
| `BoxF1_curve.png` | 183.5 KB | F1-confidence curve |
| `BoxPR_curve.png` | 149.0 KB | Precision-recall curve |
| `BoxP_curve.png` | 150.2 KB | Precision-confidence curve |
| `BoxR_curve.png` | 174.9 KB | Recall-confidence curve |
| `confusion_matrix.png` | 133.2 KB | Raw confusion matrix |
| `confusion_matrix_normalized.png` | 131.0 KB | Normalized confusion matrix |
| `labels.jpg` | 112.5 KB | Label distribution visualization |
| `train_batch0.jpg` | 471.4 KB | Training batch visual check |
| `train_batch1.jpg` | 529.8 KB | Training batch visual check |
| `train_batch2.jpg` | 466.4 KB | Training batch visual check |
| `train_batch14960.jpg` | 351.1 KB | Late training batch visual check |
| `train_batch14961.jpg` | 322.8 KB | Late training batch visual check |
| `train_batch14962.jpg` | 337.2 KB | Late training batch visual check |
| `val_batch0_labels.jpg` | 301.2 KB | Validation labels visualization |
| `val_batch0_pred.jpg` | 306.5 KB | Validation predictions visualization |
| `val_batch1_labels.jpg` | 294.5 KB | Validation labels visualization |
| `val_batch1_pred.jpg` | 302.1 KB | Validation predictions visualization |
| `val_batch2_labels.jpg` | 289.0 KB | Validation labels visualization |
| `val_batch2_pred.jpg` | 296.2 KB | Validation predictions visualization |

Weights:

| File | Size | Notes |
|---|---:|---|
| `weights/best.pt` | 5.93 MB | Best model for inference, validation, demo, and FPS benchmark |
| `weights/last.pt` | 5.93 MB | Final epoch checkpoint |
| `weights/epoch0.pt` | 23.32 MB | Periodic checkpoint |
| `weights/epoch5.pt` | 23.32 MB | Periodic checkpoint |
| `weights/epoch10.pt` | 23.32 MB | Periodic checkpoint |
| `weights/epoch15.pt` | 23.33 MB | Periodic checkpoint |
| `weights/epoch20.pt` | 23.33 MB | Periodic checkpoint |
| `weights/epoch25.pt` | 23.33 MB | Periodic checkpoint |
| `weights/epoch30.pt` | 23.33 MB | Periodic checkpoint |
| `weights/epoch35.pt` | 23.33 MB | Periodic checkpoint |
| `weights/epoch40.pt` | 23.33 MB | Periodic checkpoint |
| `weights/epoch45.pt` | 23.33 MB | Periodic checkpoint |

Storage footprint:

| Component | Size |
|---|---:|
| Non-weight artifacts | 5.47 MB |
| Weights folder | 245.11 MB |
| Total run folder | 250.58 MB |

## Epoch Metrics

| Epoch | Time Sec | Train Box | Train Cls | Train DFL | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 53.07 | 1.3668 | 1.4320 | 1.0785 | 0.6422 | 0.5006 | 0.5565 | 0.3191 |
| 2 | 93.99 | 1.2780 | 0.9946 | 1.0594 | 0.7138 | 0.4703 | 0.5607 | 0.3290 |
| 3 | 137.22 | 1.2393 | 0.8968 | 1.0475 | 0.6452 | 0.5721 | 0.6268 | 0.3593 |
| 4 | 179.25 | 1.2165 | 0.8439 | 1.0388 | 0.7326 | 0.5279 | 0.6222 | 0.3560 |
| 5 | 221.92 | 1.1692 | 0.8030 | 1.0253 | 0.7403 | 0.5858 | 0.6556 | 0.3880 |
| 6 | 262.32 | 1.1422 | 0.7684 | 1.0142 | 0.7292 | 0.5765 | 0.6575 | 0.3893 |
| 7 | 302.40 | 1.1186 | 0.7498 | 1.0084 | 0.7560 | 0.6293 | 0.7011 | 0.4247 |
| 8 | 346.66 | 1.1050 | 0.7324 | 0.9995 | 0.7732 | 0.6350 | 0.7194 | 0.4318 |
| 9 | 388.79 | 1.0899 | 0.7168 | 0.9965 | 0.8077 | 0.6229 | 0.7228 | 0.4358 |
| 10 | 430.22 | 1.0797 | 0.7063 | 0.9898 | 0.7705 | 0.6366 | 0.7121 | 0.4359 |
| 11 | 473.91 | 1.0590 | 0.6876 | 0.9826 | 0.8261 | 0.6558 | 0.7467 | 0.4641 |
| 12 | 514.54 | 1.0475 | 0.6758 | 0.9792 | 0.8009 | 0.6403 | 0.7378 | 0.4527 |
| 13 | 558.30 | 1.0406 | 0.6694 | 0.9762 | 0.8333 | 0.6617 | 0.7600 | 0.4645 |
| 14 | 598.43 | 1.0241 | 0.6569 | 0.9701 | 0.8101 | 0.6625 | 0.7493 | 0.4615 |
| 15 | 640.01 | 1.0126 | 0.6456 | 0.9698 | 0.7988 | 0.6698 | 0.7578 | 0.4727 |
| 16 | 681.73 | 1.0063 | 0.6380 | 0.9642 | 0.8263 | 0.7043 | 0.7913 | 0.4858 |
| 17 | 723.08 | 0.9975 | 0.6323 | 0.9609 | 0.8027 | 0.6895 | 0.7652 | 0.4787 |
| 18 | 763.72 | 0.9940 | 0.6252 | 0.9580 | 0.8438 | 0.7075 | 0.7998 | 0.5065 |
| 19 | 804.16 | 0.9841 | 0.6197 | 0.9572 | 0.8351 | 0.6818 | 0.7839 | 0.4982 |
| 20 | 845.07 | 0.9716 | 0.6112 | 0.9524 | 0.8552 | 0.7080 | 0.7958 | 0.5004 |
| 21 | 887.43 | 0.9644 | 0.6057 | 0.9483 | 0.8484 | 0.7012 | 0.7991 | 0.5052 |
| 22 | 928.27 | 0.9569 | 0.5980 | 0.9469 | 0.8314 | 0.7185 | 0.8064 | 0.5178 |
| 23 | 969.62 | 0.9499 | 0.5916 | 0.9441 | 0.8807 | 0.7097 | 0.8198 | 0.5190 |
| 24 | 1014.07 | 0.9405 | 0.5846 | 0.9392 | 0.8561 | 0.7145 | 0.8097 | 0.5211 |
| 25 | 1054.16 | 0.9336 | 0.5766 | 0.9375 | 0.8790 | 0.7326 | 0.8271 | 0.5278 |
| 26 | 1094.49 | 0.9292 | 0.5766 | 0.9367 | 0.8638 | 0.7351 | 0.8281 | 0.5401 |
| 27 | 1137.97 | 0.9194 | 0.5682 | 0.9321 | 0.9020 | 0.7048 | 0.8222 | 0.5380 |
| 28 | 1179.26 | 0.9115 | 0.5639 | 0.9328 | 0.8644 | 0.7229 | 0.8245 | 0.5327 |
| 29 | 1219.95 | 0.9099 | 0.5613 | 0.9294 | 0.8591 | 0.7519 | 0.8340 | 0.5480 |
| 30 | 1261.43 | 0.9004 | 0.5549 | 0.9275 | 0.8361 | 0.7679 | 0.8395 | 0.5506 |
| 31 | 1302.11 | 0.8952 | 0.5516 | 0.9254 | 0.8621 | 0.7520 | 0.8390 | 0.5534 |
| 32 | 1344.14 | 0.8834 | 0.5405 | 0.9227 | 0.8751 | 0.7458 | 0.8461 | 0.5608 |
| 33 | 1385.77 | 0.8829 | 0.5388 | 0.9205 | 0.8428 | 0.7718 | 0.8490 | 0.5628 |
| 34 | 1425.23 | 0.8730 | 0.5354 | 0.9184 | 0.8596 | 0.7667 | 0.8507 | 0.5670 |
| 35 | 1468.00 | 0.8660 | 0.5294 | 0.9146 | 0.8609 | 0.7658 | 0.8506 | 0.5649 |
| 36 | 1509.27 | 0.8631 | 0.5262 | 0.9168 | 0.8680 | 0.7609 | 0.8529 | 0.5656 |
| 37 | 1549.38 | 0.8514 | 0.5178 | 0.9137 | 0.8719 | 0.7589 | 0.8519 | 0.5751 |
| 38 | 1591.58 | 0.8532 | 0.5220 | 0.9148 | 0.8563 | 0.7821 | 0.8536 | 0.5738 |
| 39 | 1631.22 | 0.8461 | 0.5160 | 0.9075 | 0.8647 | 0.7762 | 0.8570 | 0.5782 |
| 40 | 1674.00 | 0.8413 | 0.5121 | 0.9101 | 0.8649 | 0.7816 | 0.8604 | 0.5850 |
| 41 | 1715.91 | 0.8183 | 0.4886 | 0.9001 | 0.8704 | 0.7683 | 0.8540 | 0.5705 |
| 42 | 1755.10 | 0.8033 | 0.4825 | 0.8942 | 0.8765 | 0.7750 | 0.8565 | 0.5714 |
| 43 | 1796.76 | 0.7923 | 0.4732 | 0.8920 | 0.8770 | 0.7738 | 0.8573 | 0.5783 |
| 44 | 1837.23 | 0.7839 | 0.4686 | 0.8895 | 0.8783 | 0.7709 | 0.8586 | 0.5814 |
| 45 | 1876.55 | 0.7760 | 0.4646 | 0.8868 | 0.8916 | 0.7719 | 0.8597 | 0.5869 |
| 46 | 1916.79 | 0.7721 | 0.4617 | 0.8870 | 0.8774 | 0.7734 | 0.8586 | 0.5845 |
| 47 | 1956.72 | 0.7614 | 0.4551 | 0.8832 | 0.8988 | 0.7670 | 0.8641 | 0.5878 |
| 48 | 1998.15 | 0.7584 | 0.4520 | 0.8810 | 0.8918 | 0.7706 | 0.8636 | 0.5899 |
| 49 | 2047.07 | 0.7514 | 0.4465 | 0.8799 | 0.8714 | 0.7852 | 0.8658 | 0.5898 |
| 50 | 2090.81 | 0.7450 | 0.4430 | 0.8782 | 0.8829 | 0.7789 | 0.8659 | 0.5940 |

## Future Benchmark Fields

For future runs, record these fields in the same format:

- Run name and folder.
- Model size: `yolov8n`, `yolov8s`, etc.
- Dataset version and object counts.
- Epochs, image size, batch size, workers, GPU.
- Runtime and average epoch time.
- Final precision, recall, mAP50, mAP50-95.
- Best epoch for mAP50-95.
- Per-class metrics.
- Local FPS benchmark on the GTX 1050 Ti.
- Demo video path and ByteTrack tracking output path.

## Next Benchmark Targets

Recommended next comparisons:

| Run | Purpose |
|---|---|
| YOLOv8n local FPS | Deployment speed baseline on laptop |
| YOLOv8n ByteTrack demo | Tracking proof for portfolio |
| YOLOv8s HPC training | Accuracy-speed comparison if HPC time allows |
| Failure-case analysis | Recruiter-readable discussion of limitations |

