# Real-Time Road Object Detection and Tracking using YOLOv8

This project builds a basic road-scene perception pipeline using YOLOv8 and object tracking.

The goal is to detect and track common road objects such as cars, pedestrians, and cyclists, then evaluate the model using accuracy metrics and FPS benchmarks.

---

## Objective

- Train or fine-tune YOLOv8 on road-scene data
- Detect cars, pedestrians, and cyclists
- Track detected objects across video frames
- Measure model accuracy using mAP, precision, and recall
- Measure inference speed using FPS
- Compare YOLOv8n and YOLOv8s if time allows
- Document results, failure cases, and limitations

---

## Hardware

Local machine:

- Intel i7 7th Gen
- NVIDIA GTX 1050 Ti

Used for:

- setup
- debugging
- inference
- FPS testing
- demo videos

Training:

- RWTH HPC GPU resources

Used for:

- full model training
- validation
- model comparison

---

## Dataset

Dataset: TBD

Target classes:

```text
0: car
1: pedestrian
2: cyclist


Expected dataset format:

data/
└── road_dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml

Example data.yaml:

path: data/road_dataset
train: images/train
val: images/val

names:
  0: car
  1: pedestrian
  2: cyclist
Tech Stack
Python
YOLOv8
PyTorch
OpenCV
ByteTrack
NumPy
Matplotlib
Project Structure
road-scene-yolov8-tracking/
├── data/
├── scripts/
├── models/
├── outputs/
│   ├── images/
│   ├── videos/
│   ├── plots/
│   └── failure_cases/
├── README.md
└── requirements.txt

Metrics to record:

mAP50
mAP50-95
Precision
Recall
F1 Score
Results
Model	mAP50	mAP50-95	Precision	Recall	F1 Score	FPS
YOLOv8n	TBD	TBD	TBD	TBD	TBD	TBD
YOLOv8s	TBD	TBD	TBD	TBD	TBD	TBD