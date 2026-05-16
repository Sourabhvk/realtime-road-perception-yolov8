# ML, Neural Networks, Computer Vision, and YOLOv8 Study Notes  
## Project: Real-Time Road Object Detection and Tracking using YOLOv8 + ByteTrack

**Purpose of this document:**  
This is a study note for everything we have touched so far in the road-scene YOLOv8 project. It explains the concepts behind the practical steps we already did: repo setup, virtual environment, pretrained inference, dataset folder structure, label visualization, YOLO-format annotations, training/validation, metrics, local vs HPC usage, and why YOLOv8 is built the way it is.

This is not a research paper. It is a practical learning document for understanding the project well enough to:
- continue implementation without blindly copy-pasting,
- explain the project in interviews,
- write a strong GitHub README,
- write honest CV bullets.

---

# 1. Project Context: What We Are Building

## 1.1 Project objective

We are building a road-scene perception pipeline:

```text
Input image/video
      ↓
YOLOv8 object detector
      ↓
Bounding boxes + class labels + confidence scores
      ↓
ByteTrack object tracker
      ↓
Tracked object IDs across video frames
      ↓
Metrics, FPS benchmark, demo video, README, CV bullets
```

Target classes:

```yaml
0: car
1: pedestrian
2: cyclist
```

The final project should show that we can:
- prepare a real dataset,
- fine-tune a YOLOv8 model,
- evaluate detection quality,
- run inference locally,
- generate a tracking demo,
- benchmark speed,
- document limitations.

## 1.2 Current project stage

So far, we have mainly worked on the **foundation and dataset sanity-check stage**.

We have done or discussed:

```text
1. GitHub repo created and linked.
2. .gitignore prepared.
3. Python virtual environment created.
4. PyTorch / Ultralytics / OpenCV installed.
5. GPU detection checked on local GTX 1050 Ti.
6. requirements.txt generated.
7. YOLO-style project structure started.
8. Dataset folder structure planned:
   data/road_dataset/images/train
   data/road_dataset/images/val
   data/road_dataset/labels/train
   data/road_dataset/labels/val
9. data.yaml concept discussed.
10. Label visualization script created.
11. Output images with boxes were generated to verify labels.
12. Local vs RWTH HPC division discussed.
```

Important: the output folder with 5 images containing boxes is **not a model result yet**. It is a **dataset verification result**.

---

# 2. Big Picture: AI vs ML vs Deep Learning vs Computer Vision

## 2.1 Artificial Intelligence

Artificial Intelligence is the broad field of making machines perform tasks that normally require human-like intelligence.

Examples:
- recognizing objects,
- planning robot motion,
- translating language,
- predicting failures,
- making decisions from sensor data.

Our project is AI because it gives a system the ability to understand a road scene.

## 2.2 Machine Learning

Machine Learning is a subset of AI where the system learns patterns from data instead of being manually programmed with every rule.

Traditional programming:

```text
Rules + input data → output
```

Machine learning:

```text
Input data + correct outputs → learned rules/model
```

For our project:

```text
Images + labels → trained YOLOv8 model
```

The model learns patterns like:
- cars usually have rectangular shapes,
- pedestrians are vertical objects,
- cyclists often include a person + bicycle structure,
- objects farther away are smaller,
- occlusions make detection harder.

## 2.3 Deep Learning

Deep Learning is machine learning using neural networks with many layers.

Why deep learning matters for computer vision:
- raw pixels are messy,
- useful features are hierarchical,
- early layers detect simple patterns,
- deeper layers detect more meaningful objects.

Example hierarchy:

```text
Pixels
  ↓
Edges
  ↓
Corners / textures
  ↓
Wheels / windows / legs
  ↓
Cars / pedestrians / cyclists
```

YOLOv8 is a deep learning model.

## 2.4 Computer Vision

Computer Vision is the field of making computers understand visual data.

Common CV tasks:

| Task | Meaning | Example |
|---|---|---|
| Image classification | One label for whole image | "This image contains a car" |
| Object detection | Boxes + labels for objects | car at x/y/w/h |
| Semantic segmentation | Class for every pixel | road/pavement/car pixels |
| Instance segmentation | Pixel mask per object | exact car shape |
| Object tracking | Same object over video frames | car ID 7 across frames |
| Pose estimation | Keypoints | human skeleton |

Our project combines:

```text
Object detection + object tracking
```

YOLOv8 handles detection. ByteTrack handles tracking.

---

# 3. What a Model Actually Learns

## 3.1 Model

A model is a mathematical function with adjustable parameters.

For our detector:

```text
image → model → boxes, class labels, confidence scores
```

In simplified form:

```text
prediction = f(image, weights)
```

The **weights** are the learned parameters.

Before training, weights may be:
- random,
- pretrained on a large dataset like COCO.

After training/fine-tuning, weights should become better for our road-scene classes.

## 3.2 Training

Training means repeatedly showing data to the model, comparing predictions with ground truth labels, and adjusting weights.

Simplified loop:

```text
for each epoch:
    load batch of images
    predict boxes/classes
    compare prediction with ground truth
    calculate loss
    update weights
```

## 3.3 Loss

Loss measures how wrong the model is.

For object detection, loss is not just one thing. It usually includes:
- bounding box localization error,
- classification error,
- confidence/objectness-related error depending on architecture.

In simple language:

```text
Loss = how bad the predicted boxes/classes are compared to labels
```

Training tries to reduce loss.

## 3.4 Optimizer

The optimizer updates model weights to reduce loss.

Common optimizers:
- SGD,
- Adam,
- AdamW.

YOLO training handles this internally unless we customize it.

## 3.5 Epoch

One epoch means the model has seen the whole training dataset once.

Example:

```text
epochs=50
```

means the model goes through the training set 50 times.

For local debugging, we only use something tiny like:

```text
epochs=1
```

This is not for final accuracy. It only checks that the pipeline works.

## 3.6 Batch size

Batch size is the number of images processed before one weight update.

Example:

```text
batch=8
```

means 8 images are passed together.

Bigger batch:
- uses more GPU memory,
- may train faster per epoch,
- may be impossible on small GPUs.

Our GTX 1050 Ti is weak for full training, so local batch size must stay small.

---

# 4. Train, Validation, and Test

## 4.1 Training set

The training set is used to update the model weights.

```text
data/road_dataset/images/train
data/road_dataset/labels/train
```

The model learns from this data.

## 4.2 Validation set

The validation set is used to evaluate the model during/after training.

```text
data/road_dataset/images/val
data/road_dataset/labels/val
```

The model does **not** directly learn from validation images. We use validation to check whether training generalizes.

## 4.3 Why not train on everything?

If we train and evaluate on the same data, we may fool ourselves.

The model might memorize the training examples but fail on new images.

That is called **overfitting**.

## 4.4 Typical split

A common split is:

```text
80% train
20% val
```

For the project, exact split is less important than:
- no broken labels,
- correct class mapping,
- enough examples per class,
- validation images being representative.

---

# 5. Dataset Format: What YOLO Expects

## 5.1 Target folder structure

Our target structure:

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

Each image should have a matching label file.

Example:

```text
images/train/000123.png
labels/train/000123.txt
```

## 5.2 YOLO label file format

Each object is one row:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.5123 0.6341 0.2210 0.1844
```

This means:

```text
class_id = 0 → car
x_center = 0.5123
y_center = 0.6341
width = 0.2210
height = 0.1844
```

Important: YOLO coordinates are **normalized**.

That means values are between 0 and 1.

They are relative to image width and height, not absolute pixels.

## 5.3 Why normalized coordinates?

Because images may have different sizes.

If the model used raw pixel positions, labels would depend heavily on image resolution.

Normalized labels work across image sizes.

Example:

```text
x_pixel = x_center_normalized * image_width
y_pixel = y_center_normalized * image_height
box_width_pixel = width_normalized * image_width
box_height_pixel = height_normalized * image_height
```

That is exactly what our visualization script did.

## 5.4 data.yaml

`data.yaml` tells YOLO where the dataset is and what the classes are.

Example:

```yaml
path: data/road_dataset
train: images/train
val: images/val

names:
  0: car
  1: pedestrian
  2: cyclist
```

Without this file, YOLO does not know:
- where images are,
- where labels are,
- how many classes exist,
- what class ID means what.

---

# 6. The 5 Output Images With Boxes: What Was Happening?

We wrote a script that:
1. takes a few images from `images/train`,
2. finds the matching `.txt` label files,
3. reads YOLO-format labels,
4. converts normalized coordinates into pixel coordinates,
5. draws bounding boxes,
6. saves the result to `outputs/images`.

This stage is called:

```text
dataset visualization
```

or

```text
annotation sanity check
```

It is used **before training**.

## 6.1 Why this matters

Bad labels destroy training.

Before training for hours on HPC, we must confirm:
- boxes appear around the correct objects,
- class names match the object,
- no boxes are shifted,
- no wrong coordinate conversion,
- no missing labels,
- no class ID mismatch.

## 6.2 What good output means

If the 5 output images show boxes correctly around cars/pedestrians/cyclists, it means:

```text
Image loading works.
Label loading works.
Class IDs are readable.
YOLO normalized coordinates are being interpreted correctly.
Folder paths are likely correct.
```

It does **not** yet mean:

```text
The model is trained.
The model is accurate.
The project is complete.
```

This is a pipeline sanity check.

---

# 7. Neural Networks: Practical Understanding

## 7.1 What is a neural network?

A neural network is a stack of mathematical operations.

Each layer transforms the input into a more useful representation.

For images:

```text
raw image tensor
  ↓
convolution layers
  ↓
feature maps
  ↓
deeper semantic features
  ↓
detection head
  ↓
boxes/classes/confidences
```

## 7.2 Tensor

A tensor is a multi-dimensional array.

An image can be represented as:

```text
height × width × channels
```

Example RGB image:

```text
640 × 640 × 3
```

Deep learning frameworks often use:

```text
batch × channels × height × width
```

Example:

```text
8 × 3 × 640 × 640
```

This means:
- 8 images,
- 3 color channels,
- each 640 by 640.

## 7.3 Parameters / weights

Weights are numbers inside the network that get updated during training.

Pretrained YOLOv8 weights already contain useful visual knowledge.

Custom fine-tuning adjusts those weights to our dataset.

## 7.4 Activation functions

Activation functions add non-linearity.

Without non-linearity, many layers would behave like one big linear function.

Common activations:
- ReLU,
- SiLU,
- sigmoid,
- softmax.

YOLO uses modern activation choices internally. We usually do not need to touch them for this project.

## 7.5 Backpropagation

Backpropagation calculates how each weight contributed to the error.

Then the optimizer updates weights.

Simplified:

```text
prediction wrong → calculate loss → calculate gradients → update weights
```

You do not need to manually implement backpropagation in this project. PyTorch handles it.

---

# 8. CNNs: Why They Matter for Images

## 8.1 Fully connected layers are bad for raw images

A normal fully connected neural network would treat every pixel independently.

That is inefficient because images have spatial structure:
- nearby pixels are related,
- edges matter,
- local patterns repeat.

CNNs solve this.

## 8.2 Convolution

A convolution uses small filters/kernels that slide across an image.

Example kernel size:

```text
3 × 3
```

A filter can learn to detect:
- horizontal edges,
- vertical edges,
- corners,
- textures,
- object parts.

## 8.3 Feature maps

After convolution, the output is a feature map.

A feature map answers:

```text
Where in the image does this learned pattern appear?
```

Early feature maps may detect edges. Later feature maps may detect object parts.

## 8.4 Stride

Stride controls how far the kernel moves.

Stride 1:
- more detailed,
- more computation.

Stride 2:
- reduces spatial size,
- faster,
- loses some fine detail.

YOLO uses downsampling because detecting objects at high speed requires efficiency.

## 8.5 Receptive field

The receptive field is the area of the original image that influences one feature value.

Deep layers have larger receptive fields.

This matters because:
- small receptive field sees local details,
- large receptive field sees object-level context.

A car cannot be recognized from only one pixel. The model needs enough surrounding context.

---

# 9. Object Detection Concepts

## 9.1 Classification vs detection

Classification:

```text
Image → "car"
```

Detection:

```text
Image → car at box A, pedestrian at box B, cyclist at box C
```

Detection is harder because the model must answer:

```text
What object?
Where exactly?
How confident?
```

## 9.2 Bounding box

A bounding box is a rectangle around an object.

Two common representations:

```text
x_min, y_min, x_max, y_max
```

or

```text
x_center, y_center, width, height
```

YOLO labels use:

```text
x_center, y_center, width, height
```

normalized to image size.

## 9.3 Ground truth

Ground truth means the correct annotation.

For example:

```text
This car is at this box with class_id 0.
```

Training compares predictions against ground truth.

## 9.4 Prediction output

A detector outputs something like:

```text
class: car
confidence: 0.87
box: [x1, y1, x2, y2]
```

Confidence means how sure the model is.

## 9.5 Confidence threshold

A confidence threshold filters weak predictions.

Example:

```text
conf=0.25
```

means predictions below 0.25 confidence are ignored.

Higher confidence threshold:
- fewer false positives,
- more missed objects.

Lower confidence threshold:
- more detections,
- more false positives.

This trade-off matters for road scenes.

## 9.6 IoU

Intersection over Union measures overlap between predicted box and ground truth box.

```text
IoU = area of overlap / area of union
```

Example:

```text
IoU = 1.0 → perfect overlap
IoU = 0.0 → no overlap
```

IoU is used to decide whether a detection counts as correct.

## 9.7 Non-Maximum Suppression

Object detectors can produce multiple boxes around the same object.

Non-Maximum Suppression, or NMS, removes duplicate boxes.

Simplified:
1. keep the highest-confidence box,
2. suppress overlapping lower-confidence boxes,
3. repeat.

Without NMS, one car could appear as many detections.

---

# 10. Metrics We Need for the Project

## 10.1 Precision

Precision answers:

```text
Of all predicted objects, how many were correct?
```

Formula:

```text
Precision = TP / (TP + FP)
```

Where:
- TP = true positives,
- FP = false positives.

High precision means fewer false alarms.

For road detection:
- high precision means the model does not hallucinate cars/pedestrians/cyclists too often.

## 10.2 Recall

Recall answers:

```text
Of all real objects, how many did the model find?
```

Formula:

```text
Recall = TP / (TP + FN)
```

Where:
- FN = false negatives.

High recall means fewer missed objects.

For road detection:
- missing pedestrians is bad,
- missing small cyclists is also bad.

## 10.3 Precision vs recall trade-off

Usually:

```text
Higher confidence threshold → higher precision, lower recall
Lower confidence threshold → lower precision, higher recall
```

This is why one number alone is not enough.

## 10.4 F1 score

F1 balances precision and recall.

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

Use F1 when you want one balanced score.

## 10.5 mAP50

mAP50 means mean Average Precision at IoU threshold 0.50.

It is more forgiving.

A prediction is counted as correct if:

```text
IoU >= 0.50
```

mAP50 is useful but can look optimistic.

## 10.6 mAP50-95

mAP50-95 averages mAP across IoU thresholds:

```text
0.50, 0.55, 0.60, ..., 0.95
```

This is stricter.

It rewards not only detecting the object, but localizing it accurately.

For our results table, we need:

```text
mAP50
mAP50-95
Precision
Recall
F1 Score
Local FPS
```

## 10.7 FPS

FPS means frames per second.

For videos:

```text
higher FPS = faster inference
```

For real-time perception, FPS matters because the system must react quickly.

Our local FPS benchmark on GTX 1050 Ti is important because it shows deployment-oriented performance, not just HPC training performance.

---

# 11. YOLO: The Core Idea

## 11.1 What YOLO means

YOLO means:

```text
You Only Look Once
```

The idea: detect objects in one forward pass through the network.

Older two-stage detectors first generated proposals, then classified them.

YOLO is a one-stage detector:
- faster,
- simpler for real-time use,
- good fit for robotics demos.

## 11.2 Why YOLO is suitable for this project

Our project cares about:
- road scenes,
- real-time perception,
- local inference,
- tracking video,
- speed vs accuracy trade-off.

YOLO is practical because:
- easy training API,
- pretrained models available,
- strong speed,
- direct object detection outputs,
- built-in validation and tracking support through Ultralytics.

## 11.3 YOLO model sizes

YOLOv8 has multiple sizes:

```text
YOLOv8n  = nano
YOLOv8s  = small
YOLOv8m  = medium
YOLOv8l  = large
YOLOv8x  = extra large
```

For our project:

```text
YOLOv8n = minimum success
YOLOv8s = nice-to-have comparison
```

Why:
- YOLOv8n is fast and light,
- YOLOv8s may be more accurate but slower,
- comparing them gives a clear accuracy-latency trade-off.

This is recruiter-readable and practical.

---

# 12. YOLOv8 Architecture: Backbone, Neck, Head

YOLOv8 can be understood in three main parts:

```text
Backbone → Neck → Head
```

## 12.1 Backbone

The backbone extracts visual features from the image.

Input:

```text
raw image
```

Output:

```text
multi-level feature maps
```

The backbone learns:
- edges,
- textures,
- shapes,
- object parts,
- higher-level visual patterns.

For our road project, the backbone learns features useful for:
- car shapes,
- pedestrian silhouettes,
- bicycle wheels,
- road context,
- object scale.

## 12.2 Neck

The neck combines features from different scales.

Why scales matter:

```text
small pedestrian far away → needs fine features
large nearby car → needs large/context features
```

The neck helps the model detect objects of different sizes.

YOLO-style necks often use feature pyramid ideas:
- combine high-resolution shallow features,
- combine low-resolution deep semantic features.

In practical terms:

```text
Neck = feature mixing across scales
```

## 12.3 Head

The head produces final predictions.

For detection, the head predicts:
- bounding box location,
- class probabilities,
- confidence-related scores.

YOLOv8 uses a more modern detection head design than older YOLO versions.

The head is the part most directly tied to our target classes.

If we change from COCO 80 classes to our 3 classes, the model must adapt the detection head.

---

# 13. C2f in YOLOv8: Important Deeper Concept

## 13.1 What C2f is

C2f stands for:

```text
Cross-Stage Partial bottleneck with two convolutions
```

It is a building block used inside YOLOv8.

It replaced older C3-style blocks used in YOLOv5-like architectures.

## 13.2 Why C2f exists

A good object detector needs:
- strong feature extraction,
- efficient computation,
- good gradient flow during training.

C2f is designed to improve feature learning while keeping the model lightweight.

In plain language:

```text
C2f helps the model learn richer image features without becoming too heavy.
```

## 13.3 What problem does it solve?

Deep CNNs can struggle with:
- losing information through many layers,
- inefficient computation,
- weak gradient flow,
- too many parameters.

C2f helps by:
- splitting feature paths,
- processing part of the features through bottlenecks,
- concatenating features,
- preserving useful information,
- improving gradient flow.

## 13.4 Simplified C2f flow

A simplified view:

```text
Input feature map
      ↓
Initial convolution
      ↓
Split feature channels
      ↓
Some channels go through bottleneck layers
      ↓
Intermediate outputs are kept
      ↓
Concatenate features
      ↓
Final convolution
      ↓
Output feature map
```

The key idea is not just "more layers".

The key idea is:

```text
reuse and combine features efficiently
```

## 13.5 Why splitting channels helps

Instead of pushing all information through the same heavy path, C2f lets some information move more directly while another part is transformed.

This gives:
- better information preservation,
- lower computation,
- stronger gradient paths.

That helps training stability and efficiency.

## 13.6 What "gradient flow" means here

During training, gradients tell each layer how to update.

If gradients become weak or noisy in deep networks, early layers may learn poorly.

C2f creates multiple paths through the block, making it easier for training signals to flow backward.

This is why architecture details matter.

## 13.7 What C2f means for our project

We do not need to implement C2f manually.

But we should understand it because:
- it is part of why YOLOv8 is efficient,
- it explains why YOLOv8n can still work reasonably well,
- it helps us discuss architecture in interviews,
- it helps us avoid treating YOLO as a black box.

For our project report, one clean sentence is enough:

```text
YOLOv8 uses C2f blocks for efficient feature extraction, improving gradient flow and multi-scale feature representation while keeping the model lightweight enough for real-time inference.
```

## 13.8 What not to claim

Do not claim:

```text
I invented C2f.
I modified YOLOv8 architecture.
I created a novel detector.
```

Honest claim:

```text
I fine-tuned YOLOv8, understood its architecture, and evaluated accuracy/speed trade-offs for road-scene detection.
```

That is enough for a CV project.

---

# 14. Pretrained Weights: Why We Used Them

## 14.1 What pretrained weights are

Pretrained weights are model parameters learned from a large dataset before our project starts.

For YOLOv8, common pretrained weights are trained on COCO-like object detection data.

Example:

```text
yolov8n.pt
yolov8s.pt
```

These files already contain learned visual features.

## 14.2 Why not start from random weights?

Training from scratch needs:
- lots of data,
- long training time,
- more GPU resources,
- careful hyperparameter tuning.

Our project is not trying to build a model from zero.

We want a practical perception pipeline.

So pretrained weights are the correct choice.

## 14.3 What pretrained weights already know

A pretrained detector has already learned:
- edges,
- textures,
- object shapes,
- common objects,
- visual hierarchy,
- bounding box localization behavior.

Even if our final classes are only:

```text
car, pedestrian, cyclist
```

the pretrained backbone already knows useful visual features.

## 14.4 Important nuance: COCO vs our classes

COCO includes classes like:
- person,
- car,
- bicycle,
- bus,
- truck,
- motorcycle.

Our project uses:
- car,
- pedestrian,
- cyclist.

There is overlap, but not perfect.

Example:
- `pedestrian` is close to COCO `person`,
- `car` is close to COCO `car`,
- `cyclist` is not exactly one COCO class; it is often a person + bicycle combination.

This means pretrained weights help, but fine-tuning is still necessary.

## 14.5 Why we tested pretrained inference early

Running pretrained YOLO early checks:

```text
Ultralytics installed correctly.
PyTorch works.
GPU/CPU inference works.
Image loading works.
Prediction output format is understandable.
OpenCV/display/export flow works.
```

This is before custom training.

It is a sanity test.

If pretrained inference fails, custom training will also fail.

## 14.6 Pretrained inference vs fine-tuned inference

Pretrained inference:

```text
model = YOLO("yolov8n.pt")
```

The model predicts COCO classes.

Fine-tuned inference:

```text
model = YOLO("runs/detect/train/weights/best.pt")
```

The model predicts our custom classes.

Final README should use results from fine-tuned model, not only pretrained model.

---

# 15. Fine-Tuning

## 15.1 What fine-tuning means

Fine-tuning means starting from pretrained weights and continuing training on our custom dataset.

Instead of:

```text
random model → train from zero
```

we do:

```text
pretrained YOLOv8 → train on road dataset → road-specific detector
```

## 15.2 Why fine-tuning is practical

Fine-tuning:
- trains faster,
- needs less data,
- usually gives better results,
- is realistic for student projects,
- is standard practice in applied CV.

## 15.3 What changes during fine-tuning?

The model adjusts:
- feature extraction patterns,
- object localization,
- class-specific detection behavior,
- final detection head for the target classes.

If `data.yaml` defines 3 classes, YOLO trains for those 3 classes.

## 15.4 What we should save after fine-tuning

Important outputs:

```text
runs/detect/train/weights/best.pt
runs/detect/train/weights/last.pt
runs/detect/train/results.csv
runs/detect/train/results.png
runs/detect/train/confusion_matrix.png
runs/detect/train/PR_curve.png
runs/detect/train/F1_curve.png
```

For the repo, we may rename/copy:

```text
models/yolov8n_road_best.pt
outputs/plots/
```

But large model files may or may not be pushed to GitHub depending on size. Usually:
- code and README go to Git,
- very large weights may go to release assets, Google Drive, or be excluded.

---

# 16. Local Laptop vs RWTH HPC

## 16.1 Local laptop role

Local hardware:

```text
i7 7th Gen + NVIDIA GTX 1050 Ti
```

Use locally for:
- setup,
- package testing,
- dataset inspection,
- label visualization,
- tiny debug training,
- inference,
- tracking demo generation,
- FPS benchmark.

Do not use local laptop for full heavy training unless the run is intentionally tiny.

## 16.2 HPC role

RWTH CLAIX / HPC should be used for:
- full YOLOv8n training,
- validation,
- YOLOv8s comparison if time allows,
- longer experiments.

Reason:
- HPC has much stronger GPUs,
- training is faster,
- avoids overloading local machine,
- makes the project more realistic.

## 16.3 Why Git matters for HPC

Git helps sync code.

Workflow:

```text
local machine:
    write scripts
    test small
    git add/commit/push

HPC:
    git clone/pull repo
    install environment
    run training jobs
    save metrics/weights

local machine:
    pull results or copy outputs
    run inference/tracking/FPS
    update README
```

Git should not be treated as dataset storage.

Usually do not push:
- large datasets,
- training runs,
- huge weight files,
- cache folders,
- video outputs unless small.

## 16.4 Why local FPS still matters

Training on HPC gives accuracy.

But local FPS tells us whether the model is realistic for lightweight deployment.

For CV/recruiter value, this is strong:

```text
Trained on HPC, benchmarked locally on GTX 1050 Ti.
```

It shows we understand accuracy-latency trade-offs.

---

# 17. Virtual Environment and Dependencies

## 17.1 What the virtual environment does

A virtual environment isolates Python packages for this project.

Instead of installing everything globally, we install inside:

```text
.venv
```

This avoids conflicts with other Python projects.

## 17.2 Why activating venv matters

When venv is active, commands like:

```powershell
python
pip
```

refer to the project environment.

Expected terminal sign:

```text
(.venv)
```

If venv is not active, packages may install in the wrong Python environment.

## 17.3 Where packages install

If the venv lives here:

```text
S:\project\yolo\.venv
```

then packages installed while the venv is active go inside that `.venv`.

This helps keep packages on the S drive instead of filling the C drive.

## 17.4 requirements.txt

`requirements.txt` is a snapshot of installed packages.

Command:

```powershell
pip freeze > requirements.txt
```

This is not automatic unless we run the command.

It helps recreate the environment:

```powershell
pip install -r requirements.txt
```

For HPC, requirements help install similar dependencies.

---

# 18. KITTI and Why It Is Famous

## 18.1 What KITTI is

KITTI is a well-known autonomous driving / road-scene dataset.

It is famous because it includes real driving scenes and has been widely used for:
- object detection,
- tracking,
- stereo vision,
- optical flow,
- odometry,
- autonomous driving benchmarks.

## 18.2 Why KITTI fits this project

Our target classes match road-scene perception:

```text
car
pedestrian
cyclist
```

KITTI has these kinds of objects.

This makes it more relevant than a random generic image dataset.

## 18.3 KITTI format vs YOLO format

KITTI labels are not originally in YOLO format.

KITTI annotations include object information in a different text format.

So we need a conversion step:

```text
KITTI labels → YOLO labels
```

That is why we planned:

```text
scripts/convert_kitti_to_yolo.py
```

## 18.4 Why conversion must be verified

Conversion bugs are common.

Possible issues:
- wrong class mapping,
- wrong coordinate normalization,
- wrong image dimensions,
- boxes shifted,
- train/val split mismatch,
- missing label files.

That is exactly why the 5-image visualization step matters.

---

# 19. YOLO Training Command: What It Means

A typical training command may look like:

```powershell
yolo detect train model=yolov8n.pt data=data/road_dataset/data.yaml epochs=50 imgsz=640 batch=8 device=0
```

Breakdown:

```text
yolo
```
Runs Ultralytics CLI.

```text
detect train
```
Use object detection training mode.

```text
model=yolov8n.pt
```
Start from YOLOv8n pretrained weights.

```text
data=data/road_dataset/data.yaml
```
Use our dataset configuration.

```text
epochs=50
```
Train for 50 passes through the dataset.

```text
imgsz=640
```
Resize images to 640×640 for training.

```text
batch=8
```
Process 8 images per batch.

```text
device=0
```
Use GPU 0.

For tiny local debug:

```powershell
yolo detect train model=yolov8n.pt data=data/road_dataset/data.yaml epochs=1 imgsz=320 batch=2 device=0
```

This only checks that training starts and finishes.

It is not the final model.

---

# 20. Validation Command: What It Means

Validation evaluates the trained model on the validation set.

Example:

```powershell
yolo detect val model=runs/detect/train/weights/best.pt data=data/road_dataset/data.yaml imgsz=640 device=0
```

Validation gives metrics such as:
- precision,
- recall,
- mAP50,
- mAP50-95.

It may also generate plots:
- confusion matrix,
- PR curve,
- F1 curve.

These are important for the README.

---

# 21. Inference: What Happens When We Test on Images

Inference means using a trained model to make predictions.

Example:

```powershell
yolo detect predict model=models/yolov8n_road_best.pt source=sample.jpg conf=0.25
```

Output:
- image with boxes,
- predicted classes,
- confidences.

Inference does not update weights.

It is the model "using" what it learned.

## 21.1 Pretrained inference

Uses general COCO model:

```text
yolov8n.pt
```

Good for testing installation.

## 21.2 Custom inference

Uses our trained model:

```text
best.pt
```

Good for final project outputs.

---

# 22. Tracking: YOLOv8 + ByteTrack

## 22.1 Detection vs tracking

Detection:

```text
Frame 1: car
Frame 2: car
Frame 3: car
```

Tracking:

```text
Frame 1: car ID 1
Frame 2: car ID 1
Frame 3: car ID 1
```

Tracking assigns consistent identities across frames.

## 22.2 What ByteTrack does

ByteTrack is a multi-object tracker.

It takes detections from YOLO and links them across frames.

Inputs:
- boxes,
- confidence scores,
- class labels.

Outputs:
- boxes,
- class labels,
- track IDs.

## 22.3 Why ByteTrack is useful

For road-scene perception, tracking helps answer:
- Is this the same car as before?
- How long has this pedestrian been visible?
- How many unique cyclists appear?
- Is an object moving across the scene?

## 22.4 Important nuance

ByteTrack does not detect objects by itself.

It depends on detector quality.

Pipeline:

```text
YOLOv8 detects objects
ByteTrack associates detections over time
```

If YOLO misses an object for many frames, tracking can fail.

## 22.5 Ultralytics tracking command

Example:

```powershell
yolo track model=models/yolov8n_road_best.pt source=inputs/video.mp4 tracker=bytetrack.yaml conf=0.25
```

This should generate a video with:
- bounding boxes,
- class names,
- track IDs.

That video is a key README/GitHub demo artifact.

---

# 23. YOLOv8n vs YOLOv8s

## 23.1 YOLOv8n

YOLOv8n is the smallest model.

Pros:
- fastest,
- low memory,
- better for weak GPUs,
- good for real-time demos.

Cons:
- lower accuracy,
- struggles more with small/occluded objects.

## 23.2 YOLOv8s

YOLOv8s is larger.

Pros:
- usually better accuracy,
- better feature capacity,
- may detect small/harder objects better.

Cons:
- slower,
- more GPU memory,
- less suitable for weak hardware.

## 23.3 Why comparing them is valuable

A comparison gives a clean engineering story:

```text
YOLOv8n: faster, lower accuracy
YOLOv8s: slower, higher accuracy
```

This is the kind of trade-off robotics/CV teams care about.

Final results table:

```markdown
| Model | mAP50 | mAP50-95 | Precision | Recall | F1 Score | Local FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLOv8s | TBD | TBD | TBD | TBD | TBD | TBD |
```

---

# 24. Common Failure Cases in Road Detection

We should intentionally document failure cases.

This makes the project stronger and more honest.

## 24.1 Occlusion

Objects partly hidden by other objects.

Example:
- pedestrian behind car,
- cyclist partly behind traffic sign.

Effect:
- missed detection,
- inaccurate box,
- track ID switch.

## 24.2 Small objects

Far-away pedestrians or cyclists occupy few pixels.

Effect:
- low confidence,
- missed detection,
- poor localization.

## 24.3 Class confusion

Cyclist may be confused with:
- pedestrian,
- bicycle,
- motorcycle.

Reason:
- cyclist is visually complex,
- body + bicycle may be partly visible,
- class boundary is harder.

## 24.4 Lighting and weather

Night, glare, rain, shadows, low contrast.

Effect:
- lower confidence,
- false positives,
- missed objects.

## 24.5 Dataset bias

If training data is mostly clear daytime road scenes, model may struggle with:
- night scenes,
- unusual camera angles,
- rare vehicles,
- dense crowds.

This should go into README limitations.

---

# 25. Why We Do Tiny Debug Runs

A tiny debug run is not for accuracy.

It checks the pipeline.

Example:

```powershell
yolo detect train model=yolov8n.pt data=data/road_dataset/data.yaml epochs=1 imgsz=320 batch=2
```

This verifies:
- dataset paths work,
- labels load,
- classes are recognized,
- training loop runs,
- GPU works,
- no format errors,
- output folder is created.

Only after this should we run full training on HPC.

This avoids wasting HPC time on basic path/label errors.

---

# 26. What "Good Engineering" Means in This Project

This project is not novel research.

That is okay.

For internships, execution quality matters.

Good engineering means:
- clean repo,
- reproducible setup,
- clear scripts,
- correct dataset format,
- measurable metrics,
- honest limitations,
- demo video,
- readable README.

Bad project style:
- only screenshots,
- no metrics,
- no explanation,
- no reproducible commands,
- no failure analysis,
- vague claims like "AI-powered road safety system".

Good project style:
- "Fine-tuned YOLOv8n on road-scene data for 3 classes."
- "Evaluated mAP50, mAP50-95, precision, recall, F1."
- "Integrated ByteTrack for video tracking."
- "Benchmarked local FPS on GTX 1050 Ti."
- "Compared YOLOv8n and YOLOv8s accuracy-latency trade-off."

---

# 27. Important Files in Our Repo

Expected final structure:

```text
road-scene-yolov8-tracking/
├── data/
├── scripts/
│   ├── test_inference.py
│   ├── benchmark_fps.py
│   └── convert_kitti_to_yolo.py
├── models/
│   ├── yolov8n_road_best.pt
│   └── yolov8s_road_best.pt
├── outputs/
│   ├── images/
│   ├── videos/
│   ├── plots/
│   └── failure_cases/
├── notebooks/
├── README.md
└── requirements.txt
```

## 27.1 scripts/convert_kitti_to_yolo.py

Purpose:
- convert KITTI labels to YOLO labels,
- map KITTI classes to our class IDs,
- split data into train/val,
- save normalized labels.

## 27.2 scripts/test_inference.py

Purpose:
- load trained YOLO model,
- run prediction on images/videos,
- save outputs.

## 27.3 scripts/benchmark_fps.py

Purpose:
- measure inference speed,
- calculate FPS,
- compare YOLOv8n vs YOLOv8s.

## 27.4 outputs/images

Purpose:
- annotation visualizations,
- prediction screenshots,
- README images.

## 27.5 outputs/videos

Purpose:
- tracking demo videos.

## 27.6 outputs/failure_cases

Purpose:
- save examples where model fails,
- document limitations.

This section is underrated. Recruiters like honest analysis.

---

# 28. What We Should Be Able to Explain in an Interview

## 28.1 Simple project explanation

```text
I built a road-scene object detection and tracking pipeline using YOLOv8 and ByteTrack. I prepared the dataset in YOLO format, fine-tuned YOLOv8 for car, pedestrian, and cyclist detection, evaluated mAP/precision/recall/F1, benchmarked local FPS, and generated a tracking demo video.
```

## 28.2 Why YOLOv8?

```text
YOLOv8 is a practical one-stage detector with strong speed-accuracy trade-off, pretrained weights, simple training APIs, and built-in support for detection, validation, prediction, and tracking workflows.
```

## 28.3 Why pretrained weights?

```text
Starting from pretrained YOLOv8 weights gives the model useful visual features from large-scale data, reduces training time, and improves performance compared with training from scratch on a smaller road-scene dataset.
```

## 28.4 What did C2f do?

```text
C2f is a YOLOv8 feature extraction block that improves gradient flow and feature reuse using cross-stage partial connections and concatenation, helping the model remain efficient while learning rich visual features.
```

## 28.5 Why train on HPC and benchmark locally?

```text
HPC is used for efficient model training and validation, while the local GTX 1050 Ti benchmark shows realistic inference performance on limited hardware.
```

## 28.6 What are the limitations?

```text
The detector may struggle with small distant objects, occlusions, poor lighting, and cyclist/pedestrian confusion. Tracking quality also depends on detector stability across frames.
```

---

# 29. Practical Mental Model of the Full Pipeline

## 29.1 Data preparation

```text
raw dataset
  ↓
class filtering
  ↓
label conversion
  ↓
YOLO folder structure
  ↓
data.yaml
  ↓
visual label check
```

## 29.2 Training

```text
pretrained yolov8n.pt
  ↓
train on road_dataset
  ↓
best.pt
  ↓
validation metrics
```

## 29.3 Evaluation

```text
best.pt
  ↓
val set
  ↓
precision, recall, mAP50, mAP50-95
  ↓
plots and confusion matrix
```

## 29.4 Inference

```text
best.pt
  ↓
image/video
  ↓
boxes + classes + confidences
```

## 29.5 Tracking

```text
YOLO detections per frame
  ↓
ByteTrack association
  ↓
consistent object IDs
  ↓
demo video
```

## 29.6 Benchmarking

```text
trained model
  ↓
local inference loop
  ↓
average latency
  ↓
FPS
```

---

# 30. Things We Should Not Overcomplicate Yet

Avoid for now:
- custom YOLO architecture modification,
- training from scratch,
- multi-GPU distributed training,
- ROS integration,
- TensorRT optimization,
- complicated deployment pipelines,
- large hyperparameter sweeps.

Focus first on:
1. dataset ready,
2. labels verified,
3. YOLOv8n trained,
4. validation metrics collected,
5. local FPS measured,
6. ByteTrack video generated,
7. README completed.

That is enough to become CV-ready.

---

# 31. Glossary

## Annotation

A label describing object location and class in an image.

## Bounding box

Rectangle around an object.

## Class ID

Numeric class label, e.g.:

```text
0 = car
```

## Confidence

Model's certainty about a prediction.

## Dataset

Collection of images and labels.

## Epoch

One full pass through the training dataset.

## Fine-tuning

Training a pretrained model further on a custom dataset.

## Ground truth

Correct annotation used for training/evaluation.

## Inference

Using a trained model to make predictions.

## IoU

Overlap score between predicted and true box.

## mAP

Mean average precision, a standard object detection metric.

## NMS

Method to remove duplicate boxes.

## Pretrained weights

Model weights learned earlier on a large dataset.

## Validation

Evaluation on data not used for weight updates.

## YOLO format

Label format:

```text
class_id x_center y_center width height
```

with normalized coordinates.

---

# 32. What To Save for README/CV Later

Save these as we continue:

```text
1. Screenshot of correct annotation visualization.
2. Training command used on HPC.
3. Validation metrics table.
4. results.png from YOLO training.
5. confusion_matrix.png.
6. PR_curve.png.
7. F1_curve.png.
8. Example correct detections.
9. Example failure cases.
10. FPS benchmark output.
11. ByteTrack demo video.
12. Final model file names.
```

Final CV bullets can only be written honestly after metrics are collected.

Current honest status:

```text
Project setup and dataset verification stage in progress.
```

Final target status:

```text
Fine-tuned YOLOv8n/YOLOv8s on road-scene data, integrated ByteTrack tracking, and benchmarked detection metrics and local FPS.
```

---

# 33. Study Checklist

You understand the project well if you can answer these:

```text
1. What is the difference between detection and tracking?
2. Why do YOLO labels use normalized coordinates?
3. What does data.yaml do?
4. Why did we visualize 5 labeled images before training?
5. Why use pretrained weights?
6. What is fine-tuning?
7. What is the difference between train and val?
8. What are precision and recall?
9. Why is mAP50-95 stricter than mAP50?
10. What does C2f do in YOLOv8?
11. Why use HPC for training but local laptop for FPS?
12. What failure cases should we document?
```

If you can answer these in your own words, you understand the project at a good practical level.

---

# 34. One-Sentence Summary

```text
We are building a practical road-scene perception pipeline by preparing a YOLO-format dataset, fine-tuning YOLOv8 for car/pedestrian/cyclist detection, validating detection quality with standard metrics, integrating ByteTrack for object tracking, and benchmarking local FPS to show real-world accuracy-speed trade-offs.
```

---

# 35. Source Basis Used for These Notes

These notes are based on:
- our project setup decisions and previous debugging steps,
- Ultralytics YOLO documentation for training, validation, datasets, tracking, and metrics,
- Ultralytics YOLOv8 architecture documentation and comparisons,
- standard deep learning and computer vision concepts.

Important nuance:
- YOLOv8 does not have one single formal architecture paper from Ultralytics. For practical implementation details, Ultralytics documentation and GitHub are the main references.
