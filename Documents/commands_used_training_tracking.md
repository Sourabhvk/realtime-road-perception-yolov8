# Commands Used For Training And Tracking

Minimal command reference for this project.

## SSH To RWTH CLAIX

```powershell
ssh niy86040@login23-2.hpc.itc.rwth-aachen.de
```

Fallback:

```powershell
ssh niy86040@login23-g-1.hpc.itc.rwth-aachen.de
```

## HPC Project Folder

```bash
cd /hpcwork/niy86040/road-scene-yolov8
```

## Check Dataset On HPC

```bash
find data/road_dataset/images/train -type f | wc -l
find data/road_dataset/images/val -type f | wc -l
find data/road_dataset/labels/train -type f | wc -l
find data/road_dataset/labels/val -type f | wc -l
```

Expected:

```text
5984
1497
5984
1497
```

## Check HPC Usage

```bash
r_wlm_usage
```

Completed job accounting:

```bash
sacct -u niy86040 --starttime 2026-05-20 --format=JobID,JobName,State,Elapsed,AllocCPUS,ReqTRES%50
```

## Submit YOLOv8n Baseline

```bash
BATCH=16 sbatch scripts/train_hpc.sbatch
```

Note: this worked for the first baseline run, but later environment variables did not pass reliably through Slurm this way. Prefer `--wrap` for custom experiment settings.

## Submit YOLOv8s Experiments Safely

YOLOv8s default, batch 16:

```bash
sbatch --job-name=yolov8s_b16 --output=/hpcwork/niy86040/road-scene-yolov8/runs/slurm/%x-%j.out --error=/hpcwork/niy86040/road-scene-yolov8/runs/slurm/%x-%j.err --time=12:00:00 --partition=c23g --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:1 --chdir=/hpcwork/niy86040/road-scene-yolov8 --wrap="MODEL=yolov8s.pt RUN_NAME=yolov8s_b16_default EPOCHS=50 BATCH=16 COS_LR=False LR0=0.01 LRF=0.01 bash scripts/train_hpc.sbatch"
```

YOLOv8s batch 32 with cosine LR:

```bash
sbatch --job-name=yolov8s_b32 --output=/hpcwork/niy86040/road-scene-yolov8/runs/slurm/%x-%j.out --error=/hpcwork/niy86040/road-scene-yolov8/runs/slurm/%x-%j.err --time=12:00:00 --partition=c23g --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:1 --chdir=/hpcwork/niy86040/road-scene-yolov8 --wrap="MODEL=yolov8s.pt RUN_NAME=yolov8s_b32_cos EPOCHS=50 BATCH=32 COS_LR=True LR0=0.01 LRF=0.001 bash scripts/train_hpc.sbatch"
```

## Check Queue And Start Time

```bash
squeue -u niy86040
squeue --start -j JOB_ID
```

Multiple jobs:

```bash
squeue --start -j 542061,542062
```

## Check Slurm Logs

```bash
tail -60 runs/slurm/yolo_train-JOB_ID.out
cat runs/slurm/yolo_train-JOB_ID.err
```

For wrapped v8s jobs:

```bash
tail -60 runs/slurm/yolov8s_b16-JOB_ID.out
tail -60 runs/slurm/yolov8s_b32-JOB_ID.out
```

Verify actual training config after job starts:

```bash
grep -H "Training config" runs/slurm/yolov8s_b16-JOB_ID.out
grep -H "Training config" runs/slurm/yolov8s_b32-JOB_ID.out
```

## Download Runs From HPC

From local PowerShell.

YOLOv8n baseline:

```powershell
scp -r niy86040@login23-2.hpc.itc.rwth-aachen.de:/hpcwork/niy86040/road-scene-yolov8/runs/yolov8n_road "S:\Project\yolo\runs\"
```

YOLOv8s default:

```powershell
scp -r niy86040@login23-2.hpc.itc.rwth-aachen.de:/hpcwork/niy86040/road-scene-yolov8/runs/yolov8s_b16_default "S:\Project\yolo\runs\"
```

YOLOv8s batch 32 cosine:

```powershell
scp -r niy86040@login23-2.hpc.itc.rwth-aachen.de:/hpcwork/niy86040/road-scene-yolov8/runs/yolov8s_b32_cos "S:\Project\yolo\runs\"
```

## Local Inference On Images

```powershell
.\.venv\Scripts\yolo.exe task=detect mode=predict model=models/HPC_R1_best.pt source=data/road_dataset/images/val save=True conf=0.25
```

## Local Webcam Test

```powershell
.\.venv\Scripts\yolo.exe task=detect mode=predict model=models/HPC_R1_best.pt source=0 show=True
```

## Local ByteTrack Video Tracking

```powershell
.\.venv\Scripts\yolo.exe track model=models/HPC_R1_best.pt source=data/Videos/demo_video.mp4 tracker=bytetrack.yaml save=True conf=0.25
```

Phone video:

```powershell
.\.venv\Scripts\yolo.exe track model=models/HPC_R1_best.pt source=data/Videos/RoonStrasse.mp4 tracker=bytetrack.yaml save=True conf=0.25
```

Tracking outputs are saved under:

```text
runs/detect/track
runs/detect/track-2
```

## Important Slurm Lesson

This did not reliably pass custom variables into the job:

```bash
MODEL=yolov8s.pt RUN_NAME=yolov8s_b16_default BATCH=16 sbatch scripts/train_hpc.sbatch
```

Use `sbatch --wrap="MODEL=... bash scripts/train_hpc.sbatch"` for custom settings.