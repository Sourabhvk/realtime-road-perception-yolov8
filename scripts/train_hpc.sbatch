#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --chdir=/hpcwork/niy86040/road-scene-yolov8
#SBATCH --output=/hpcwork/niy86040/road-scene-yolov8/runs/slurm/%x-%j.out
#SBATCH --error=/hpcwork/niy86040/road-scene-yolov8/runs/slurm/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sourabh.kulkarni@rwth-aachen.de
#SBATCH --signal=B:USR1@300

set -Eeuo pipefail

PROJECT_ROOT="/hpcwork/niy86040/road-scene-yolov8"
LOG_DIR="$PROJECT_ROOT/runs/slurm"
RUNS_DIR="$PROJECT_ROOT/runs"
CACHE_DIR="$PROJECT_ROOT/.cache"

on_timeout_warning() {
    echo "Received Slurm timeout warning signal at $(date). YOLO checkpoints in $RUNS_DIR should be used to resume if the job is killed."
}

on_error() {
    status=$?
    echo "Job failed with exit code $status at $(date)."
    echo "Working directory: $(pwd)"
    echo "Host: $(hostname)"
    df -h "$PROJECT_ROOT" || true
    nvidia-smi || true
    exit "$status"
}

trap on_timeout_warning USR1
trap on_error ERR

cd "$PROJECT_ROOT"

mkdir -p "$LOG_DIR" "$RUNS_DIR" "$CACHE_DIR/matplotlib" "$CACHE_DIR/xdg" "$CACHE_DIR/ultralytics"
touch "$LOG_DIR/.write_test" "$RUNS_DIR/.write_test"
rm -f "$LOG_DIR/.write_test" "$RUNS_DIR/.write_test"

export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export YOLO_CONFIG_DIR="$CACHE_DIR/ultralytics"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

if [ ! -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    echo "Missing virtual environment: $PROJECT_ROOT/.venv"
    exit 2
fi

source "$PROJECT_ROOT/.venv/bin/activate"

MODEL="${MODEL:-yolov8n.pt}"
DATA="${DATA:-data/road_dataset/data.yaml}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
RUN_NAME="${RUN_NAME:-yolov8n_road}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
SAVE_PERIOD="${SAVE_PERIOD:-5}"
COS_LR="${COS_LR:-False}"
LR0="${LR0:-0.01}"
LRF="${LRF:-0.01}"

resolve_path() {
    case "$1" in
        /*) printf "%s\n" "$1" ;;
        *) printf "%s\n" "$PROJECT_ROOT/$1" ;;
    esac
}

MODEL_PATH="$(resolve_path "$MODEL")"
DATA_PATH="$(resolve_path "$DATA")"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Missing model file: $MODEL_PATH"
    echo "Copy the model into the repo before submitting; compute nodes may not have internet access."
    exit 2
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "Missing data config: $DATA_PATH"
    exit 2
fi

for dir in \
    "$PROJECT_ROOT/data/road_dataset/images/train" \
    "$PROJECT_ROOT/data/road_dataset/images/val" \
    "$PROJECT_ROOT/data/road_dataset/labels/train" \
    "$PROJECT_ROOT/data/road_dataset/labels/val"
do
    if [ ! -d "$dir" ]; then
        echo "Missing dataset directory: $dir"
        exit 2
    fi
done

train_images=$(find "$PROJECT_ROOT/data/road_dataset/images/train" -type f | wc -l)
val_images=$(find "$PROJECT_ROOT/data/road_dataset/images/val" -type f | wc -l)
train_labels=$(find "$PROJECT_ROOT/data/road_dataset/labels/train" -type f | wc -l)
val_labels=$(find "$PROJECT_ROOT/data/road_dataset/labels/val" -type f | wc -l)

echo "Dataset counts: train_images=$train_images val_images=$val_images train_labels=$train_labels val_labels=$val_labels"
echo "Training config: model=$MODEL_PATH epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH workers=$WORKERS cos_lr=$COS_LR lr0=$LR0 lrf=$LRF run_name=$RUN_NAME"

if [ "$train_images" -eq 0 ] || [ "$val_images" -eq 0 ] || [ "$train_images" -ne "$train_labels" ] || [ "$val_images" -ne "$val_labels" ]; then
    echo "Dataset image/label counts are invalid."
    exit 2
fi

python -c "import torch, ultralytics; print('torch cuda:', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 2)"
yolo version
nvidia-smi

yolo detect train \
    model="$MODEL_PATH" \
    data="$DATA_PATH" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    device=0 \
    workers="$WORKERS" \
    save_period="$SAVE_PERIOD" \
    cos_lr="$COS_LR" \
    lr0="$LR0" \
    lrf="$LRF" \
    project="$RUNS_DIR" \
    name="$RUN_NAME"
