#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --output=runs/slurm/%x-%j.out
#SBATCH --error=runs/slurm/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    # Reuse the project environment when it exists on the cluster.
    source ".venv/bin/activate"
fi

mkdir -p runs/slurm

MODEL="${MODEL:-yolov8n.pt}"
DATA="${DATA:-data/road_dataset/data.yaml}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
RUN_NAME="${RUN_NAME:-yolov8n_road}"

yolo detect train \
    model="$MODEL" \
    data="$DATA" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    device=0 \
    project=runs \
    name="$RUN_NAME"