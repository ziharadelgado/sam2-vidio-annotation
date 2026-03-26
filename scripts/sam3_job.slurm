#!/bin/bash
#SBATCH --job-name=sam3_shark_annotation
#SBATCH --output=logs/sam3_%j.out
#SBATCH --error=logs/sam3_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="l40s"          # L40S has 48GB VRAM — enough for SAM 3 (3.5GB)
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=12:00:00

# 1. Environment Setup
module purge
module load cuda/12.6
module load conda/latest
conda activate sam3

# Ensure log directory exists
mkdir -p logs

# 2. Sync Queue from GDrive
echo "Syncing queue from Google Drive..."
local_queue="/home/zihara_delgado_uri_edu/annotated-video/queue"
mkdir -p "$local_queue"

timeout 120s rclone copy gdrive:DeepSea_ObjectDetection/rclone/queue/ "$local_queue" -v
if [ $? -ne 0 ]; then
    echo "ERROR: Rclone sync failed or timed out."
    exit 1
fi

# 3. Enter project directory
cd /home/zihara_delgado_uri_edu/sam3-vidio-annotation

# 4. Install/verify dependencies
pip install ultralytics roboflow --quiet

# 5. Execute Processing
echo "Starting SAM 3 + YOLO Processing at $(date)"
python3 -u main.py
echo "Job finished at $(date)"
