#!/bin/bash
#SBATCH --job-name=sam2_shark_annotation
#SBATCH --output=logs/sam2_%j.out
#SBATCH --error=logs/sam2_%j.err
#SBATCH --partition=gpu              # Common name, check 'sinfo' for your cluster
#SBATCH --gpus=1                     # Request 1 GPU
#SBATCH --constraint="vram32"        # Request at least 32GB VRAM (Cluster specific)
#SBATCH --cpus-per-task=8            # 8 cores to prevent GPU bottlenecking
#SBATCH --mem=60G                    # 60GB System RAM to handle large image sets
#SBATCH --time=12:00:00              # 12 hour limit

# 1. Environment Setup
module purge
# Python 3.11.1 is available by default, no module load needed
# module load CUDA/12.1              # Load CUDA if required by your HPC

# Ensure log directory exists
mkdir -p logs

# 2. Sync Queue from GDrive
echo "Syncing queue from Google Drive..."
local_queue="/home/kamron_aggor_uri_edu/annotated-video/queue"
mkdir -p "$local_queue"

# Use a 2-minute timeout and verbose output
timeout 120s rclone copy gdrive:DeepSea_ObjectDetection/rclone/queue/ "$local_queue" -v

if [ $? -ne 0 ]; then
    echo "ERROR: Rclone sync failed or timed out. Check connection or 'rclone config'."
    exit 1
fi

# 3. Enter project directory and activate environment
cd /home/kamron_aggor_uri_edu/ocean_detect/trainer/sam2-vidio-annotation
source ../trainerenv/bin/activate

# 4. Final verification of requirements
pip install -r requirements.txt --quiet  # Ensure environment is synced

# 5. Execute Processing (Unbuffered to see logs live)
echo "Starting SAM 2.1 Processing at $(date)"
python3 -u main.py

echo "Job finished at $(date)"