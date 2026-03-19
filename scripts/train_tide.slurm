#!/bin/bash
#SBATCH --job-name=train_tide_eval
#SBATCH --output=logs/train_tide_%j.out
#SBATCH --error=logs/train_tide_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# 1. Environment Setup
module purge
# Load any necessary modules here if required by your HPC (e.g., CUDA)
# module load CUDA/12.1

# Ensure log directory exists
mkdir -p logs

# 2. Activate Environment
# Assuming your environment is in ../trainerenv
cd /home/kamron_aggor_uri_edu/ocean_detect/trainer/sam2-vidio-annotation
source ../trainerenv/bin/activate

# 3. Install/Update Dependencies
echo "Ensuring dependencies are installed..."
pip install --upgrade pip
pip install ultralytics roboflow tidecv --quiet

# 4. Run Training
# This uses your existing train.py
# Note: Ensure train.py is configured with your Roboflow API key correctly.
echo "Starting Training at $(date)"
python3 train.py --model yolov11 --epochs 100 --batch 16

# 5. Prepare for TIDE Evaluation
# We need the COCO ground truth JSON.
# We'll run a quick python snippet to download it if it doesn't exist.
echo "Downloading COCO annotations for TIDE..."
python3 -c "
from roboflow import Roboflow
import os
rf = Roboflow(api_key='mvFKQaefMTrBAiSy194g')
project = rf.workspace('DeepSeaObjectDetection').project('Fish-Detection-v5')
version = project.version(1)
if not os.path.exists('./dataset/coco'):
    version.download(model_format='coco', location='./dataset/coco')
"

# 6. Run TIDE Evaluation
# Find the best model from the training run
# Ultralytics saves runs in ./runs/yolov11_underwater/weights/best.pt by default in your train.py
MODEL_PATH="./runs/yolov11_underwater/weights/best.pt"
DATA_YAML="./dataset/data.yaml"
GT_JSON="./dataset/coco/valid/_annotations.coco.json"
TIDE_OUT="./tide_results_$(date +%Y%m%d_%H%M%S)"

if [ -f "$MODEL_PATH" ]; then
    echo "Starting TIDE Evaluation at $(date)"
    python3 eval_tide.py \
        --model "$MODEL_PATH" \
        --data "$DATA_YAML" \
        --gt "$GT_JSON" \
        --out "$TIDE_OUT" \
        --type yolov11
else
    echo "ERROR: Model not found at $MODEL_PATH. Training might have failed."
    exit 1
fi

echo "Job finished at $(date)"
echo "TIDE results are available in $TIDE_OUT"
