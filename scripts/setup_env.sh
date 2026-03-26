#!/bin/bash
#SBATCH --job-name=sam3_setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# 1. Load modules
module purge
module load cuda/12.6
module load conda/latest

# 2. Create conda environment
echo "Creating SAM 3 conda environment..."
conda create -n sam3 python=3.12 -y
conda activate sam3

# 3. Install PyTorch with CUDA 12.6
echo "Installing PyTorch..."
pip install torch==2.7.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# 4. Clone and install SAM 3
echo "Setting up SAM 3..."
SAM3_DIR="/home/zihara_delgado_uri_edu/sam3"
if [ ! -d "$SAM3_DIR" ]; then
    cd /home/zihara_delgado_uri_edu/
    git clone https://github.com/facebookresearch/sam3.git
fi

cd "$SAM3_DIR"
pip install -e .

# 5. Fix dependencies
pip install "setuptools<70"
pip install einops psutil

# 6. Install other requirements
echo "Installing pipeline requirements..."
cd /home/zihara_delgado_uri_edu/sam3-vidio-annotation
pip install -r requirements.txt

# 7. Verify
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
from sam3.model_builder import build_sam3_video_predictor
print('SAM 3 import: OK')
from ultralytics import YOLO
print('YOLO import: OK')
from roboflow import Roboflow
print('Roboflow import: OK')
"

echo ""
echo "Setup complete at $(date)"
