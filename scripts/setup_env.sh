#!/bin/bash
#SBATCH --job-name=sam2_setup
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00

# 1. Activate Environment
cd /home/kamron_aggor_uri_edu/ocean_detect/trainer/sam2-vidio-annotation
source ../trainerenv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch with CUDA 12.1 (The bulk of the download)
echo "Downloading and installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install SAM 2 and other requirements
echo "Installing SAM 2 dependencies..."
pip install -e .
pip install -r requirements.txt

echo "Setup complete at $(date)"
