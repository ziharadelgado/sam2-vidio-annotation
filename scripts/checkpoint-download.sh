#!/bin/bash

# Ensure we are in the project root
# Navigate to the directory where models/ should be
PROJECT_ROOT=$(pwd)

if [ ! -d "models/sam2" ]; then
    echo "Cloning SAM 2 repository..."
    mkdir -p models
    cd models
    git clone https://github.com/facebookresearch/sam2.git
    cd ..
fi

CHECKPOINT_DIR="models/sam2/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

cd "$CHECKPOINT_DIR"

if [ -f "download_ckpts.sh" ]; then
    echo "Starting download of SAM 2 checkpoints..."
    bash download_ckpts.sh
else
    echo "download_ckpts.sh not found in $CHECKPOINT_DIR"
    echo "Attempting to download manually via wget..."
    wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
    wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
    wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
    wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    # Also for SAM 2.1
    wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
    wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
    wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
    wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi

cd "$PROJECT_ROOT"
