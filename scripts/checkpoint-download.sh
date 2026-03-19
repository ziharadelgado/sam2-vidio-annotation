#!/bin/bash

# Ensure we are in the project root
PROJECT_ROOT="/home/kamron_aggor_uri_edu/ocean_detect/trainer/sam2-vidio-annotation"
cd "$PROJECT_ROOT"

# Check for setup.py to ensure the repo is actually there and not just an empty folder
if [ ! -f "models/sam2/setup.py" ]; then
    echo "SAM 2 repository missing or incomplete. Cloning..."
    rm -rf models/sam2  # Clean up any empty/partial directory
    mkdir -p models
    cd models
    git clone https://github.com/facebookresearch/sam2.git
    cd ..
fi

CHECKPOINT_DIR="models/sam2/checkpoints"
mkdir -p "$CHECKPOINT_DIR"
cd "$CHECKPOINT_DIR"

echo "Checking for checkpoints in $PWD..."
# Use wget with -nc (no-clobber) to skip already downloaded files
wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget -nc https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

echo "Checkpoint check complete."
cd "$PROJECT_ROOT"
