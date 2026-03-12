#!/bin/bash
mkdir -p models/sam2/checkpoints
cd models/sam2/checkpoints
if [ -f "download_ckpts.sh" ]; then
    echo "Starting download of SAM 2 checkpoints..."
    bash download_ckpts.sh
else
    echo "download_ckpts.sh not found in models/sam2/checkpoints"
    # Fallback to wget if download_ckpts.sh is missing? 
    # Better to just fail and ask user to check directory
    exit 1
fi
cd ../../..
