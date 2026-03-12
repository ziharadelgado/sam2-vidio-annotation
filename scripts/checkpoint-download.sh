#!/bin/bash
mkdir -p models/sam2/checkpoints
cd models/sam2/checkpoints
if [ -f "download_ckpts.sh" ]; then
    bash download_ckpts.sh
else
    echo "download_ckpts.sh not found in models/sam2/checkpoints"
    exit 1
fi
cd ../../..
