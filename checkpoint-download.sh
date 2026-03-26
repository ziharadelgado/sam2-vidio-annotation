#!/bin/bash
# Download SAM 3 checkpoint from HuggingFace
# Requires: HuggingFace access approved + huggingface-cli login

CHECKPOINT_DIR="/home/zihara_delgado_uri_edu/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

echo "Checking for SAM 3 checkpoint..."

if [ -f "$CHECKPOINT_DIR/sam3.pt" ]; then
    echo "✓ sam3.pt already exists at $CHECKPOINT_DIR"
else
    echo "Downloading SAM 3 checkpoint..."
    echo "NOTE: You must have HuggingFace access approved first."
    echo "      Run 'huggingface-cli login' if not authenticated."
    echo ""

    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/sam3',
    filename='sam3.pt',
    local_dir='$CHECKPOINT_DIR'
)
print('✓ SAM 3 checkpoint downloaded!')
"

    if [ $? -ne 0 ]; then
        echo "❌ Download failed. Check your HuggingFace access."
        exit 1
    fi
fi

echo ""
echo "Checkpoint directory contents:"
ls -lh "$CHECKPOINT_DIR"
