#!/bin/bash
# Helper script to sync data on login node and then submit the SLURM job
# Same pattern as Kamron's sync_and_run.sh

PROJECT_DIR="/home/zihara_delgado_uri_edu/sam3-vidio-annotation"
QUEUE_DIR="/home/zihara_delgado_uri_edu/annotated-video/queue"

echo "1. Syncing queue from Google Drive (Running on login node)..."
mkdir -p "$QUEUE_DIR"
rclone copy gdrive:DeepSea_ObjectDetection/rclone/queue/ "$QUEUE_DIR" -v

if [ $? -eq 0 ]; then
    echo "✓ Sync complete."
    echo "2. Submitting SLURM job..."
    sbatch "$PROJECT_DIR/scripts/sam3_job.slurm"
else
    echo "❌ Rclone sync failed. Job not submitted."
    exit 1
fi
