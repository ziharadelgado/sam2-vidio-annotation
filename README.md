# SAM 3 Video Annotation Pipeline
## URI AI Lab — Deep-Sea Object Detection

Automated shark video annotation using **SAM 3** + **YOLO26n-seg (SharkDetect-V4-seg)**.

Replaces the previous SAM 2 bounding-box workflow. No manual bounding boxes needed —
just drop a video in the queue on Google Drive and the pipeline handles everything.

## What Changed from SAM 2

| | SAM 2 (old) | SAM 3 (new) |
|---|---|---|
| Prompt | Manual bounding box per shark | Text: "shark" + auto exemplar |
| Sharks per prompt | One | All instances automatically |
| Re-entry | Fails | Detector re-finds from scratch |
| Empty frames | Hallucinated masks | Presence Token → no output |
| First pass | None | YOLO26n-seg finds best exemplar |
| Upload | Manual | Auto-upload to Roboflow |

## Pipeline Flow

```
Google Drive queue → rclone sync
    ↓
YOLO26n-seg (SharkDetect-V4-seg) first pass
    → finds sharks, picks best detection as exemplar
    ↓
SAM 3 (text "shark" + YOLO exemplar)
    → segments ALL sharks across ALL frames
    → handles re-entry, multi-instance, empty frames
    ↓
Export YOLO segmentation format + COCO JSON
    ↓
Auto-upload to Roboflow
```

## Quick Start

```bash
# 1. Sync and run (same as before)
bash scripts/sync_and_run.sh

# 2. Or submit directly
sbatch scripts/sam3_job.slurm

# 3. Check logs
tail -f logs/sam3_*.out
```

## Setup

Requires the `sam3` conda environment on Unity:
```bash
module load conda/latest cuda/12.6
conda activate sam3
pip install -r requirements.txt
```

## Files

```
sam3-vidio-annotation/
├── main.py              # Entry point (same structure as Kamron's)
├── annotate.py          # SharkAnnotator class (SAM 3 + YOLO)
├── train.py             # Train SharkDetect-V4-seg (YOLO26n-seg)
├── requirements.txt
├── .gitignore
└── scripts/
    ├── sam3_job.slurm   # SLURM job script
    └── sync_and_run.sh  # Sync GDrive + submit job
```
