# Rewrite of Kamron's main.py for SAM 3 + YOLO26n-seg
# Same flow: download checkpoint → init annotator → download queue → process → export

import os
import subprocess
from pathlib import Path
from annotate import SharkAnnotator


# ──────────────────────────────────────────────
# CONFIG — edit these to match your setup
# ──────────────────────────────────────────────
HOME = Path("/home/zihara_delgado_uri_edu")

# Model paths
SAM3_CHECKPOINT = HOME / "checkpoints" / "sam3.pt"
YOLO_MODEL = HOME / "runs" / "SharkDetect-V4-seg" / "weights" / "best.pt"

# Working directories
WORK_DIR = HOME / "annotated-video"
GDRIVE_QUEUE = "gdrive:DeepSea_ObjectDetection/rclone/queue/"

# SAM 3 settings
TEXT_PROMPT = "shark"
THRESHOLD = 0.2
YOLO_CONF = 0.25
FRAME_RATE = 5

# Roboflow settings
ROBOFLOW_API_KEY = "mvFKQaefMTrBAiSy194g"
ROBOFLOW_WORKSPACE = "DeepSeaObjectDetection"
ROBOFLOW_PROJECT = "deep-sea-object-detection"
UPLOAD_TO_ROBOFLOW = True


def main():
    # ── Step 1: Check SAM 3 checkpoint ──
    if not SAM3_CHECKPOINT.exists():
        print(f"SAM 3 checkpoint not found at {SAM3_CHECKPOINT}")
        print("Make sure you've downloaded it from HuggingFace.")
        print("Run: python -c \"from huggingface_hub import hf_hub_download; "
              "hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt', "
              f"local_dir='{SAM3_CHECKPOINT.parent}')\"")
        return

    # ── Step 2: Check YOLO model (optional) ──
    yolo_path = str(YOLO_MODEL) if YOLO_MODEL.exists() else None
    if yolo_path:
        print(f"✓ YOLO model found: {YOLO_MODEL}")
    else:
        print(f"⚠ YOLO model not found at {YOLO_MODEL}")
        print("  Running in SAM 3-only mode (text prompt without exemplar)")

    # ── Step 3: Initialize annotator ──
    annotator = SharkAnnotator(
        sam3_checkpoint=str(SAM3_CHECKPOINT),
        yolo_model_path=yolo_path,
        work_dir=str(WORK_DIR),
        text_prompt=TEXT_PROMPT,
        threshold=THRESHOLD,
        yolo_conf=YOLO_CONF,
        frame_rate=FRAME_RATE,
    )

    # ── Step 4: Setup models ──
    try:
        annotator.setup_model()
    except Exception as e:
        print(f"Error setting up models: {e}")
        return

    # ── Step 5: Download queue from GDrive ──
    queue_dir = annotator.download_queue(GDRIVE_QUEUE)
    if not queue_dir:
        print(f"CRITICAL: Failed to download queue from {GDRIVE_QUEUE}")
        print("Make sure rclone is configured and the remote is accessible.")
        return

    # ── Step 6: Find videos and COCO directories ──
    # Same discovery logic as Kamron's
    video_files = []
    coco_dirs = []

    for root, dirs, files in os.walk(queue_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))
        if "_annotations.coco.json" in files:
            coco_dirs.append(root)

    if not video_files and not coco_dirs:
        print("No videos or COCO image directories found in the queue.")
        return

    print(f"\nFound {len(video_files)} video(s) and {len(coco_dirs)} COCO director(ies)")

    # ── Step 7: Process videos ──
    for video_path in video_files:
        video_file = os.path.basename(video_path)
        print(f"\n--- Processing video: {video_file} ---")

        try:
            coco_dir = annotator.process_video(video_path)

            # Upload to Roboflow
            if UPLOAD_TO_ROBOFLOW and coco_dir:
                annotator.upload_to_roboflow(
                    coco_dir,
                    api_key=ROBOFLOW_API_KEY,
                    workspace=ROBOFLOW_WORKSPACE,
                    project_name=ROBOFLOW_PROJECT
                )

        except Exception as e:
            print(f"Error processing {video_file}: {e}")
            import traceback
            traceback.print_exc()

    # ── Step 8: Process COCO directories (image-based, like Kamron's) ──
    for coco_dir in coco_dirs:
        json_path = os.path.join(coco_dir, "_annotations.coco.json")
        dir_name = os.path.basename(coco_dir)
        print(f"\n--- Processing COCO directory: {dir_name} ---")

        try:
            annotator.process_images(coco_dir, json_path)
        except Exception as e:
            print(f"Error processing {dir_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 50)
    print("  ✅ All processing complete!")
    print(f"  Results: {WORK_DIR / 'exports'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
