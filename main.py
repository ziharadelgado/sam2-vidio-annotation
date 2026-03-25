import os
import subprocess
from pathlib import Path
from annotate import SharkAnnotator

def main():
    # Step 1: Download the checkpoint if it doesn't exist
    checkpoint_path = Path("models/sam2/checkpoints/sam2.1_hiera_large.pt")
    if not checkpoint_path.exists():
        print("Downloading SAM 2.1 checkpoint...")
        try:
            subprocess.run(["bash", "scripts/checkpoint-download.sh"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download checkpoint: {e}")
            # If standard download fails, maybe it's already in home?
            home_checkpoint = Path.home() / "checkpoints/sam2.1_hiera_large.pt"
            if home_checkpoint.exists():
                print(f"Using checkpoint from {home_checkpoint}")
                checkpoint_path = home_checkpoint
            else:
                print("Could not find checkpoint.")
                return

    # Step 2: Initialize the annotator
    # We use dynamic paths relative to home directory or current directory
    work_dir = Path.home() / "annotated-video"
    gdrive_queue = "gdrive:DeepSea_ObjectDetection/rclone/queue/"
    
    annotator = SharkAnnotator(
        checkpoint_path=str(checkpoint_path),
        config_path="configs/sam2.1/sam2.1_hiera_l.yaml",
        work_dir=str(work_dir)
    )

    # Step 3: Setup model
    try:
        annotator.setup_model()
    except Exception as e:
        print(f"Error setting up model: {e}")
        return

    # Step 4: Download and process the queue
    queue_dir = annotator.download_queue(gdrive_queue)
    if not queue_dir:
        raise RuntimeError(f"CRITICAL ERROR: Failed to download queue from {gdrive_queue}. The remote must be accessible to proceed.")

    # Search for videos or image directories
    # 1. Look for .mp4 files (original behavior)
    video_files = []
    for root, dirs, files in os.walk(queue_dir):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))

    # 2. Look for COCO JSON files in directories (new behavior)
    coco_dirs = []
    for root, dirs, files in os.walk(queue_dir):
        if "_annotations.coco.json" in files:
            coco_dirs.append(root)

    if not video_files and not coco_dirs:
        print("No videos or COCO image directories found in the queue.")
        return

    # Process videos
    for video_path in video_files:
        video_file = os.path.basename(video_path)
        json_file = video_file.replace(".mp4", ".json")
        json_path = os.path.join(os.path.dirname(video_path), json_file)
        
        if os.path.exists(json_path):
            print(f"\n--- Processing video {video_file} ---")
            try:
                annotator.process_video(video_path, json_path)
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
        else:
            print(f"\n--- Skipping {video_file} (no JSON annotation found) ---")

    # Process COCO directories
    for coco_dir in coco_dirs:
        json_path = os.path.join(coco_dir, "_annotations.coco.json")
        print(f"\n--- Processing COCO directory {os.path.basename(coco_dir)} ---")
        try:
            annotator.process_images(coco_dir, json_path)
        except Exception as e:
            print(f"Error processing COCO directory {coco_dir}: {e}")

    print("\n✅ All processing complete!")
    print(f"Results are in: {work_dir / 'exports'}")

if __name__ == "__main__":
    main()
