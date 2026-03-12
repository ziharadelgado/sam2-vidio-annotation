import os
import subprocess
from pathlib import Path
from annotate import SharkAnnotator

def main():
    # Step 1: Download the checkpoint if it doesn't exist
    checkpoint_path = Path("models/sam2/checkpoints/sam2_hiera_large.pt")
    if not checkpoint_path.exists():
        print("Downloading SAM 2.1 checkpoint...")
        try:
            subprocess.run(["bash", "scripts/checkpoint-download.sh"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to download checkpoint: {e}")
            # If standard download fails, maybe it's already in home?
            home_checkpoint = Path.home() / "checkpoints/sam2_hiera_large.pt"
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
        config_path="sam2_hiera_l.yaml",
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
        print("Queue download failed or empty.")
        return

    # Process each video in the queue
    video_files = [f for f in os.listdir(queue_dir) if f.endswith(".mp4")]
    if not video_files:
        print("No videos found in the queue.")
        return

    for video_file in video_files:
        video_path = os.path.join(queue_dir, video_file)
        json_file = video_file.replace(".mp4", ".json")
        json_path = os.path.join(queue_dir, json_file)
        
        if os.path.exists(json_path):
            print(f"\n--- Processing {video_file} ---")
            try:
                annotator.process_video(video_path, json_path)
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
        else:
            print(f"\n--- Skipping {video_file} (no JSON annotation found) ---")

    print("\n✅ All processing complete!")
    print(f"Results are in: {work_dir / 'exports'}")

if __name__ == "__main__":
    main()
