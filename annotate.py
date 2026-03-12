# Automation of shark_annotation_all_in_one (2).ipynb for HPC
import os
import subprocess
import time
import argparse
import sys
import json
import zipfile
import shutil
import pathlib
import torch
import numpy as np
import random
import gc
import cv2
from PIL import Image
from tqdm import tqdm

# Add SAM2 to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR = os.path.join(SCRIPT_DIR, "models", "sam2")
if SAM2_DIR not in sys.path:
    sys.path.insert(0, SAM2_DIR)

try:
    from sam2.build_sam import build_sam2_video_predictor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False

class SharkAnnotator:
    def __init__(self, 
                 checkpoint_path, 
                 config_path="configs/sam2.1/sam2.1_hiera_l.yaml", 
                 device=None,
                 work_dir=None):
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.config_path = config_path
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.work_dir = os.path.abspath(work_dir or os.path.expanduser("~/annotated-video"))
        os.makedirs(self.work_dir, exist_ok=True)
        
        self.predictor = None
        self.inference_state = None
        self.seed = 42
        self._setup_seeds()

    def _setup_seeds(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def setup_model(self):
        if not HAS_SAM2:
            raise ImportError("SAM 2 not found. Please ensure it's in models/sam2 and dependencies are installed.")
        
        print(f"Loading SAM 2 model from {self.checkpoint_path}...")
        # SAM 2 predictor needs to be initialized in its own directory often due to hydra configs
        original_cwd = os.getcwd()
        try:
            os.chdir(SAM2_DIR)
            self.predictor = build_sam2_video_predictor(
                self.config_path,
                self.checkpoint_path,
                device=self.device
            )
            print("✓ Model loaded successfully")
        finally:
            os.chdir(original_cwd)

    def download_queue(self, gdrive_source="kamgdrive:DeepSea_ObjectDetection/rclone/queue/"):
        print(f"Downloading queue from {gdrive_source}...")
        local_queue = os.path.join(self.work_dir, "queue")
        os.makedirs(local_queue, exist_ok=True)
        
        try:
            subprocess.run(["rclone", "copy", gdrive_source, local_queue], check=True)
            print(f"✓ Queue downloaded to {local_queue}")
            return local_queue
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download queue: {e}")
            return None

    def extract_frames(self, video_path, target_fps=30, target_res=(1024, 576)):
        video_name = pathlib.Path(video_path).stem
        frames_dir = os.path.join(self.work_dir, "frames", video_name)
        os.makedirs(frames_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, int(fps / target_fps))
        
        print(f"Extracting frames from {video_name}...")
        count = 0
        saved_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % interval == 0:
                resized = cv2.resize(frame, target_res, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(frames_dir, f"{saved_count:05d}.jpg"), resized)
                saved_count += 1
            count += 1
        cap.release()
        print(f"✓ Extracted {saved_count} frames to {frames_dir}")
        return frames_dir, saved_count

    def load_bounding_boxes(self, json_path):
        """
        Load bounding boxes from a JSON file. 
        Expects COCO format or a simple list of objects with 'bbox' and 'frame_idx'.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Simple heuristic: if it's COCO, extract bbox and frame_idx
        prompts = []
        if isinstance(data, dict) and "annotations" in data:
            # COCO format
            for ann in data["annotations"]:
                bbox = ann["bbox"] # [x, y, w, h]
                # Convert [x, y, w, h] to [x1, y1, x2, y2]
                box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                image_id = ann.get("image_id", 0)
                prompts.append({"box": box, "frame_idx": image_id, "obj_id": ann.get("category_id", 1)})
        elif isinstance(data, list):
            # Simple list format
            prompts = data
            
        return prompts

    def clean_mask(self, mask):
        mask_uint8 = (mask * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = (labels == largest_label).astype(np.uint8)
        else:
            cleaned = mask_uint8
        return cleaned.astype(bool)

    def process_video(self, video_path, json_path):
        video_name = pathlib.Path(video_path).stem
        frames_dir, num_frames = self.extract_frames(video_path)
        prompts = self.load_bounding_boxes(json_path)
        
        if not prompts:
            print(f"⚠️ No prompts found for {video_name}, skipping.")
            return
        
        print(f"Initializing SAM 2 state for {video_name}...")
        inference_state = self.predictor.init_state(
            video_path=frames_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True
        )
        
        # Add bounding box prompts
        for prompt in prompts:
            frame_idx = prompt["frame_idx"]
            obj_id = prompt["obj_id"]
            box = prompt["box"] # [x1, y1, x2, y2]
            
            print(f"Adding box prompt on frame {frame_idx}: {box}")
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=box
            )

        # Propagate
        print("Propagating masks through video...")
        video_segments = {}
        batch_size = 100
        processed_count = 0
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: self.clean_mask((out_mask_logits[i] > 0.0).cpu().numpy().squeeze())
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            processed_count += 1
            if processed_count % batch_size == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # Export
        self.export_coco(video_name, video_segments, frames_dir, num_frames)
        self.create_comparison_video(video_name, video_segments, frames_dir, num_frames)
        
        # Cleanup
        self.predictor.reset_state(inference_state)
        torch.cuda.empty_cache()
        gc.collect()

    def export_coco(self, video_name, video_segments, frames_dir, num_frames):
        export_dir = os.path.join(self.work_dir, "exports", video_name)
        os.makedirs(os.path.join(export_dir, "images"), exist_ok=True)
        
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "shark"}]
        }
        
        ann_id = 1
        for frame_idx in range(num_frames):
            frame_file = f"{frame_idx:05d}.jpg"
            src = os.path.join(frames_dir, frame_file)
            dst = os.path.join(export_dir, "images", frame_file)
            shutil.copy(src, dst)
            
            img = cv2.imread(src)
            h, w = img.shape[:2]
            coco["images"].append({
                "id": frame_idx,
                "file_name": frame_file,
                "width": w,
                "height": h
            })
            
            if frame_idx in video_segments:
                for obj_id, mask in video_segments[frame_idx].items():
                    # Convert mask to polygon
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours: continue
                    largest = max(contours, key=cv2.contourArea)
                    epsilon = 0.002 * cv2.arcLength(largest, True)
                    approx = cv2.approxPolyDP(largest, epsilon, True)
                    polygon = approx.flatten().tolist()
                    
                    if len(polygon) < 6: continue
                    
                    x_coords = polygon[0::2]
                    y_coords = polygon[1::2]
                    bbox = [min(x_coords), min(y_coords), max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)]
                    
                    coco["annotations"].append({
                        "id": ann_id,
                        "image_id": frame_idx,
                        "category_id": obj_id,
                        "segmentation": [polygon],
                        "bbox": bbox,
                        "area": cv2.contourArea(approx),
                        "iscrowd": 0
                    })
                    ann_id += 1
        
        with open(os.path.join(export_dir, "_annotations.coco.json"), 'w') as f:
            json.dump(coco, f)
        print(f"✓ Exported COCO for {video_name} to {export_dir}")

    def create_comparison_video(self, video_name, video_segments, frames_dir, num_frames):
        output_path = os.path.join(self.work_dir, "exports", f"{video_name}_comparison.mp4")
        sample = cv2.imread(os.path.join(frames_dir, "00000.jpg"))
        h, w = sample.shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10, (w * 2, h))
        
        for frame_idx in range(num_frames):
            frame = cv2.imread(os.path.join(frames_dir, f"{frame_idx:05d}.jpg"))
            annotated = frame.copy()
            
            if frame_idx in video_segments:
                for obj_id, mask in video_segments[frame_idx].items():
                    annotated[mask] = (annotated[mask] * 0.6).astype(np.uint8) + np.array([0, 0, 150], dtype=np.uint8)
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
            
            combined = np.hstack([frame, annotated])
            out.write(combined)
        out.release()
        print(f"✓ Comparison video created: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="SAM 2.1 Shark Video Annotator")
    parser.add_argument("--checkpoint", type=str, help="Path to SAM 2 checkpoint")
    parser.add_argument("--config", type=str, default="sam2_hiera_l.yaml", help="Model config name")
    parser.add_argument("--work-dir", type=str, help="Working directory for frames and exports")
    parser.add_argument("--gdrive-queue", type=str, default="gdrive:DeepSea_ObjectDetection/rclone/queue/", help="GDrive queue path")
    
    args = parser.parse_args()
    
    # Auto-resolve checkpoint if not provided
    checkpoint = args.checkpoint
    if not checkpoint:
        # Check standard locations or run download script
        checkpoint = os.path.expanduser("~/checkpoints/sam2_hiera_large.pt")
        if not os.path.exists(checkpoint):
             print("Checkpoint not found. Please provide --checkpoint or ensure it exists at ~/checkpoints/sam2_hiera_large.pt")
             return

    annotator = SharkAnnotator(
        checkpoint_path=checkpoint,
        config_path=args.config,
        work_dir=args.work_dir
    )
    
    annotator.setup_model()
    queue_dir = annotator.download_queue(args.gdrive_queue)
    if not queue_dir:
        return
    
    # Process each video in the queue
    # Assuming the queue has .mp4 files and corresponding .json files
    video_files = [f for f in os.listdir(queue_dir) if f.endswith(".mp4")]
    for video_file in video_files:
        video_path = os.path.join(queue_dir, video_file)
        json_file = video_file.replace(".mp4", ".json")
        json_path = os.path.join(queue_dir, json_file)
        
        if os.path.exists(json_path):
            print(f"\n--- Processing {video_file} ---")
            annotator.process_video(video_path, json_path)
        else:
            print(f"\n--- Skipping {video_file} (no JSON annotation found) ---")

if __name__ == "__main__":
    main()
