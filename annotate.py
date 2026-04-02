# Automation of shark_annotation_all_in_one (2).ipynb for HPC
import os
import subprocess
import time
import argparse
import sys
import json
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
    try:
        sys.path.insert(0, SAM2_DIR)
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
        self.export_dir = os.path.join(self.work_dir, "exports")
        os.makedirs(self.export_dir, exist_ok=True)

        self.predictor = None
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

    def download_queue(self, gdrive_source="gdrive:DeepSea_ObjectDetection/rclone/queue/"):
        # This is now handled in sync_and_run.sh, but kept for compatibility
        local_queue = os.path.join(self.work_dir, "queue")
        if os.path.exists(local_queue) and os.listdir(local_queue):
            print(f"✓ Using existing queue at {local_queue}")
            return local_queue

        print(f"Downloading queue from {gdrive_source}...")
        os.makedirs(local_queue, exist_ok=True)
        try:
            subprocess.run(["rclone", "copy", gdrive_source, local_queue], check=True)
            return local_queue
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to download queue: {e}")
            return None

    def clean_mask(self, mask):
        mask_uint8 = (mask * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = (labels == largest_label).astype(np.uint8)
        else:
            cleaned = mask_uint8
        return cleaned.astype(bool)

    def process_images(self, image_dir, coco_json_path):
    def _coco_bbox_to_xyxy(self, bbox):
        """
        Convert COCO bbox [x, y, w, h] into [x1, y1, x2, y2].
        SAM box prompts are expected in corner format.
        """
        x, y, w, h = bbox
        return [x, y, x + w, y + h]

    def _bbox_to_polygon(self, bbox):
        """
        Fallback segmentation polygon when SAM does not return a usable mask.
        Produces a rectangular polygon in COCO segmentation format.
        """
        x, y, w, h = bbox
        return [x, y, x + w, y, x + w, y + h, x, y + h]

    def _add_optional_text_support(self, inference_state, frame_idx, obj_id, text_prompt):
        """
        Optionally add semantic text support when the loaded predictor exposes
        a text-prompt API. This is best-effort and never replaces box prompts.
        """
        if not text_prompt:
            return

        # Different SAM builds can expose different method names for text prompting.
        # We try known patterns safely and continue if unavailable.
        text_methods = [
            "add_new_text",
            "add_new_text_prompt",
            "add_new_prompt",
        ]
        for method_name in text_methods:
            method = getattr(self.predictor, method_name, None)
            if callable(method):
                try:
                    method(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        text=text_prompt
                    )
                except TypeError:
                    # Some implementations may use positional args.
                    method(inference_state, frame_idx, obj_id, text_prompt)
                except Exception:
                    # Text support is optional; a failure here should not block bbox prompts.
                    pass
                return

    def process_images(self, image_dir, coco_json_path, text_prompt=None):
        """
        Process a directory of images using COCO annotations.
        Optimized for memory by processing image-by-image.
        """
        split_name = pathlib.Path(image_dir).name
        print(f"--- Processing split: {split_name} ---")

        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        # Group COCO annotations by image_id so we can process one image at a time.
        img_id_to_anns = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            img_id_to_anns.setdefault(img_id, []).append(ann)

        images_info = {img["id"]: img for img in coco_data["images"]}
        sorted_img_ids = sorted(images_info.keys())

        new_annotations = []
        ann_id = 1

        # We'll use a temporary directory for single-image "videos" to reuse the video predictor
        # as it's already built and configured.
        temp_dir = os.path.join(self.work_dir, "temp_single")

        for img_id in tqdm(sorted_img_ids, desc=f"Refining {split_name}"):
            img_anns = img_id_to_anns.get(img_id, [])
            if not img_anns:
                continue

            img_info = images_info[img_id]
            file_name = img_info["file_name"]
            src_path = os.path.join(image_dir, file_name)

            if not os.path.exists(src_path):
                # If image missing, still keep original annotations but we can't refine others
                for ann in img_anns:
                    new_ann = ann.copy()
                    new_ann["id"] = ann_id
                    new_annotations.append(new_ann)
                    ann_id += 1
                continue

            # Separate annotations into those with segmentation and those without
            to_process = []
            for ann in img_anns:
                if ann.get("segmentation") and len(ann["segmentation"]) > 0:
                    # Keep existing segmentation
                    new_ann = ann.copy()
                    new_ann["id"] = ann_id
                    new_annotations.append(new_ann)
                    ann_id += 1
                else:
                    # To be processed by SAM
                    to_process.append(ann)

            if not to_process:
                continue

            # Create temporary single-frame "video" for the ones that need refining
            if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            shutil.copy(src_path, os.path.join(temp_dir, "00000.jpg"))

            try:
                inference_state = self.predictor.init_state(video_path=temp_dir)

                for ann in to_process:
                    bbox = ann["bbox"] # [x, y, w, h]
                    box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                    # Use a unique obj_id for each annotation to track them
                    # We'll use the original category_id for SAM but we need to map results back
                    # Actually, SAM uses obj_id as an identifier.
                    # Main prompt source for COCO directories: human COCO bounding boxes.
                    # Convert [x, y, w, h] -> [x1, y1, x2, y2] for SAM.
                    bbox = ann["bbox"]
                    box = self._coco_bbox_to_xyxy(bbox)
                    # Use original annotation ID as SAM object ID to map masks back cleanly.
                    self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=ann["id"], # Use original ID as obj_id for mapping
                        obj_id=ann["id"],
                        box=box
                    )
                    # Optional semantic support. This is additive and never text-only.
                    self._add_optional_text_support(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=ann["id"],
                        text_prompt=text_prompt
                    )

                # Get refined masks
                for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                    for i, obj_id in enumerate(out_obj_ids):
                        mask = self.clean_mask((out_mask_logits[i] > 0.0).cpu().numpy().squeeze())
                        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours: 
                            # If SAM failed to find a mask, keep the original bbox annotation
                            # Search for the original annotation in to_process
                            # If SAM fails, keep one instance annotation using a bbox polygon fallback.
                            orig_ann = next((a for a in to_process if a["id"] == obj_id), None)
                            if orig_ann:
                                fallback_polygon = self._bbox_to_polygon(orig_ann["bbox"])
                                new_ann = orig_ann.copy()
                                new_ann["id"] = ann_id
                                new_ann["segmentation"] = [fallback_polygon]
                                new_ann["area"] = orig_ann["bbox"][2] * orig_ann["bbox"][3]
                                new_ann["iscrowd"] = orig_ann.get("iscrowd", 0)
                                new_annotations.append(new_ann)
                                ann_id += 1
                            continue

                        largest = max(contours, key=cv2.contourArea)
                        epsilon = 0.002 * cv2.arcLength(largest, True)
                        approx = cv2.approxPolyDP(largest, epsilon, True)
                        polygon = approx.flatten().tolist()

                        if len(polygon) < 6:
                            # Too small, fallback to original bbox
                            # Too small to form a valid polygon; fallback to bbox polygon.
                            orig_ann = next((a for a in to_process if a["id"] == obj_id), None)
                            if orig_ann:
                                fallback_polygon = self._bbox_to_polygon(orig_ann["bbox"])
                                new_ann = orig_ann.copy()
                                new_ann["id"] = ann_id
                                new_ann["segmentation"] = [fallback_polygon]
                                new_ann["area"] = orig_ann["bbox"][2] * orig_ann["bbox"][3]
                                new_ann["iscrowd"] = orig_ann.get("iscrowd", 0)
                                new_annotations.append(new_ann)
                                ann_id += 1
                            continue

                        # Calculate new bbox from mask
                        x_coords = polygon[0::2]
                        y_coords = polygon[1::2]
                        new_bbox = [min(x_coords), min(y_coords), max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)]

                        orig_ann = next((a for a in to_process if a["id"] == obj_id), None)
                        category_id = orig_ann["category_id"] if orig_ann else 1

                        new_annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": category_id,
                            "segmentation": [polygon],
                            "bbox": new_bbox,
                            "area": cv2.contourArea(approx),
                            "iscrowd": 0
                        })
                        ann_id += 1

                self.predictor.reset_state(inference_state)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                # Fallback: keep original annotations if SAM fails
                # Fallback: preserve one segmentation per bbox using rectangular polygons.
                for ann in to_process:
                    fallback_polygon = self._bbox_to_polygon(ann["bbox"])
                    new_ann = ann.copy()
                    new_ann["id"] = ann_id
                    new_ann["segmentation"] = [fallback_polygon]
                    new_ann["area"] = ann["bbox"][2] * ann["bbox"][3]
                    new_ann["iscrowd"] = ann.get("iscrowd", 0)
                    new_annotations.append(new_ann)
                    ann_id += 1
            finally:
                if (img_id % 50 == 0):
                    torch.cuda.empty_cache()
                    gc.collect()

        # Save resulting JSON
        coco_data["annotations"] = new_annotations
        output_json = os.path.join(self.export_dir, f"{split_name}_refined.json")
        with open(output_json, 'w') as f:
            json.dump(coco_data, f)

        print(f"✓ Exported refined COCO to {output_json}")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

    def process_video(self, video_path, json_path):
        """
        Kept for compatibility, but stripped of comparison video.
        """
        video_name = pathlib.Path(video_path).stem
        print(f"--- Processing video: {video_name} ---")
        # Logic would be similar to process_images but with frame extraction
        # For now, we focus on the user's image splits.
        pass

def main():
    # CLI entry point remains similar but main logic is in main.py
    pass

if __name__ == "__main__":
    main()
