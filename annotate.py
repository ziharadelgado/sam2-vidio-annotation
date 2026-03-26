# Rewrite of Kamron's annotate.py for SAM 3 + YOLO26n-seg
# Original: SAM 2 bbox prompting on single images
# New: SAM 3 text + exemplar prompting on videos, with YOLO first pass

import os
import subprocess
import time
import sys
import json
import shutil
import pathlib
import gc

import torch
import numpy as np
import random
import cv2
from PIL import Image
from tqdm import tqdm


class SharkAnnotator:
    """
    Replaces Kamron's SAM 2 SharkAnnotator.

    Changes from original:
      - SAM 2 → SAM 3 (text + exemplar prompting)
      - Single-instance bbox → exhaustive concept segmentation
      - Added YOLO26n-seg first pass for exemplar selection
      - Added Roboflow auto-upload
      - process_video() is now the main method (was a stub in Kamron's version)
      - process_images() updated to use SAM 3 text prompts instead of bbox prompts
    """

    def __init__(self,
                 sam3_checkpoint,
                 yolo_model_path=None,
                 device=None,
                 work_dir=None,
                 text_prompt="shark",
                 threshold=0.2,
                 yolo_conf=0.25,
                 frame_rate=5,
                 target_width=1024,
                 target_height=576):

        self.sam3_checkpoint = os.path.abspath(sam3_checkpoint)
        self.yolo_model_path = os.path.abspath(yolo_model_path) if yolo_model_path else None
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.work_dir = os.path.abspath(work_dir or os.path.expanduser("~/annotated-video"))
        self.export_dir = os.path.join(self.work_dir, "exports")
        os.makedirs(self.export_dir, exist_ok=True)

        self.text_prompt = text_prompt
        self.threshold = threshold
        self.yolo_conf = yolo_conf
        self.frame_rate = frame_rate
        self.target_width = target_width
        self.target_height = target_height

        self.sam3_predictor = None
        self.yolo_model = None
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

    # ──────────────────────────────────────────
    # Model Setup
    # ──────────────────────────────────────────
    def setup_model(self):
        """Load SAM 3 and optionally YOLO26n-seg."""
        # SAM 3
        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError:
            raise ImportError(
                "SAM 3 not found. Install with:\n"
                "  cd /home/zihara_delgado_uri_edu/sam3 && pip install -e ."
            )

        print(f"Loading SAM 3 from {self.sam3_checkpoint}...")
        self.sam3_predictor = build_sam3_video_predictor(
            checkpoint_path=self.sam3_checkpoint
        )
        print("✓ SAM 3 loaded")

        # YOLO (optional)
        if self.yolo_model_path and os.path.exists(self.yolo_model_path):
            try:
                from ultralytics import YOLO
                print(f"Loading YOLO from {self.yolo_model_path}...")
                self.yolo_model = YOLO(self.yolo_model_path)
                print("✓ YOLO loaded (SharkDetect-V4-seg)")
            except ImportError:
                print("⚠ ultralytics not installed — running SAM 3 only")
                self.yolo_model = None
        else:
            if self.yolo_model_path:
                print(f"⚠ YOLO model not found at {self.yolo_model_path} — running SAM 3 only")
            self.yolo_model = None

    # ──────────────────────────────────────────
    # Google Drive Sync (same as Kamron's)
    # ──────────────────────────────────────────
    def download_queue(self, gdrive_source="gdrive:DeepSea_ObjectDetection/rclone/queue/"):
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

    # ──────────────────────────────────────────
    # Mask Utilities
    # ──────────────────────────────────────────
    def clean_mask(self, mask):
        """Keep only largest connected component (same as Kamron's)."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = (labels == largest_label).astype(np.uint8)
        else:
            cleaned = mask_uint8
        return cleaned.astype(bool)

    def mask_to_polygon(self, mask, min_points=6):
        """Convert binary mask to polygon coordinates."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.002 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        polygon = approx.flatten().tolist()

        if len(polygon) < min_points:
            return None
        return polygon

    def mask_to_yolo_line(self, mask, class_id=0):
        """Convert mask to YOLO segmentation format line."""
        polygon = self.mask_to_polygon(mask)
        if polygon is None:
            return None

        h, w = mask.shape
        normalized = []
        for i in range(0, len(polygon), 2):
            normalized.append(f"{polygon[i] / w:.6f}")
            normalized.append(f"{polygon[i + 1] / h:.6f}")

        return f"{class_id} " + " ".join(normalized)

    # ──────────────────────────────────────────
    # Frame Extraction
    # ──────────────────────────────────────────
    def extract_frames(self, video_path, output_dir):
        """Extract and resize frames from MP4."""
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"  Video: {orig_w}x{orig_h} @ {fps:.1f} FPS | {total} frames")
        print(f"  Extracting every {self.frame_rate}th frame → {self.target_width}x{self.target_height}")

        extracted = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.frame_rate == 0:
                resized = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_AREA
                )
                cv2.imwrite(os.path.join(output_dir, f"{extracted:06d}.jpg"), resized)
                extracted += 1
            frame_idx += 1

        cap.release()
        print(f"  ✓ Extracted {extracted} frames")
        return extracted

    # ──────────────────────────────────────────
    # YOLO First Pass
    # ──────────────────────────────────────────
    def yolo_first_pass(self, frames_dir):
        """
        Run SharkDetect-V4-seg on all frames.
        Returns the best detection for use as SAM 3 exemplar.
        """
        if self.yolo_model is None:
            print("  ⚠ No YOLO model — skipping first pass")
            return None, None

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        print(f"  Running YOLO on {len(frame_files)} frames...")

        detections = {}
        best_exemplar = None
        best_conf = 0.0

        for i, fname in enumerate(tqdm(frame_files, desc="  YOLO first pass")):
            img_path = os.path.join(frames_dir, fname)
            results = self.yolo_model(img_path, conf=self.yolo_conf, verbose=False)

            frame_dets = []
            for r in results:
                if r.boxes is None:
                    continue
                for j in range(len(r.boxes)):
                    box = r.boxes[j].xyxy[0].cpu().numpy().tolist()
                    conf_score = float(r.boxes[j].conf[0])

                    det = {
                        "box": box,
                        "conf": conf_score,
                        "frame_idx": i,
                    }
                    frame_dets.append(det)

                    if conf_score > best_conf:
                        best_conf = conf_score
                        best_exemplar = det.copy()

            detections[i] = frame_dets

        total_dets = sum(len(d) for d in detections.values())
        frames_with = sum(1 for d in detections.values() if len(d) > 0)
        print(f"  ✓ YOLO: {total_dets} detections across {frames_with} frames")
        if best_exemplar:
            print(f"  Best exemplar: frame {best_exemplar['frame_idx']}, conf={best_exemplar['conf']:.3f}")

        return detections, best_exemplar

    # ──────────────────────────────────────────
    # SAM 3 Video Processing (NEW — was a stub in Kamron's version)
    # ──────────────────────────────────────────
    def process_video(self, video_path, json_path=None):
        """
        Full video annotation pipeline:
          1. Extract frames
          2. YOLO first pass → find best exemplar
          3. SAM 3 text prompt + exemplar → segment all sharks
          4. Export to YOLO + COCO format
        """
        video_name = pathlib.Path(video_path).stem
        print(f"\n{'='*50}")
        print(f"  Processing video: {video_name}")
        print(f"{'='*50}")

        # Step 1: Extract frames
        frames_dir = os.path.join(self.work_dir, "frames", video_name)
        if os.path.exists(frames_dir) and os.listdir(frames_dir):
            num_frames = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
            print(f"  Using existing {num_frames} frames at {frames_dir}")
        else:
            self.extract_frames(video_path, frames_dir)

        # Step 2: YOLO first pass
        yolo_detections, best_exemplar = self.yolo_first_pass(frames_dir)

        # Step 3: SAM 3 annotation
        all_masks = self._run_sam3_video(frames_dir, best_exemplar)

        # Step 4: Export
        yolo_dir = os.path.join(self.export_dir, video_name, "yolo_export")
        coco_dir = os.path.join(self.export_dir, video_name, "coco_export")

        self._export_yolo(all_masks, frames_dir, yolo_dir)
        coco_json = self._export_coco(all_masks, frames_dir, coco_dir)

        print(f"\n✓ Video {video_name} complete!")
        print(f"  YOLO export: {yolo_dir}")
        print(f"  COCO export: {coco_dir}")

        return coco_dir

    def _run_sam3_video(self, frames_dir, exemplar=None):
        """Run SAM 3 on a frames directory with text prompt + optional exemplar."""
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        num_frames = len(frame_files)

        if exemplar:
            print(f"  SAM 3: text=\"{self.text_prompt}\" + exemplar from frame {exemplar['frame_idx']}")
        else:
            print(f"  SAM 3: text=\"{self.text_prompt}\" (no exemplar)")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Start session
            response = self.sam3_predictor.handle_request(
                request=dict(type="start_session", resource_path=str(frames_dir))
            )
            session_id = response["session_id"]

            # Add exemplar prompt if available (YOLO's best detection as bbox)
            if exemplar:
                self.sam3_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=exemplar["frame_idx"],
                        box=exemplar["box"],
                    )
                )

            # Add text prompt on frame 0
            response = self.sam3_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=self.text_prompt,
                )
            )

            out = response.get("outputs")
            if out:
                n = len(out.get("obj_ids", []))
                print(f"  Detected {n} object(s) on frame 0")
            else:
                print("  ⚠ No detections on frame 0")

            # Propagate
            print(f"  Propagating through {num_frames} frames...")
            all_masks = {}

            for resp in self.sam3_predictor.handle_request(
                request=dict(type="propagate", session_id=session_id)
            ):
                frame_idx = resp.get("frame_index")
                if frame_idx is None:
                    continue

                masks_for_frame = []
                mask_logits = resp.get("mask_logits")
                obj_ids = resp.get("obj_ids", [])

                if mask_logits is not None:
                    for i, obj_id in enumerate(obj_ids):
                        mask = (mask_logits[i] > self.threshold).cpu().numpy().squeeze()
                        if mask.any():
                            cleaned = self.clean_mask(mask)
                            if cleaned.any():
                                masks_for_frame.append(cleaned)

                all_masks[frame_idx] = masks_for_frame

                if frame_idx % 100 == 0 and frame_idx > 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"    Frame {frame_idx}/{num_frames}")

            # End session
            self.sam3_predictor.handle_request(
                request=dict(type="end_session", session_id=session_id)
            )

        frames_with = sum(1 for m in all_masks.values() if len(m) > 0)
        total_inst = sum(len(m) for m in all_masks.values())
        print(f"  ✓ SAM 3: {frames_with} frames with sharks, {total_inst} total instances")

        return all_masks

    # ──────────────────────────────────────────
    # Image Processing (updated from Kamron's for SAM 3)
    # ──────────────────────────────────────────
    def process_images(self, image_dir, coco_json_path):
        """
        Process a directory of images using SAM 3 text prompts.
        Replaces Kamron's bbox-based SAM 2 refinement.

        For each image:
          - SAM 3 text prompt "shark" → finds all instances
          - Outputs refined segmentation masks
          - Saves as COCO JSON
        """
        split_name = pathlib.Path(image_dir).name
        print(f"\n--- Processing split: {split_name} ---")

        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)

        images_info = {img["id"]: img for img in coco_data["images"]}
        sorted_img_ids = sorted(images_info.keys())

        new_annotations = []
        ann_id = 1

        # We'll use a temp dir for single-image "videos" (same trick as Kamron's)
        temp_dir = os.path.join(self.work_dir, "temp_single")

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            print("  Using SAM 3 image processor for single images...")
            image_model = build_sam3_image_model(
                checkpoint_path=self.sam3_checkpoint
            )
            processor = Sam3Processor(image_model)
        except Exception as e:
            print(f"  ⚠ Could not load SAM 3 image model: {e}")
            print("  Falling back to video predictor with single-frame trick")
            processor = None

        for img_id in tqdm(sorted_img_ids, desc=f"  Refining {split_name}"):
            img_info = images_info[img_id]
            file_name = img_info["file_name"]
            src_path = os.path.join(image_dir, file_name)

            if not os.path.exists(src_path):
                continue

            try:
                if processor:
                    # Use SAM 3 image processor directly
                    image = Image.open(src_path)
                    inference_state = processor.set_image(image)
                    output = processor.set_text_prompt(
                        state=inference_state,
                        prompt=self.text_prompt
                    )

                    masks = output["masks"]
                    scores = output["scores"]

                    for i in range(len(masks)):
                        if scores[i] < self.threshold:
                            continue

                        mask = masks[i].cpu().numpy().squeeze().astype(bool)
                        cleaned = self.clean_mask(mask)
                        polygon = self.mask_to_polygon(cleaned)

                        if polygon is None:
                            continue

                        x_coords = polygon[0::2]
                        y_coords = polygon[1::2]
                        bbox = [
                            min(x_coords), min(y_coords),
                            max(x_coords) - min(x_coords),
                            max(y_coords) - min(y_coords)
                        ]

                        new_annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 1,
                            "segmentation": [polygon],
                            "bbox": bbox,
                            "area": cv2.contourArea(
                                np.array(polygon).reshape(-1, 2).astype(np.float32)
                            ),
                            "iscrowd": 0
                        })
                        ann_id += 1

                else:
                    # Fallback: single-frame video trick (like Kamron's approach)
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    os.makedirs(temp_dir)
                    shutil.copy(src_path, os.path.join(temp_dir, "00000.jpg"))

                    masks = self._run_sam3_video(temp_dir)
                    for mask in masks.get(0, []):
                        polygon = self.mask_to_polygon(mask)
                        if polygon is None:
                            continue

                        x_coords = polygon[0::2]
                        y_coords = polygon[1::2]
                        bbox = [
                            min(x_coords), min(y_coords),
                            max(x_coords) - min(x_coords),
                            max(y_coords) - min(y_coords)
                        ]

                        new_annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": 1,
                            "segmentation": [polygon],
                            "bbox": bbox,
                            "area": cv2.contourArea(
                                np.array(polygon).reshape(-1, 2).astype(np.float32)
                            ),
                            "iscrowd": 0
                        })
                        ann_id += 1

            except Exception as e:
                print(f"  Error processing {file_name}: {e}")

            finally:
                if img_id % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

        # Save
        coco_data["annotations"] = new_annotations
        output_json = os.path.join(self.export_dir, f"{split_name}_refined.json")
        with open(output_json, 'w') as f:
            json.dump(coco_data, f)

        print(f"  ✓ Exported refined COCO to {output_json}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    # ──────────────────────────────────────────
    # Export Methods
    # ──────────────────────────────────────────
    def _export_yolo(self, all_masks, frames_dir, output_dir):
        """Export masks to YOLO segmentation format."""
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        annotations = 0

        for frame_idx, masks in all_masks.items():
            if frame_idx >= len(frame_files):
                continue

            fname = frame_files[frame_idx]
            stem = pathlib.Path(fname).stem

            shutil.copy2(
                os.path.join(frames_dir, fname),
                os.path.join(images_dir, fname)
            )

            lines = []
            for mask in masks:
                line = self.mask_to_yolo_line(mask, class_id=0)
                if line:
                    lines.append(line)
                    annotations += 1

            with open(os.path.join(labels_dir, f"{stem}.txt"), "w") as f:
                f.write("\n".join(lines))

        # data.yaml
        with open(os.path.join(output_dir, "data.yaml"), "w") as f:
            f.write(f"train: {images_dir}\n")
            f.write(f"val: {images_dir}\n")
            f.write(f"nc: 1\n")
            f.write(f"names: ['shark']\n")

        print(f"  ✓ YOLO export: {annotations} annotations → {output_dir}")

    def _export_coco(self, all_masks, frames_dir, output_dir):
        """Export masks to COCO JSON format."""
        os.makedirs(output_dir, exist_ok=True)

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "shark"}]
        }
        ann_id = 1

        for frame_idx, masks in all_masks.items():
            if frame_idx >= len(frame_files):
                continue

            fname = frame_files[frame_idx]
            shutil.copy2(
                os.path.join(frames_dir, fname),
                os.path.join(output_dir, fname)
            )

            img = cv2.imread(os.path.join(frames_dir, fname))
            h, w = img.shape[:2]

            coco["images"].append({
                "id": frame_idx, "file_name": fname, "width": w, "height": h
            })

            for mask in masks:
                polygon = self.mask_to_polygon(mask)
                if polygon is None:
                    continue

                x_c = polygon[0::2]
                y_c = polygon[1::2]
                bbox = [min(x_c), min(y_c), max(x_c) - min(x_c), max(y_c) - min(y_c)]

                coco["annotations"].append({
                    "id": ann_id, "image_id": frame_idx, "category_id": 0,
                    "segmentation": [polygon], "bbox": bbox,
                    "area": cv2.contourArea(
                        np.array(polygon).reshape(-1, 2).astype(np.float32)
                    ),
                    "iscrowd": 0
                })
                ann_id += 1

        json_path = os.path.join(output_dir, "_annotations.coco.json")
        with open(json_path, 'w') as f:
            json.dump(coco, f)

        print(f"  ✓ COCO export: {len(coco['annotations'])} annotations → {json_path}")
        return json_path

    # ──────────────────────────────────────────
    # Roboflow Upload (NEW)
    # ──────────────────────────────────────────
    def upload_to_roboflow(self, coco_dir, api_key, workspace, project_name):
        """Auto-upload COCO export to Roboflow."""
        try:
            from roboflow import Roboflow
        except ImportError:
            print("  ⚠ roboflow not installed — skipping upload")
            return False

        print(f"\n  Uploading to Roboflow: {workspace}/{project_name}")

        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)

        json_path = os.path.join(coco_dir, "_annotations.coco.json")
        image_files = sorted([f for f in os.listdir(coco_dir) if f.endswith(".jpg")])

        for img_file in tqdm(image_files, desc="  Uploading"):
            img_path = os.path.join(coco_dir, img_file)
            try:
                project.upload(
                    image_path=img_path,
                    annotation_path=json_path,
                    annotation_format="coco"
                )
            except Exception as e:
                print(f"    ⚠ Failed: {img_file}: {e}")

        print(f"  ✓ Upload complete!")
        return True
