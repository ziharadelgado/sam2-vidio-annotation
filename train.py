# train.py — Updated from Kamron's version for YOLO26n-seg
# Changes: yolo11l-seg.pt → yolo26n-seg.pt, project name updated

import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from collections import Counter
from roboflow import Roboflow
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ROBOFLOW_API_KEY  = "mvFKQaefMTrBAiSy194g"
ROBOFLOW_WORKSPACE = "DeepSeaObjectDetection"
ROBOFLOW_PROJECT   = "deep-sea-object-detection"
ROBOFLOW_VERSION   = 1

DATASET_DIR   = Path("./dataset")
RUNS_DIR      = Path("./runs")
CHECKPOINT    = None

EPOCHS        = 100
IMGSZ         = 640
BATCH         = 16
WORKERS       = 4
DEVICE        = 0

# Updated: YOLO26n-seg instead of yolo11l-seg
YOLO_WEIGHTS  = "yolo26n-seg.pt"


def download_dataset() -> Path:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)
    dataset = version.download(
        model_format="yolov11",
        location=str(DATASET_DIR),
        overwrite=False
    )
    data_yaml = DATASET_DIR / "data.yaml"
    assert data_yaml.exists(), f"data.yaml not found at {data_yaml}"
    print(f"[dataset] Downloaded to: {DATASET_DIR}")
    return data_yaml


def compute_class_weights(data_yaml: Path) -> dict:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    label_dir = DATASET_DIR / "train" / "labels"
    num_classes = cfg.get("nc", len(cfg.get("names", [])))

    counts = Counter()
    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1

    total = sum(counts.values()) or 1
    weights = {}
    for cls_idx in range(num_classes):
        freq = counts.get(cls_idx, 1) / total
        weights[cls_idx] = 1.0 / (freq * num_classes)

    min_w = min(weights.values())
    weights = {k: v / min_w for k, v in weights.items()}

    print("[imbalance] Class weights:")
    names = cfg.get("names", {})
    for idx, w in sorted(weights.items()):
        name = names[idx] if isinstance(names, list) else names.get(idx, str(idx))
        count = counts.get(idx, 0)
        print(f"  [{idx}] {name:30s} count={count:6d}  weight={w:.3f}")

    return weights


def train(model_type: str = "yolo26n-seg"):
    print(f"\n{'='*50}")
    print(f" Training: SharkDetect-V4-seg ({model_type})")
    print(f"{'='*50}\n")

    data_yaml = download_dataset()
    class_weights = compute_class_weights(data_yaml)

    if CHECKPOINT and Path(CHECKPOINT).exists():
        model = YOLO(str(CHECKPOINT))
        print(f"[model] Resuming from: {CHECKPOINT}")
    else:
        model = YOLO(YOLO_WEIGHTS)
        print(f"[model] Loaded pretrained: {YOLO_WEIGHTS}")

    run_name = "SharkDetect-V4-seg"

    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        # Loss
        fl_gamma=1.5,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        cos_lr=True,
        # Regularization
        weight_decay=0.0005,
        # Augmentation (underwater-specific from Kamron)
        fliplr=0.5,
        flipud=0.1,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        hsv_h=0.03,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        erasing=0.2,
        # Checkpointing
        save=True,
        save_period=10,
        patience=30,
        plots=True,
        verbose=True,
    )

    print(f"\n[done] Training complete. Results: {RUNS_DIR / run_name}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SharkDetect-V4-seg")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH = args.batch
    if args.checkpoint:
        CHECKPOINT = Path(args.checkpoint)

    train()
