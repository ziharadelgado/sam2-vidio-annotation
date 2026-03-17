import os
import argparse
import yaml
import numpy as np
from pathlib import Path
from collections import Counter

from roboflow import Roboflow
from ultralytics import YOLO, RTDETR

# ─────────────────────────────────────────────
# CONFIG — fill these in or pass as CLI args
# ─────────────────────────────────────────────

ROBOFLOW_API_KEY  = "mvFKQaefMTrBAiSy194g"
ROBOFLOW_WORKSPACE = "DeepSeaObjectDetection"
ROBOFLOW_PROJECT   = "Fish-Detection-v5"
ROBOFLOW_VERSION   = 1               # dataset version number

# Paths
DATASET_DIR   = Path("./dataset")   # where Roboflow dataset will be downloaded
RUNS_DIR      = Path("./runs")      # training output dir
CHECKPOINT    = None                # e.g. Path("./runs/train/weights/last.pt")
                                    # Set to None to start from pretrained weights

# Training hyperparameters
EPOCHS        = 100
IMGSZ         = 640
BATCH         = 16                  # reduce to 8 if OOM on smaller GPU
WORKERS       = 4
DEVICE        = 0                   # 0 = first GPU; "cpu" for CPU-only

# Model selection: "yolov11" or "rtdetr"
MODEL_TYPE    = "yolov11"

# Pretrained weight defaults (used when CHECKPOINT is None)
YOLO_WEIGHTS  = "yolo11l-seg.pt"        # options: yolo11n/s/m/l/x.pt
RTDETR_WEIGHTS = "rtdetr-l.pt"      # options: rtdetr-l.pt, rtdetr-x.pt


# ─────────────────────────────────────────────
# STEP 1: PULL DATASET FROM ROBOFLOW
# ─────────────────────────────────────────────

def download_dataset() -> Path:
    """
    Downloads the dataset from Roboflow in YOLOv11 format.
    Returns the path to the dataset.yaml file.

    NOTE: Downloading does NOT consume Roboflow training credits.
    Only using Roboflow's hosted training does.
    """
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    version = project.version(ROBOFLOW_VERSION)

    dataset = version.download(
        model_format="yolov11",    # works for both YOLO and RT-DETR
        location=str(DATASET_DIR),
        overwrite=False            # skip re-download if already present
    )

    data_yaml = DATASET_DIR / "data.yaml"
    assert data_yaml.exists(), f"data.yaml not found at {data_yaml}"
    print(f"[dataset] Downloaded to: {DATASET_DIR}")
    return data_yaml


# ─────────────────────────────────────────────
# STEP 2: ANALYZE CLASS IMBALANCE
# ─────────────────────────────────────────────

def compute_class_weights(data_yaml: Path) -> dict:
    """
    Scans the training label files and computes per-class weights
    inversely proportional to class frequency.

    Returns a dict of class_index -> weight (normalized so min weight = 1.0).

    Strategy used here: inverse frequency weighting.
    Other options to consider:
      - Focal loss (set fl_gamma > 0 in training args) — penalizes easy examples
      - Oversample rare classes by duplicating their images in the dataset
      - Class-weighted sampler (requires custom DataLoader, not native in Ultralytics)

    For most underwater datasets, focal loss (fl_gamma=1.5–2.0) combined with
    inverse frequency weights is a good starting point.
    """
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    train_img_dir = DATASET_DIR / "train" / "images"
    label_dir     = DATASET_DIR / "train" / "labels"
    num_classes   = cfg.get("nc", len(cfg.get("names", [])))

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

    # Normalize: min weight = 1.0
    min_w = min(weights.values())
    weights = {k: v / min_w for k, v in weights.items()}

    print("[imbalance] Class weights (higher = rarer class):")
    names = cfg.get("names", {})
    for idx, w in sorted(weights.items()):
        name = names[idx] if isinstance(names, list) else names.get(idx, str(idx))
        count = counts.get(idx, 0)
        print(f"  [{idx}] {name:30s} count={count:6d}  weight={w:.3f}")

    return weights


# ─────────────────────────────────────────────
# STEP 3: BUILD AUGMENTATION CONFIG
# ─────────────────────────────────────────────

def get_augmentation_args() -> dict:
    """
    Returns Ultralytics training kwargs for underwater-specific augmentation.

    Ultralytics applies augmentations natively — no separate albumentations
    pipeline needed unless you want custom transforms beyond what's listed here.

    Key underwater considerations:
      - Color cast: water shifts imagery toward blue/green. HSV jitter corrects for this.
      - Turbidity: blur + noise simulate murky water conditions.
      - Mosaic/mixup: helps with small, clustered objects common in underwater scenes.
      - Flipping: safe for underwater (no orientation cue like sky/ground).
    """
    return {
        # ── Spatial ──────────────────────────────────────
        "fliplr":       0.5,     # horizontal flip probability
        "flipud":       0.1,     # vertical flip (unusual but ok underwater)
        "degrees":      10.0,    # rotation range ±10°
        "translate":    0.1,     # translation fraction
        "scale":        0.5,     # scale jitter ±50%
        "shear":        2.0,     # shear degrees

        # ── Color / appearance ───────────────────────────
        "hsv_h":        0.03,    # hue shift — wider than default for water color cast
        "hsv_s":        0.7,     # saturation jitter
        "hsv_v":        0.4,     # brightness jitter (accounts for depth-related dimming)

        # ── Mosaic / mixing ──────────────────────────────
        "mosaic":       1.0,     # mosaic augmentation (4-image composite), 0.0 to disable
        "mixup":        0.15,    # mixup blending probability
        "copy_paste":   0.1,     # copy-paste augmentation (good for rare objects)

        # ── Blur / noise (turbidity simulation) ─────────
        "blur":         0.05,    # Gaussian blur probability — keep low to avoid over-blurring
        # Note: additional turbidity augmentation (e.g. albumentations RandomFog)
        # can be applied as a custom preprocessing step before training if needed.

        # ── Erasing ──────────────────────────────────────
        "erasing":      0.2,     # random erasing (simulates occlusion by marine debris)
    }


# ─────────────────────────────────────────────
# STEP 4: BUILD TRAINING ARGS
# ─────────────────────────────────────────────

def get_training_args(data_yaml: Path, class_weights: dict, run_name: str) -> dict:
    """
    Assembles the full dict of kwargs passed to model.train().
    """
    aug_args = get_augmentation_args()

    base_args = {
        "data":         str(data_yaml),
        "epochs":       EPOCHS,
        "imgsz":        IMGSZ,
        "batch":        BATCH,
        "workers":      WORKERS,
        "device":       DEVICE,
        "project":      str(RUNS_DIR),
        "name":         run_name,
        "exist_ok":     True,       # allow resuming into same run directory

        # ── Loss / imbalance ─────────────────────────────
        "fl_gamma":     1.5,        # focal loss gamma; 0.0 disables focal loss
                                    # increase to 2.0 if rare classes still underfit

        # ── Optimizer ────────────────────────────────────
        "optimizer":    "AdamW",
        "lr0":          0.001,      # initial learning rate
        "lrf":          0.01,       # final lr = lr0 * lrf
        "warmup_epochs": 3,
        "cos_lr":       True,       # cosine LR schedule

        # ── Regularization ───────────────────────────────
        "weight_decay": 0.0005,
        "dropout":      0.0,        # set 0.1–0.3 for RT-DETR if overfitting

        # ── Checkpointing ────────────────────────────────
        "save":         True,
        "save_period":  10,         # save checkpoint every N epochs
        "patience":     30,         # early stopping patience (epochs without improvement)

        # ── Logging ──────────────────────────────────────
        "plots":        True,       # save training plots (loss curves, PR curves, etc.)
        "verbose":      True,
    }

    # Resume from checkpoint if provided
    if CHECKPOINT and Path(CHECKPOINT).exists():
        print(f"[checkpoint] Resuming from: {CHECKPOINT}")
        base_args["resume"] = True
        # When resume=True, Ultralytics loads the checkpoint path via `model = YOLO(checkpoint)`
        # so we don't pass it here — handled in load_model() below.
    else:
        print("[checkpoint] No checkpoint found — starting from pretrained weights.")

    # Merge augmentation args
    base_args.update(aug_args)

    # NOTE: Ultralytics doesn't natively accept a class_weights tensor in train().
    # The recommended approach for class imbalance is focal loss (fl_gamma above)
    # combined with oversampling. If you need hard per-class loss weights, you
    # would subclass the Ultralytics Trainer and override compute_loss() — see
    # the comment block at the bottom of this file for a starter on that approach.

    return base_args


# ─────────────────────────────────────────────
# STEP 5: LOAD MODEL
# ─────────────────────────────────────────────

def load_model(model_type: str):
    """
    Loads YOLOv11 or RT-DETR, either from a checkpoint or pretrained weights.
    """
    if CHECKPOINT and Path(CHECKPOINT).exists():
        checkpoint_path = str(CHECKPOINT)
        if model_type == "yolov11":
            model = YOLO(checkpoint_path)
        else:
            model = RTDETR(checkpoint_path)
        print(f"[model] Loaded from checkpoint: {checkpoint_path}")
    else:
        if model_type == "yolov11":
            model = YOLO(YOLO_WEIGHTS)
            print(f"[model] YOLOv11 loaded from pretrained: {YOLO_WEIGHTS}")
        else:
            model = RTDETR(RTDETR_WEIGHTS)
            print(f"[model] RT-DETR loaded from pretrained: {RTDETR_WEIGHTS}")
    return model


# ─────────────────────────────────────────────
# STEP 6: TRAIN
# ─────────────────────────────────────────────

def train(model_type: str):
    print(f"\n{'='*50}")
    print(f" Training: {model_type.upper()}")
    print(f"{'='*50}\n")

    # 1. Download dataset
    data_yaml = download_dataset()

    # 2. Analyze class imbalance
    class_weights = compute_class_weights(data_yaml)

    # 3. Load model
    model = load_model(model_type)

    # 4. Assemble training args
    run_name  = f"{model_type}_underwater"
    train_args = get_training_args(data_yaml, class_weights, run_name)

    # 5. Train
    results = model.train(**train_args)

    print(f"\n[done] Training complete. Results saved to: {RUNS_DIR / run_name}")
    return results


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Underwater object detection trainer")
    parser.add_argument(
        "--model", type=str, default=MODEL_TYPE,
        choices=["yolov11", "rtdetr"],
        help="Which model to train"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a .pt checkpoint to resume from (overrides CHECKPOINT constant)"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch", type=int, default=BATCH,
        help="Batch size"
    )
    return parser.parse_args()



args = parse_args()

# Allow CLI to override constants
MODEL_TYPE = args.model
EPOCHS     = args.epochs
BATCH      = args.batch
if args.checkpoint:
    CHECKPOINT = Path(args.checkpoint)

train(MODEL_TYPE)


# ─────────────────────────────────────────────
# APPENDIX: Custom loss weighting (advanced)
# ─────────────────────────────────────────────
#
# If focal loss alone isn't enough to handle class imbalance, you can subclass
# the Ultralytics trainer to inject per-class loss weights. Skeleton below:
#
# from ultralytics.models.yolo.detect import DetectionTrainer
# import torch
#
# class WeightedDetectionTrainer(DetectionTrainer):
#     def __init__(self, *args, class_weights=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights_tensor = class_weights  # torch.Tensor of shape [nc]
#
#     def get_model(self, cfg=None, weights=None, verbose=True):
#         model = super().get_model(cfg, weights, verbose)
#         if self.class_weights_tensor is not None:
#             # Inject weights into the detection loss head
#             model.model[-1].cls_weights = self.class_weights_tensor.to(self.device)
#         return model
#
# # Then train with:
# # weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float32)
# # trainer = WeightedDetectionTrainer(overrides=train_args, class_weights=weights_tensor)
# # trainer.train()
#
# Note: This requires familiarity with the Ultralytics internals and may break
# across versions. Focal loss (fl_gamma) is the simpler, more stable path.