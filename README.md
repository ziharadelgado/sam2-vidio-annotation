# SAM 2 Video Annotation for Deep-Sea Shark Detection

**Automated annotation pipeline using SAM 2 to track sharks through underwater video footage, reducing manual annotation time from hours to minutes.**

## 📋 Overview

This Jupyter notebook workflow uses Meta's Segment Anything Model 2 (SAM 2) to automatically propagate annotations through video frames. Weakly-annotated frames (such as bounding boxes either by hand or generated via a detector) are used to prompt the segmentation model.


**Benefits:**
- Higher annotation quality for little cost
- Quick and easy
- Easily shareable

---

## 🚀 Quickstart

### Prerequisites
- Access to HPC cluster with GPU (tested on Unity @ URI)
- Slurm job scheduler
- Python 3.11+
- CUDA-compatible GPU (A100, L40S, or H100 recommended)

### Directions

**1.** Ensure your annotated dataset is available on your remote of choice (preferred: Rclone).
**2.** Edit `sync_and_run.sh` such that it references your remote and the annotated data contained within.
**3.** Run `sync_and_run.sh` in terminal as `bash` command.

```

**Why these specific commands?**
1. **`--constraint="a100|l40s|h100"`**: Ensures consistent performance across different GPU nodes. SAM 2's behavior varies on different GPUs without proper configuration.
2. **`--mem=64G`**: Allocates sufficient RAM for video frame processing (default 8GB is insufficient).
3. **`cd sam2` before Jupyter**: Prevents Python import conflicts. SAM 2 will crash if launched from the wrong directory.

--
