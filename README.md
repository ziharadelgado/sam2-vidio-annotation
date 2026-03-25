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

**1.** Ensure your annotated dataset is available on your rclone. Rclone remote MUST be named 'gdrive'.

**2.** Edit `sync_and_run.sh` such that it references your remote and the annotated data contained within. This means editing <PROJECT_DIR> and <QUEUE_DIR> such that the pathing matches your login node file structure.

**3.** Make sure DeepSea_ObjectDetection folder in Google Drive has an alias/shortcut within your HOME directory.

**4.** Make sure to clone this project into a folder called 'trainer'.

**5.** Run `scripts/sync_and_run.sh`
```
