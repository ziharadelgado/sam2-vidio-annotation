# SAM 2 Video Annotation for Deep-Sea Shark Detection

**Automated annotation pipeline using SAM 2 to track sharks through underwater video footage, reducing manual annotation time from hours to minutes.**

## 📋 Overview

This Jupyter notebook workflow uses Meta's Segment Anything Model 2 (SAM 2) to automatically propagate annotations through video frames. You annotate **one frame** manually in Roboflow, and SAM 2 tracks the object through the entire video.


**Key Benefits:**
- ✅ Annotate 500+ frames in ~10 minutes (vs 3+ hours manually)
- ✅ Consistent annotation quality across frames
- ✅ Exports directly to COCO format for Roboflow/YOLO training

---

## 🚀 Quick Start

### Prerequisites
- Access to HPC cluster with GPU (tested on Unity @ URI)
- Slurm job scheduler
- Python 3.12+
- CUDA-compatible GPU (A100, L40S, or H100 recommended)

### Before Starting Jupyter Notebook

**CRITICAL:** Run these commands in your terminal **before** launching Jupyter:
```bash
# Request GPU with sufficient memory
# Use constraint to specify which GPUs are acceptable
srun --time=5:00:00 --mem=64G --gres=gpu:1 --constraint="a100|l40s|h100" --pty bash

# Navigate to SAM 2 directory
cd /home/your_username/sam2

# Launch Jupyter Lab from SAM 2 directory
jupyter lab
```

**Why these specific commands?**
1. **`--constraint="a100|l40s|h100"`**: Ensures consistent performance across different GPU nodes. SAM 2's behavior varies on different GPUs without proper configuration.
2. **`--mem=64G`**: Allocates sufficient RAM for video frame processing (default 8GB is insufficient).
3. **`cd sam2` before Jupyter**: Prevents Python import conflicts. SAM 2 will crash if launched from the wrong directory.

---

## 💡 Workflow

1. **Initialize SAM 2 Model** - Load model into GPU memory
2. **Extract Frames** - Resize video frames to 1024×576 (reduces memory from 113GB → 58GB)
3. **Export Frame for Annotation** - Download one clear frame
4. **Annotate in Roboflow** - Use SAM tool (not manual polygon!) for best results
5. **Load Annotation** - Import your Roboflow COCO export
6. **Propagate** - SAM 2 tracks through entire video automatically
7. **Export** - COCO format ready for YOLO training

**Total Time:** ~30 minutes for 500-frame video (including annotation)

---

## 🐛 Common Issues & Solutions

### Problem: Kernel Dies/Restarts During Propagation

**Cause:** Video resolution too high for GPU memory.

**Solution:** Cell 3 resizes frames to 1024×576. For longer videos (>1000 frames), either:
- Increase `FRAME_RATE` from 5 to 10 (extracts fewer frames)
- Use `TARGET_WIDTH = 800, TARGET_HEIGHT = 450` (smaller resolution)

---

### Problem: SAM 2 Loses Tracking When Shark Leaves Frame

**Cause:** SAM 2 can't reacquire objects after they exit/re-enter the frame.

**Solution:** We dont have it

---

### Problem: Tracking Includes Bite Marks or Background Artifacts

**Cause:** Dark underwater footage causes false positives.

**Solutions:**
1. Use `clean_mask()` function (already in Cell 11) - removes small disconnected regions
2. Adjust threshold: Change `> 0.2` to `> 0.3` in Cell 12 for stricter segmentation


---

### Problem: "RuntimeError: Running from parent directory" on Import

**Cause:** Jupyter launched from wrong directory.

**Solution:** Always launch Jupyter from `/home/username/sam2` directory (see Quick Start above).

---

### Problem: Frame Size Mismatch Between Roboflow and Extracted Frames

**Cause:** Roboflow annotation is 1920×1080, but extracted frames are 1024×576.

**Solution:** Cell 8 automatically resizes the mask - verify the overlay looks correct before proceeding.

---

## 🔬 Technical Challenges & Design Decisions

### 1. GPU Memory Management

**Challenge:** Original 1920×1080 frames required 113GB GPU memory, exceeding H100's 85GB capacity.

**Solution:**
- Resize frames to 1024×576 (~58GB)
- Use `offload_video_to_cpu=True` and `offload_state_to_cpu=True`
- Clear GPU cache every 100 frames during propagation
- Batch processing prevents memory accumulation

**Code:** See Cell 3 (frame resizing) and Cell 12 (batch clearing)

---

### 2. Inconsistent Results Across Different GPU Nodes

**Challenge:** SAM 2 produced different tracking results on A100 vs L40S vs H100 GPUs in the same session.

**Root Cause:** `torch.backends.cudnn.benchmark = True` (PyTorch default) enables auto-optimization that selects different algorithms per GPU architecture.

**Solution:**
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Forces same algorithm on all GPUs
```

Also set random seeds for reproducibility across sessions.

**Code:** See Cell 1, lines 35-36

---

### 3. Python Import Conflicts

**Challenge:** SAM 2 has strict directory requirements and crashes if imported from the wrong path.

**Root Cause:** SAM 2's `build_sam.py` checks if you're running from the parent directory of the repo (not allowed) vs inside the repo (required).

**Solution:**
- Always `cd` into `/sam2` before launching Jupyter
- Cell 1 explicitly sets working directory and clears cached modules
- Try-catch blocks provide clear error messages instead of silent crashes

**Code:** See Cell 1, lines 13-23

---

### 4. Bite Mark Artifacts in Dark Underwater Footage

**Challenge:** Sharks with visible bite marks (lighter-colored scar tissue) get tracked as separate objects in dark water.

**Solution:** 
- `clean_mask()` function keeps only largest connected component
- Removes disconnected regions < 500 pixels
- Works because shark body is always the largest region

**Code:** See Cell 11 (function definition) and Cell 12 (application)

---

### 5. Polygon vs Mask Annotation Format

**Challenge:** Roboflow exports different formats depending on annotation tool used (polygon tool → list, SAM tool → RLE dict).

**Solution:** Cell 7 auto-detects format and handles both:
```python
if isinstance(segmentation, dict):  # RLE mask
    mask = mask_utils.decode(segmentation)
elif isinstance(segmentation, list):  # Polygon
    cv2.fillPoly(mask, [polygon_np], 1)
```

**Recommendation:** Use Roboflow's Smart Polygon/SAM tool (not manual brush) - produces masks that match SAM 2's internal representation better.

---

## 🎯 Future Improvements

1. **Multi-Object Tracking**
   - Current: Tracks one shark per video
   - Enhancement: Track multiple sharks with different `obj_id` values
   - Use case: Videos with multiple individuals
   

2. **Active Learning Integration**
   - Re-train YOLO on corrected annotations
   - Iterative improvement loop

