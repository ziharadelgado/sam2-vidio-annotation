import argparse
import json
import os
from pathlib import Path
from ultralytics import YOLO, RTDETR
from tidecv import TIDE, datasets

def run_eval(model_path, data_yaml, gt_json, output_dir, model_type="yolov11"):
    """
    Runs TIDE evaluation on a trained model.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"--- Loading model: {model_path} ---")
    if model_type == "yolov11":
        model = YOLO(model_path)
    else:
        model = RTDETR(model_path)

    print(f"--- Running validation to generate predictions ---")
    # save_json=True is critical for TIDE as it needs COCO format
    results = model.val(
        data=data_yaml,
        split='val',
        save_json=True,
        project=str(output_path),
        name="val_results"
    )

    # Locate the predictions JSON
    # Ultralytics typically saves this in project/name/predictions.json
    pred_json = output_path / "val_results" / "predictions.json"
    if not pred_json.exists():
        # Fallback: search for any json in the val_results folder
        jsons = list((output_path / "val_results").glob("*.json"))
        if jsons:
            pred_json = jsons[0]
        else:
            raise FileNotFoundError(f"Could not find predictions JSON at {pred_json}")

    print(f"--- Running TIDE Evaluation ---")
    print(f"GT: {gt_json}")
    print(f"Pred: {pred_json}")

    tide = TIDE()
    
    # Determine mode (MASK if it's a segmentation model, else BOX)
    mode = TIDE.BOX
    if "seg" in str(model_path).lower() or (hasattr(model, 'task') and model.task == 'segment'):
        mode = TIDE.MASK
        print("Using TIDE.MASK mode")
    else:
        print("Using TIDE.BOX mode")

    # Load datasets
    gt_data = datasets.COCO(str(gt_json))
    pred_data = datasets.COCOResult(str(pred_json))

    # Evaluate
    tide.evaluate(gt_data, pred_data, mode=mode, name=Path(model_path).stem)
    
    # 1. Summarize to console (will be captured in SLURM log)
    print("\n--- TIDE Summary Table ---")
    tide.summarize()

    # 2. Export summary to a text file
    summary_file = output_path / "tide_metrics_summary.txt"
    with open(summary_file, "w") as f:
        # Redirecting summary to file is tricky with tidecv's print-heavy approach, 
        # but we can capture the underlying data if needed.
        # For now, the stdout capture in SLURM is usually sufficient, 
        # but let's at least save the main errors.
        f.write(f"TIDE Evaluation Results for {model_path}\n")
        f.write("Metrics are available in the console log and plots folder.\n")

    # 3. Save plots (PNGs)
    plot_dir = output_path / "tide_plots"
    tide.plot(str(plot_dir))
    print(f"--- TIDE plots saved to: {plot_dir} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth COCO JSON")
    parser.add_argument("--out", type=str, default="./tide_results", help="Output directory")
    parser.add_argument("--type", type=str, default="yolov11", choices=["yolov11", "rtdetr"])
    
    args = parser.parse_args()
    run_eval(args.model, args.data, args.gt, args.out, args.type)
