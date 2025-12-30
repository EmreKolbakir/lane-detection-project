#!/usr/bin/env python3
"""
TuSimple Evaluation Script for UFLDv2 (Cross-Dataset Evaluation)
NOTE: UFLDv2 was trained on CULane, testing on TuSimple (different dataset!)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Import from inference_video.py
from ufldv2_inference import load_config, load_model, preprocess_frame, pred2coords


def lanes_to_mask(lanes: List[List[int]], h_samples: List[int], 
                  img_height: int = 720, img_width: int = 1280,
                  lane_width: int = 16) -> np.ndarray:
    """Convert TuSimple lane format to binary mask."""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for lane in lanes:
        points = []
        for x, y in zip(lane, h_samples):
            if x >= 0:
                points.append((int(x), int(y)))
        
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(mask, points[i], points[i+1], 1, thickness=lane_width)
    
    return mask


def coords_to_mask(coords: list, img_height: int = 720, img_width: int = 1280,
                   lane_width: int = 16) -> np.ndarray:
    """Convert UFLDv2 coords to binary mask."""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for lane in coords:
        if len(lane) >= 2:
            for i in range(len(lane) - 1):
                pt1 = (int(lane[i][0]), int(lane[i][1]))
                pt2 = (int(lane[i+1][0]), int(lane[i+1][1]))
                # Clip to image bounds
                pt1 = (max(0, min(img_width-1, pt1[0])), max(0, min(img_height-1, pt1[1])))
                pt2 = (max(0, min(img_width-1, pt2[0])), max(0, min(img_height-1, pt2[1])))
                cv2.line(mask, pt1, pt2, 1, thickness=lane_width)
    
    return mask


def calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    TP = np.sum(pred & gt)
    FP = np.sum(pred & ~gt)
    FN = np.sum(~pred & gt)
    TN = np.sum(~pred & ~gt)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    
    return {
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "IoU": iou,
        "Accuracy": accuracy
    }


def load_test_annotations(label_path: str) -> List[Dict]:
    """Load TuSimple test annotations."""
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    return annotations


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UFLDv2 on TuSimple (Cross-Dataset)")
    parser.add_argument("--weight", default="weights/culane_res18.pth",
                        help="Path to UFLDv2 weights")
    parser.add_argument("--config", default="configs/culane_res18.py",
                        help="Path to config file")
    parser.add_argument("--test-dir", default="../data/tusimple/test_set",
                        help="Path to TuSimple test set")
    parser.add_argument("--label", default="../data/tusimple/test_label.json",
                        help="Path to test_label.json")
    parser.add_argument("--lane-width", type=int, default=16,
                        help="Lane width for masks")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate")
    parser.add_argument("--save-examples", type=int, default=10,
                        help="Number of examples to save")
    parser.add_argument("--output-dir", default="../outputs/evaluation_ufldv2",
                        help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading UFLDv2 config from {args.config}...")
    cfg = load_config(args.config)
    
    print(f"Loading UFLDv2 model from {args.weight}...")
    model = load_model(cfg, args.weight, device)
    print("Model loaded!")
    print(f"NOTE: Model trained on {cfg.dataset}, testing on TuSimple (CROSS-DATASET)")
    
    # Load annotations
    print(f"Loading annotations from {args.label}...")
    annotations = load_test_annotations(args.label)
    print(f"Loaded {len(annotations)} test samples")
    
    if args.max_samples:
        annotations = annotations[:args.max_samples]
        print(f"Using first {args.max_samples} samples")
    
    # Metrics
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0
    all_metrics = []
    
    start_time = time.perf_counter()
    examples_saved = 0
    
    print("\nEvaluating UFLDv2 (CULane model on TuSimple)...")
    for idx, ann in enumerate(tqdm(annotations)):
        img_path = test_dir / ann["raw_file"]
        if not img_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # UFLDv2 inference
        input_tensor = preprocess_frame(image, cfg).to(device)
        
        with torch.no_grad():
            pred = model(input_tensor)
        
        # Get lane coordinates
        coords = pred2coords(
            pred, cfg.row_anchor, cfg.col_anchor,
            original_image_width=w, original_image_height=h
        )
        
        # Convert to mask
        pred_mask = coords_to_mask(coords, h, w, args.lane_width)
        
        # GT mask
        gt_mask = lanes_to_mask(ann["lanes"], ann["h_samples"], h, w, args.lane_width)
        
        # Metrics
        metrics = calculate_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        
        total_TP += metrics["TP"]
        total_FP += metrics["FP"]
        total_FN += metrics["FN"]
        total_TN += metrics["TN"]
        
        # Save examples
        if examples_saved < args.save_examples:
            vis = image.copy()
            
            pred_color = np.zeros_like(image)
            pred_color[pred_mask > 0] = [0, 255, 0]
            
            gt_color = np.zeros_like(image)
            gt_color[gt_mask > 0] = [0, 0, 255]
            
            overlap = (pred_mask > 0) & (gt_mask > 0)
            
            vis = cv2.addWeighted(vis, 0.7, pred_color, 0.3, 0)
            vis = cv2.addWeighted(vis, 1.0, gt_color, 0.3, 0)
            vis[overlap] = [0, 255, 255]
            
            text = f"UFLDv2 (CULane) | IoU: {metrics['IoU']:.3f} | F1: {metrics['F1']:.3f}"
            cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            save_path = output_dir / f"ufldv2_example_{examples_saved:03d}.jpg"
            cv2.imwrite(str(save_path), vis)
            examples_saved += 1
    
    elapsed = time.perf_counter() - start_time
    
    # Overall metrics
    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
    overall_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
    
    mean_precision = np.mean([m["Precision"] for m in all_metrics])
    mean_recall = np.mean([m["Recall"] for m in all_metrics])
    mean_f1 = np.mean([m["F1"] for m in all_metrics])
    mean_iou = np.mean([m["IoU"] for m in all_metrics])
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS - UFLDv2 (CULane-trained) on TuSimple Test")
    print("⚠️  CROSS-DATASET EVALUATION - Results may be lower than expected!")
    print("="*70)
    print(f"\nTotal samples evaluated: {len(all_metrics)}")
    print(f"Evaluation time: {elapsed:.2f}s ({len(all_metrics)/elapsed:.1f} img/s)")
    
    print("\n--- Overall Metrics (Aggregated) ---")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall:    {overall_recall:.4f}")
    print(f"  F1 Score:  {overall_f1:.4f}")
    print(f"  IoU:       {overall_iou:.4f}")
    print(f"  Accuracy:  {overall_accuracy:.4f}")
    
    print("\n--- Mean Per-Image Metrics ---")
    print(f"  Precision: {mean_precision:.4f}")
    print(f"  Recall:    {mean_recall:.4f}")
    print(f"  F1 Score:  {mean_f1:.4f}")
    print(f"  IoU (mIoU): {mean_iou:.4f}")
    
    # Save results
    results = {
        "method": "UFLDv2 (CULane pre-trained)",
        "note": "CROSS-DATASET: Model trained on CULane, tested on TuSimple",
        "config": {
            "trained_on": cfg.dataset,
            "tested_on": "TuSimple",
            "lane_width": args.lane_width,
            "samples": len(all_metrics)
        },
        "overall_metrics": {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1_score": overall_f1,
            "iou": overall_iou,
            "accuracy": overall_accuracy
        },
        "mean_metrics": {
            "precision": mean_precision,
            "recall": mean_recall,
            "f1_score": mean_f1,
            "mIoU": mean_iou
        },
        "confusion_matrix": {
            "TP": total_TP,
            "FP": total_FP,
            "FN": total_FN,
            "TN": total_TN
        }
    }
    
    results_path = output_dir / "ufldv2_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
