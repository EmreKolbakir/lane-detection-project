#!/usr/bin/env python3
"""
TuSimple Evaluation Script for Classical Lane Detection (Canny + Hough)
Calculates: Accuracy, F1 Score, Precision, Recall, IoU
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
from tqdm import tqdm

# Import the LaneDetector class
from lane_detection_hough import LaneDetector


def lanes_to_mask(lanes: List[List[int]], h_samples: List[int], 
                  img_height: int = 720, img_width: int = 1280,
                  lane_width: int = 16) -> np.ndarray:
    """Convert TuSimple lane format to binary mask."""
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for lane in lanes:
        points = []
        for x, y in zip(lane, h_samples):
            if x >= 0:  # -2 means no lane at this y
                points.append((int(x), int(y)))
        
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(mask, points[i], points[i+1], 1, thickness=lane_width)
    
    return mask


def classical_to_mask(detector: LaneDetector, image: np.ndarray) -> np.ndarray:
    """Convert classical lane detection output to binary mask.
    
    The classical approach draws lines on the image. We need to extract
    the lane mask from it.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Get edges and color mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (detector.blur_kernel, detector.blur_kernel), 0)
    edges = cv2.Canny(blur, detector.canny_low, detector.canny_high)
    
    # Apply color mask if enabled
    if detector.use_color_mask:
        color_mask = detector._color_mask(image)
        edges = cv2.bitwise_and(edges, color_mask)
    
    # Apply ROI
    roi_edges = detector._region_of_interest(edges)
    
    # Detect lines
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=detector.hough_rho,
        theta=detector.hough_theta,
        threshold=detector.hough_threshold,
        minLineLength=detector.hough_min_line_len,
        maxLineGap=detector.hough_max_line_gap,
    )
    
    # Get averaged lane lines
    left_line, right_line = detector._average_lines(lines, image.shape)
    
    # Draw lines on mask
    line_thickness = 16  # Match GT lane width
    
    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=line_thickness)
    
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(mask, (x1, y1), (x2, y2), 1, thickness=line_thickness)
    
    return mask


def calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics between prediction and ground truth."""
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
    """Load TuSimple test annotations (JSONL format)."""
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            if line.strip():
                annotations.append(json.loads(line))
    return annotations


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Classical approach on TuSimple test set")
    parser.add_argument("--test-dir", default="../data/tusimple/test_set",
                        help="Path to TuSimple test set directory")
    parser.add_argument("--label", default="../data/tusimple/test_label.json",
                        help="Path to test_label.json")
    parser.add_argument("--lane-width", type=int, default=16,
                        help="Lane width for GT mask (pixels)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate (for quick testing)")
    parser.add_argument("--save-examples", type=int, default=10,
                        help="Number of example visualizations to save")
    parser.add_argument("--output-dir", default="../outputs/evaluation_classical",
                        help="Output directory for results")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup paths
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create classical detector
    print("Initializing Classical Lane Detector (Canny + Hough)...")
    detector = LaneDetector()
    
    # Load annotations
    print(f"Loading annotations from {args.label}...")
    annotations = load_test_annotations(args.label)
    print(f"Loaded {len(annotations)} test samples")
    
    if args.max_samples:
        annotations = annotations[:args.max_samples]
        print(f"Using first {args.max_samples} samples")
    
    # Aggregate metrics
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0
    
    all_metrics = []
    
    start_time = time.perf_counter()
    examples_saved = 0
    
    print("\nEvaluating Classical Approach...")
    for idx, ann in enumerate(tqdm(annotations)):
        img_path = test_dir / ann["raw_file"]
        if not img_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Get classical prediction mask
        pred_mask = classical_to_mask(detector, image)
        
        # Create GT mask
        gt_mask = lanes_to_mask(ann["lanes"], ann["h_samples"], h, w, args.lane_width)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        
        total_TP += metrics["TP"]
        total_FP += metrics["FP"]
        total_FN += metrics["FN"]
        total_TN += metrics["TN"]
        
        # Save example visualizations
        if examples_saved < args.save_examples:
            vis = image.copy()
            
            # Green = prediction, Red = GT, Yellow = overlap
            pred_color = np.zeros_like(image)
            pred_color[pred_mask > 0] = [0, 255, 0]
            
            gt_color = np.zeros_like(image)
            gt_color[gt_mask > 0] = [0, 0, 255]
            
            overlap = (pred_mask > 0) & (gt_mask > 0)
            
            vis = cv2.addWeighted(vis, 0.7, pred_color, 0.3, 0)
            vis = cv2.addWeighted(vis, 1.0, gt_color, 0.3, 0)
            vis[overlap] = [0, 255, 255]
            
            text = f"Classical | IoU: {metrics['IoU']:.3f} | F1: {metrics['F1']:.3f}"
            cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            save_path = output_dir / f"classical_example_{examples_saved:03d}.jpg"
            cv2.imwrite(str(save_path), vis)
            examples_saved += 1
    
    elapsed = time.perf_counter() - start_time
    
    # Calculate overall metrics
    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
    overall_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
    
    # Mean per-image metrics
    mean_precision = np.mean([m["Precision"] for m in all_metrics])
    mean_recall = np.mean([m["Recall"] for m in all_metrics])
    mean_f1 = np.mean([m["F1"] for m in all_metrics])
    mean_iou = np.mean([m["IoU"] for m in all_metrics])
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS - Classical (Canny+Hough) on TuSimple")
    print("="*60)
    print(f"\nTotal samples evaluated: {len(all_metrics)}")
    print(f"Evaluation time: {elapsed:.2f}s ({len(all_metrics)/elapsed:.1f} img/s)")
    print(f"Lane width (GT): {args.lane_width}px")
    
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
    
    print("\n--- Confusion Matrix ---")
    print(f"  TP: {total_TP:,}")
    print(f"  FP: {total_FP:,}")
    print(f"  FN: {total_FN:,}")
    print(f"  TN: {total_TN:,}")
    
    # Save results
    results = {
        "method": "Classical (Canny + Hough Transform)",
        "config": {
            "lane_width": args.lane_width,
            "samples": len(all_metrics),
            "canny_low": detector.canny_low,
            "canny_high": detector.canny_high,
            "hough_threshold": detector.hough_threshold
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
    
    results_path = output_dir / "classical_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Example visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
