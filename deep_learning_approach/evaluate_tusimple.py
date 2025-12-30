#!/usr/bin/env python3
"""
TuSimple Evaluation Script for U-Net Lane Detection
Calculates: Accuracy, F1 Score, Precision, Recall, IoU
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ============== U-Net Model Definition (same as inference) ==============
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        
        # Encoder (down path)
        self.d1 = ConvBlock(n_channels, 64)
        self.d2 = ConvBlock(64, 128)
        self.d3 = ConvBlock(128, 256)
        self.d4 = ConvBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder (up path)
        self.u4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.c4 = ConvBlock(1024, 512)
        
        self.u3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c3 = ConvBlock(512, 256)
        
        self.u2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c2 = ConvBlock(256, 128)
        
        self.u1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c1 = ConvBlock(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        
        # Bottleneck
        bn = self.bottleneck(self.pool(d4))
        
        # Decoder
        u4 = self.u4(bn)
        u4 = torch.cat([u4, d4], dim=1)
        c4 = self.c4(u4)
        
        u3 = self.u3(c4)
        u3 = torch.cat([u3, d3], dim=1)
        c3 = self.c3(u3)
        
        u2 = self.u2(c3)
        u2 = torch.cat([u2, d2], dim=1)
        c2 = self.c2(u2)
        
        u1 = self.u1(c2)
        u1 = torch.cat([u1, d1], dim=1)
        c1 = self.c1(u1)
        
        return self.out(c1)


# ============== Helper Functions ==============
def load_model(weight_path: str, device: torch.device) -> UNet:
    """Load U-Net model."""
    model = UNet(n_channels=3, n_classes=2)
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image: np.ndarray, target_size=(256, 512)) -> torch.Tensor:
    """Preprocess image for U-Net input."""
    img = cv2.resize(image, (target_size[1], target_size[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def get_prediction_mask(model: UNet, image: np.ndarray, device: torch.device, 
                        threshold: float = 0.5) -> np.ndarray:
    """Get binary prediction mask from model."""
    original_size = image.shape[:2]  # H, W
    input_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get lane probability
    output = output.squeeze(0)  # [2, H, W]
    probs = torch.softmax(output, dim=0)
    lane_prob = probs[1].cpu().numpy()  # Lane class
    
    # Binary mask
    mask = (lane_prob > threshold).astype(np.uint8)
    
    # Resize to original size
    mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    
    return mask


def lanes_to_mask(lanes: List[List[int]], h_samples: List[int], 
                  img_height: int = 720, img_width: int = 1280,
                  lane_width: int = 16) -> np.ndarray:
    """Convert TuSimple lane format to binary mask.
    
    Args:
        lanes: List of lanes, each lane is list of x coordinates
        h_samples: Y coordinates for each point
        img_height: Image height
        img_width: Image width  
        lane_width: Width of lane line in pixels
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for lane in lanes:
        points = []
        for x, y in zip(lane, h_samples):
            if x >= 0:  # -2 means no lane at this y
                points.append((int(x), int(y)))
        
        if len(points) >= 2:
            # Draw lane as thick line
            for i in range(len(points) - 1):
                cv2.line(mask, points[i], points[i+1], 1, thickness=lane_width)
    
    return mask


def calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics between prediction and ground truth.
    
    Returns:
        Dictionary with: TP, FP, FN, TN, Precision, Recall, F1, IoU, Accuracy
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    # Basic counts
    TP = np.sum(pred & gt)
    FP = np.sum(pred & ~gt)
    FN = np.sum(~pred & gt)
    TN = np.sum(~pred & ~gt)
    
    # Metrics
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
    parser = argparse.ArgumentParser(description="Evaluate U-Net on TuSimple test set")
    parser.add_argument("--weight", default="weights/tusimple_unet_binary.pth",
                        help="Path to U-Net weights")
    parser.add_argument("--test-dir", default="../data/tusimple/test_set",
                        help="Path to TuSimple test set directory")
    parser.add_argument("--label", default="../data/tusimple/test_label.json",
                        help="Path to test_label.json")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Prediction threshold")
    parser.add_argument("--lane-width", type=int, default=16,
                        help="Lane width for GT mask (pixels)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to evaluate (for quick testing)")
    parser.add_argument("--save-examples", type=int, default=10,
                        help="Number of example visualizations to save")
    parser.add_argument("--output-dir", default="../outputs/evaluation",
                        help="Output directory for results")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup paths
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
    print(f"Loading model from {args.weight}...")
    model = load_model(args.weight, device)
    print("Model loaded!")
    
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
    
    # Per-image metrics for averaging
    all_metrics = []
    
    # Timing
    start_time = time.perf_counter()
    examples_saved = 0
    
    print("\nEvaluating...")
    for idx, ann in enumerate(tqdm(annotations)):
        # Load image
        img_path = test_dir / ann["raw_file"]
        if not img_path.exists():
            continue
        
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Get prediction
        pred_mask = get_prediction_mask(model, image, device, args.threshold)
        
        # Create GT mask
        gt_mask = lanes_to_mask(ann["lanes"], ann["h_samples"], h, w, args.lane_width)
        
        # Calculate metrics for this image
        metrics = calculate_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        
        # Aggregate counts
        total_TP += metrics["TP"]
        total_FP += metrics["FP"]
        total_FN += metrics["FN"]
        total_TN += metrics["TN"]
        
        # Save example visualizations
        if examples_saved < args.save_examples:
            vis = image.copy()
            
            # Green = prediction, Red = GT, Yellow = overlap
            pred_color = np.zeros_like(image)
            pred_color[pred_mask > 0] = [0, 255, 0]  # Green
            
            gt_color = np.zeros_like(image)
            gt_color[gt_mask > 0] = [0, 0, 255]  # Red
            
            overlap = (pred_mask > 0) & (gt_mask > 0)
            
            vis = cv2.addWeighted(vis, 0.7, pred_color, 0.3, 0)
            vis = cv2.addWeighted(vis, 1.0, gt_color, 0.3, 0)
            vis[overlap] = [0, 255, 255]  # Yellow for overlap
            
            # Add metrics text
            text = f"IoU: {metrics['IoU']:.3f} | F1: {metrics['F1']:.3f}"
            cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            save_path = output_dir / f"example_{examples_saved:03d}.jpg"
            cv2.imwrite(str(save_path), vis)
            examples_saved += 1
    
    elapsed = time.perf_counter() - start_time
    
    # Calculate overall metrics from aggregated counts
    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
    overall_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
    
    # Calculate mean of per-image metrics
    mean_precision = np.mean([m["Precision"] for m in all_metrics])
    mean_recall = np.mean([m["Recall"] for m in all_metrics])
    mean_f1 = np.mean([m["F1"] for m in all_metrics])
    mean_iou = np.mean([m["IoU"] for m in all_metrics])
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS - U-Net on TuSimple Test Set")
    print("="*60)
    print(f"\nTotal samples evaluated: {len(all_metrics)}")
    print(f"Evaluation time: {elapsed:.2f}s ({len(all_metrics)/elapsed:.1f} img/s)")
    print(f"\nThreshold: {args.threshold}")
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
    
    # Save results to file
    results = {
        "config": {
            "threshold": args.threshold,
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
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    print(f"Example visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
