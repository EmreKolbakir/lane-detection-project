#!/usr/bin/env python3
"""TuSimple U-Net inference script for lane detection."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# ============== U-Net Model Definition ==============
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


# ============== Inference Functions ==============
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TuSimple U-Net inference.")
    parser.add_argument("--input", default="../data/tusimple/training/frames/0313-1_13100.jpg", 
                        help="Path to input image or video.")
    parser.add_argument("--weight", default="weights/tusimple_unet_binary.pth", 
                        help="Path to trained .pth model.")
    parser.add_argument("--output", default=None, help="Output path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold.")
    parser.add_argument("--thin", type=int, default=5, 
                        help="Lane thinning amount (0=none, 1=slight, 2=moderate, 3=significant, 4+=very thin)")
    parser.add_argument("--show", action="store_true", help="Display result.")
    return parser.parse_args()


def load_model(weight_path: str, device: torch.device) -> UNet:
    model = UNet(n_channels=3, n_classes=2)
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    
    # Handle different state dict formats
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    
    # Remove 'module.' prefix if present
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
    # Resize
    img = cv2.resize(image, (target_size[1], target_size[0]))
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    # HWC to CHW
    img = img.transpose(2, 0, 1)
    # Add batch dimension
    return torch.from_numpy(img).unsqueeze(0)


def postprocess_mask(output: torch.Tensor, original_size: tuple, threshold: float = 0.5) -> np.ndarray:
    """Convert model output to binary mask."""
    # Output shape: [1, 2, H, W] - 2 classes (background, lane)
    # Take argmax or softmax for class 1 (lane)
    output = output.squeeze(0)  # [2, H, W]
    probs = torch.softmax(output, dim=0)  # [2, H, W]
    lane_prob = probs[1].cpu().numpy()  # [H, W] - lane class probability
    mask = (lane_prob > threshold).astype(np.uint8) * 255
    # Resize back to original size
    mask = cv2.resize(mask, (original_size[1], original_size[0]))
    return mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4) -> np.ndarray:
    """Overlay binary mask on image."""
    overlay = image.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
    ).astype(np.uint8)
    return overlay


def fill_lane_area(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.3) -> np.ndarray:
    """Fill the drivable lane area between detected lane lines."""
    overlay = image.copy()
    h, w = mask.shape[:2]
    
    # Find lane points for each row
    left_points = []
    right_points = []
    
    # Scan from bottom to top (more reliable at bottom)
    for y in range(h - 1, int(h * 0.4), -5):  # Skip top 40% of image
        row = mask[y, :]
        lane_pixels = np.where(row > 127)[0]
        
        if len(lane_pixels) > 5:  # Need enough pixels
            # Find clusters (left and right lane)
            center = w // 2
            left_px = lane_pixels[lane_pixels < center]
            right_px = lane_pixels[lane_pixels > center]
            
            if len(left_px) > 0:
                # İç kenar: sol şeridin EN SAĞ noktası (max)
                left_x = int(np.max(left_px))
                left_points.append((left_x, y))
            
            if len(right_px) > 0:
                # İç kenar: sağ şeridin EN SOL noktası (min)
                right_x = int(np.min(right_px))
                right_points.append((right_x, y))
    
    # Create polygon if we have both lanes
    if len(left_points) > 5 and len(right_points) > 5:
        # Combine points: left bottom to top, then right top to bottom
        polygon_pts = left_points + right_points[::-1]
        polygon = np.array([polygon_pts], dtype=np.int32)
        
        # Draw filled polygon
        lane_overlay = overlay.copy()
        cv2.fillPoly(lane_overlay, polygon, color)
        overlay = cv2.addWeighted(lane_overlay, alpha, overlay, 1 - alpha, 0)
    
    # Also draw the original mask on top (lane lines)
    mask_bool = mask > 127
    overlay[mask_bool] = (
        overlay[mask_bool] * 0.5 + np.array(color) * 0.5
    ).astype(np.uint8)
    
    return overlay


def temporal_smooth_mask(current_mask: np.ndarray, prev_mask: np.ndarray, smooth_factor: float = 0.7) -> np.ndarray:
    """Smooth mask between frames to reduce flickering."""
    if prev_mask is None:
        return current_mask
    
    # Weighted average
    smoothed = cv2.addWeighted(
        prev_mask.astype(np.float32), smooth_factor,
        current_mask.astype(np.float32), 1 - smooth_factor,
        0
    )
    return smoothed.astype(np.uint8)


def thin_lane_mask(mask: np.ndarray, erosion_size: int = 3) -> np.ndarray:
    """Thin the lane mask using morphological erosion.
    
    Args:
        mask: Binary lane mask (0 or 255)
        erosion_size: Size of erosion kernel (larger = thinner lanes)
                      1 = slight thinning
                      2 = moderate thinning  
                      3 = significant thinning
                      4+ = very thin
    """
    if erosion_size <= 0:
        return mask
    
    # Create erosion kernel
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (erosion_size * 2 + 1, erosion_size * 2 + 1)
    )
    
    # Apply erosion (shrinks white regions)
    thinned = cv2.erode(mask, kernel, iterations=1)
    
    return thinned


def process_image(model: UNet, image_path: str, device: torch.device, threshold: float = 0.5,
                  thin_amount: int = 3):
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    original_size = image.shape[:2]  # H, W
    
    # Preprocess
    input_tensor = preprocess_image(image).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Postprocess
    mask = postprocess_mask(output, original_size, threshold)
    
    # Thin the lane mask
    mask = thin_lane_mask(mask, thin_amount)
    
    result = fill_lane_area(image, mask)
    
    return result, mask, image


def process_video(model: UNet, video_path: str, output_path: str, device: torch.device, 
                  threshold: float = 0.5, show: bool = False, smooth_factor: float = 0.6,
                  thin_amount: int = 2):
    """Process a video file with temporal smoothing."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.perf_counter()
    prev_mask = None  # For temporal smoothing
    
    print(f"Processing {total_frames} frames (thin={thin_amount})...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        original_size = frame.shape[:2]
        input_tensor = preprocess_image(frame).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        mask = postprocess_mask(output, original_size, threshold)
        
        # Thin the lane mask
        mask = thin_lane_mask(mask, thin_amount)
        
        # Apply temporal smoothing
        mask = temporal_smooth_mask(mask, prev_mask, smooth_factor)
        prev_mask = mask.copy()
        
        # Fill lane area instead of just overlay
        result = fill_lane_area(frame, mask)
        
        writer.write(result)
        frame_count += 1
        
        if show:
            cv2.imshow("TuSimple U-Net", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        if frame_count % 50 == 0:
            elapsed = time.perf_counter() - start_time
            current_fps = frame_count / elapsed
            print(f"  Processed {frame_count}/{total_frames} frames ({current_fps:.1f} FPS)")
    
    elapsed = time.perf_counter() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    
    print(f"\nProcessed {frame_count} frames in {elapsed:.2f}s ({avg_fps:.2f} FPS)")
    print(f"Saved output to: {output_path}")


def main():
    args = parse_args()
    input_path = Path(args.input)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    
    # Device selection
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
    print("Model loaded successfully!")
    
    # Check if input is image or video
    suffix = input_path.suffix.lower()
    
    if suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process single image
        result, mask, original = process_image(model, str(input_path), device, args.threshold,
                                               thin_amount=args.thin)
        
        # Output path
        if args.output is None:
            output_path = f"../outputs/{input_path.stem}_unet_result.jpg"
        else:
            output_path = args.output
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"Saved result to: {output_path} (thin={args.thin})")
        
        if args.show:
            cv2.imshow("Original", original)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif suffix in [".mp4", ".avi", ".mov", ".mkv"]:
        # Process video
        if args.output is None:
            output_path = f"../outputs/{input_path.stem}_unet_deep_test_video.mp4"
        else:
            output_path = args.output
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        process_video(model, str(input_path), output_path, device, args.threshold, args.show,
                      thin_amount=args.thin)
    
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


if __name__ == "__main__":
    main()
