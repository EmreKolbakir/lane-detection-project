#!/usr/bin/env python3
"""
Side-by-Side Comparison: Classical vs U-Net Lane Detection
Creates comparison videos showing both methods running on the same footage.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "classical_approach"))
sys.path.insert(0, str(Path(__file__).parent.parent / "deep_learning_approach"))

from lane_detection_hough import LaneDetector
from unet_inference import UNet, load_model as load_unet, preprocess_image as unet_preprocess


class SideBySideComparison:
    def __init__(self, unet_weight: str, device: str = "auto"):
        # Device setup
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load Classical detector
        print("Loading Classical detector (Canny + Hough)...")
        self.classical = LaneDetector()
        
        # Load U-Net model
        print(f"Loading U-Net model from {unet_weight}...")
        self.unet = load_unet(unet_weight, self.device)
        self.unet.eval()
        
        print("Both models loaded!")
    
    def process_classical(self, frame: np.ndarray) -> np.ndarray:
        """Process frame with Classical approach."""
        return self.classical.process(frame)
    
    def process_unet(self, frame: np.ndarray, threshold: float = 0.5, 
                     thin: int = 5) -> np.ndarray:
        """Process frame with U-Net."""
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # Preprocess
        input_tensor = unet_preprocess(frame).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.unet(input_tensor)
            if isinstance(output, dict):
                output = output['out']
            probs = torch.softmax(output, dim=1)
            mask = probs[0, 1].cpu().numpy()
        
        # Resize mask
        mask_resized = cv2.resize(mask, (w, h))
        binary_mask = (mask_resized > threshold).astype(np.uint8)
        
        # Apply morphological thinning
        if thin > 0:
            kernel = np.ones((thin, thin), np.uint8)
            binary_mask = cv2.erode(binary_mask, kernel, iterations=1)
        
        # Create green overlay
        overlay = result.copy()
        overlay[binary_mask > 0] = [0, 255, 0]
        result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
        
        return result
    
    def process_video(self, input_path: str, output_path: str, 
                      max_frames: int = None, show: bool = False):
        """Process video and create side-by-side comparison."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Output video: side by side (2x width)
        out_w = orig_w * 2
        out_h = orig_h
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        
        print(f"\nProcessing {total_frames} frames...")
        print(f"Input: {orig_w}x{orig_h} @ {fps:.1f} FPS")
        print(f"Output: {out_w}x{out_h} (side-by-side)")
        
        frame_count = 0
        classical_times = []
        unet_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            # Classical processing
            t0 = time.perf_counter()
            classical_result = self.process_classical(frame)
            classical_times.append(time.perf_counter() - t0)
            
            # U-Net processing
            t0 = time.perf_counter()
            unet_result = self.process_unet(frame)
            unet_times.append(time.perf_counter() - t0)
            
            # Add labels
            cv2.putText(classical_result, "Classical (Canny+Hough)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(classical_result, f"FPS: {1/classical_times[-1]:.0f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.putText(unet_result, "U-Net (Deep Learning)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(unet_result, f"FPS: {1/unet_times[-1]:.0f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Combine side by side
            combined = np.hstack([classical_result, unet_result])
            
            # Add dividing line
            cv2.line(combined, (orig_w, 0), (orig_w, orig_h), (255, 255, 255), 2)
            
            out.write(combined)
            
            if show:
                cv2.imshow("Comparison", cv2.resize(combined, (out_w // 2, out_h // 2)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        out.release()
        if show:
            cv2.destroyAllWindows()
        
        # Stats
        avg_classical_fps = 1 / np.mean(classical_times)
        avg_unet_fps = 1 / np.mean(unet_times)
        
        print(f"\n{'='*50}")
        print(f"COMPARISON COMPLETE")
        print(f"{'='*50}")
        print(f"Frames processed: {frame_count}")
        print(f"Output saved to: {output_path}")
        print(f"\nPerformance:")
        print(f"  Classical: {avg_classical_fps:.1f} FPS")
        print(f"  U-Net:     {avg_unet_fps:.1f} FPS")
        print(f"  Speedup:   {avg_classical_fps/avg_unet_fps:.1f}x faster (Classical)")


def main():
    parser = argparse.ArgumentParser(description="Side-by-side lane detection comparison")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--unet-weight", 
                        default="deep_learning_approach/weights/tusimple_unet_binary.pth",
                        help="Path to U-Net weights")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, mps")
    args = parser.parse_args()
    
    # Adjust paths if running from project root
    project_root = Path(__file__).parent.parent
    
    unet_weight = args.unet_weight
    if not Path(unet_weight).exists():
        unet_weight = project_root / args.unet_weight
    
    comparator = SideBySideComparison(str(unet_weight), args.device)
    comparator.process_video(args.input, args.output, args.max_frames, args.show)


if __name__ == "__main__":
    main()
