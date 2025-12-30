#!/usr/bin/env python3
"""Standalone UFLDv2 video inference script - no data module dependency."""
from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFLDv2 video inference.")
    parser.add_argument("--video", default="../data/tusimple/training/frames/0313-1_480.jpg", help="Path to input video.")
    parser.add_argument("--weight", default="weights/culane_res18.pth", help="Path to trained .pth model.")
    parser.add_argument("--config", default="configs/culane_res18.py", help="Config file path.")
    parser.add_argument("--output", default=None, help="Output video path.")
    parser.add_argument("--show", action="store_true", help="Display the video while processing.")
    return parser.parse_args()


class Config:
    """Minimal config loader."""
    
    @staticmethod
    def fromfile(filepath: str) -> "Config":
        import ast
        cfg = Config()
        with open(filepath, "r") as f:
            content = f.read()
        # Parse simple assignments
        for line in content.split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                try:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if value.startswith("[") or value.startswith("{") or value.startswith("("):
                        value = ast.literal_eval(value)
                    elif value.replace(".", "").replace("-", "").isdigit():
                        value = float(value) if "." in value else int(value)
                    elif value in ("True", "False"):
                        value = value == "True"
                    elif value.startswith("'") or value.startswith('"'):
                        value = ast.literal_eval(value)
                    setattr(cfg, key, value)
                except Exception:
                    pass
        return cfg


def load_config(config_path: str) -> Config:
    cfg = Config.fromfile(config_path)
    if cfg.dataset == "CULane":
        cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == "Tusimple":
        cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == "CurveLanes":
        cfg.row_anchor = np.linspace(0.4, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    return cfg


def get_model(cfg):
    """Import and create model based on dataset."""
    return importlib.import_module("model.model_" + cfg.dataset.lower()).get_model(cfg)


def load_model(cfg, weight_path: str, device: torch.device):
    net = get_model(cfg)
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    compatible_state = {}
    for key, value in state.items():
        if key.startswith("module."):
            compatible_state[key[7:]] = value
        else:
            compatible_state[key] = value

    net.load_state_dict(compatible_state, strict=False)
    net.to(device)
    net.eval()
    return net


def preprocess_frame(frame: np.ndarray, cfg) -> torch.Tensor:
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resize_h = int(cfg.train_height / cfg.crop_ratio)
    img = cv2.resize(img, (cfg.train_width, resize_h), interpolation=cv2.INTER_LINEAR)
    img = img[resize_h - cfg.train_height :, :, :]
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0)


def pred2coords(pred, row_anchor, col_anchor, local_width=1, original_image_width=1640, original_image_height=590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred["loc_row"].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred["loc_col"].shape

    max_indices_row = pred["loc_row"].argmax(1).cpu()
    valid_row = pred["exist_row"].argmax(1).cpu()
    max_indices_col = pred["loc_col"].argmax(1).cpu()
    valid_col = pred["exist_col"].argmax(1).cpu()

    pred["loc_row"] = pred["loc_row"].cpu()
    pred["loc_col"] = pred["loc_col"].cpu()

    coords = []
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:
                    all_ind = torch.tensor(
                        list(range(
                            max(0, max_indices_row[0, k, i] - local_width),
                            min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1,
                        ))
                    )
                    out_tmp = (pred["loc_row"][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = torch.tensor(
                        list(range(
                            max(0, max_indices_col[0, k, i] - local_width),
                            min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1,
                        ))
                    )
                    out_tmp = (pred["loc_col"][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords


def draw_lanes(frame: np.ndarray, coords: list, color=(0, 255, 0), thickness=5, fill_lane=True, fill_alpha=0.3) -> np.ndarray:
    vis = frame.copy()
    
    # İlk 2 şerit bizim gideceğimiz yol (sol ve sağ kenar)
    # Eğer her iki şerit de tespit edildiyse aralarını doldur
    if fill_lane and len(coords) >= 2:
        left_lane = coords[0]  # Sol şerit
        right_lane = coords[1]  # Sağ şerit
        
        if len(left_lane) > 1 and len(right_lane) > 1:
            # Polygon oluştur: sol şerit + sağ şerit (ters sırada)
            # Y koordinatlarına göre eşleştir
            left_sorted = sorted(left_lane, key=lambda p: p[1])
            right_sorted = sorted(right_lane, key=lambda p: p[1])
            
            # Polygon noktaları: sol yukarıdan aşağı, sağ aşağıdan yukarı
            polygon_pts = left_sorted + right_sorted[::-1]
            
            if len(polygon_pts) >= 3:
                polygon = np.array([polygon_pts], dtype=np.int32)
                # Yarı-şeffaf yeşil overlay
                overlay = vis.copy()
                cv2.fillPoly(overlay, polygon, color)
                vis = cv2.addWeighted(overlay, fill_alpha, vis, 1 - fill_alpha, 0)
    
    # Şerit çizgilerini çiz
    for lane in coords:
        if len(lane) > 1:
            for i in range(len(lane) - 1):
                cv2.line(vis, lane[i], lane[i + 1], color, thickness)
    
    return vis


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load config and model
    cfg = load_config(args.config)
    print(f"Loading model from {args.weight}...")
    net = load_model(cfg, args.weight, device)
    print("Model loaded successfully!")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video
    output_path = args.output
    if output_path is None:
        output_path = f"../outputs/{video_path.stem}_deep_test_video.mp4"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Process frames
    frame_count = 0
    start_time = time.perf_counter()

    print(f"Processing {total_frames} frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        input_tensor = preprocess_frame(frame, cfg).to(device)

        # Inference
        with torch.no_grad():
            pred = net(input_tensor)

        # Get lane coordinates
        coords = pred2coords(
            pred,
            cfg.row_anchor,
            cfg.col_anchor,
            original_image_width=width,
            original_image_height=height,
        )

        # Draw lanes
        vis_frame = draw_lanes(frame, coords)

        # Write output
        writer.write(vis_frame)
        frame_count += 1

        if args.show:
            cv2.imshow("UFLDv2 Lane Detection", vis_frame)
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
    if args.show:
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames in {elapsed:.2f}s ({avg_fps:.2f} FPS)")
    print(f"Saved output to: {output_path}")


if __name__ == "__main__":
    main()
