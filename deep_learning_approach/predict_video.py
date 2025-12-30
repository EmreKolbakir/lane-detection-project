#!/usr/bin/env python3
"""Run UFLDv2 inference on a video and save the overlayed result."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.common import get_model
from utils.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFLDv2 video inference.")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video.",
    )
    parser.add_argument(
        "--weight",
        required=True,
        help="Path to trained .pth model.",
    )
    parser.add_argument(
        "--config",
        default="configs/culane_res18.py",
        help="Config file path.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path (default: ../outputs/<name>_ufldv2.mp4).",
    )
    return parser.parse_args()


def load_cfg(config_path: str):
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
    else:
        raise NotImplementedError(f"Unsupported dataset: {cfg.dataset}")
    return cfg


def load_model(cfg, weight_path: str, device: torch.device):
    net = get_model(cfg)
    state = torch.load(weight_path, map_location="cpu")
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


def pred2coords(
    pred,
    row_anchor,
    col_anchor,
    local_width: int = 1,
    original_image_width: int = 1640,
    original_image_height: int = 590,
):
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
                        list(
                            range(
                                max(0, max_indices_row[0, k, i] - local_width),
                                min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1,
                            )
                        )
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
                        list(
                            range(
                                max(0, max_indices_col[0, k, i] - local_width),
                                min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1,
                            )
                        )
                    )

                    out_tmp = (pred["loc_col"][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords


def draw_lanes(frame: np.ndarray, lanes: list[list[tuple[int, int]]]) -> np.ndarray:
    overlay = frame.copy()
    lane_groups = []

    for lane in lanes:
        if len(lane) < 2:
            continue
        lane_sorted = sorted(lane, key=lambda p: p[1])
        bottom_x = lane_sorted[-1][0]
        lane_groups.append((bottom_x, lane_sorted))
        for i in range(1, len(lane_sorted)):
            cv2.line(overlay, lane_sorted[i - 1], lane_sorted[i], (0, 255, 0), 5)

    if len(lane_groups) <= 2:
        return cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)

    center_x = frame.shape[1] * 0.5
    left_lane = None
    right_lane = None
    for x_pos, lane_sorted in lane_groups:
        if x_pos <= center_x and (left_lane is None or x_pos > left_lane[0]):
            left_lane = (x_pos, lane_sorted)
        if x_pos >= center_x and (right_lane is None or x_pos < right_lane[0]):
            right_lane = (x_pos, lane_sorted)

    selected = []
    if left_lane and right_lane and left_lane[1] != right_lane[1]:
        selected = [left_lane[1], right_lane[1]]
    else:
        lane_groups.sort(key=lambda item: abs(item[0] - center_x))
        selected = [lane for _, lane in lane_groups[:2]]

    overlay = frame.copy()
    for lane_sorted in selected:
        for i in range(1, len(lane_sorted)):
            cv2.line(overlay, lane_sorted[i - 1], lane_sorted[i], (0, 255, 0), 5)

    return cv2.addWeighted(frame, 1.0, overlay, 0.35, 0)


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model(cfg, args.weight, device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.output) if args.output else Path("../outputs") / (Path(args.video).stem + "_ufldv2.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))

    frame_count = 0
    start = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(frame, cfg).to(device, non_blocking=True)
        with torch.no_grad():
            pred = net(input_tensor)

        coords = pred2coords(
            pred,
            cfg.row_anchor,
            cfg.col_anchor,
            original_image_width=width,
            original_image_height=height,
        )
        vis = draw_lanes(frame, coords)

        frame_count += 1
        elapsed = time.perf_counter() - start
        if elapsed > 0:
            fps = frame_count / elapsed
        cv2.putText(
            vis,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        writer.write(vis)

    cap.release()
    writer.release()
    print(f"Saved output to: {out_path}")


if __name__ == "__main__":
    main()
