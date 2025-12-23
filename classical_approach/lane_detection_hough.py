#!/usr/bin/env python3
"""Classical lane detection with Canny + Hough Transform."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


class LaneDetector:
    def __init__(
        self,
        *,
        canny_low: int = 50,
        canny_high: int = 150,
        blur_kernel: int = 5,
        hough_rho: int = 2,
        hough_theta: float = np.pi / 180,
        hough_threshold: int = 50,
        hough_min_line_len: int = 40,
        hough_max_line_gap: int = 150,
        roi_top_ratio: float = 0.62,
        roi_top_width: float = 0.2,
        roi_bottom_width: float = 0.9,
        min_slope: float = 0.5,
        smooth_factor: float = 0.9,
        line_thickness: int = 8,
        overlay_alpha: float = 0.4,
    ) -> None:
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_line_len = hough_min_line_len
        self.hough_max_line_gap = hough_max_line_gap
        self.roi_top_ratio = roi_top_ratio
        self.roi_top_width = roi_top_width
        self.roi_bottom_width = roi_bottom_width
        self.min_slope = min_slope
        self.smooth_factor = smooth_factor
        self.line_thickness = line_thickness
        self.overlay_alpha = overlay_alpha

        self.prev_left: tuple[int, int, int, int] | None = None
        self.prev_right: tuple[int, int, int, int] | None = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        masked_edges = self._region_of_interest(edges)

        lines = cv2.HoughLinesP(
            masked_edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_line_len,
            maxLineGap=self.hough_max_line_gap,
        )

        left_line, right_line = self._average_lines(lines, frame.shape)
        left_line = self._smooth_line(left_line, self.prev_left)
        right_line = self._smooth_line(right_line, self.prev_right)

        if left_line is not None:
            self.prev_left = left_line
        if right_line is not None:
            self.prev_right = right_line

        return self._draw_lanes(frame, left_line, right_line)

    def _region_of_interest(self, img: np.ndarray) -> np.ndarray:
        height, width = img.shape[:2]
        half = 0.5
        top_half_width = self.roi_top_width / 2
        bottom_half_width = self.roi_bottom_width / 2

        top_y = int(height * self.roi_top_ratio)
        top_left = (int(width * (half - top_half_width)), top_y)
        top_right = (int(width * (half + top_half_width)), top_y)
        bottom_left = (int(width * (half - bottom_half_width)), height)
        bottom_right = (int(width * (half + bottom_half_width)), height)

        polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        mask = np.zeros_like(img)
        mask_color = 255 if len(mask.shape) == 2 else (255,) * mask.shape[2]
        cv2.fillPoly(mask, polygon, mask_color)
        return cv2.bitwise_and(img, mask)

    def _average_lines(
        self,
        lines: np.ndarray | None,
        frame_shape: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        if lines is None:
            return None, None

        height, width = frame_shape[:2]
        left: list[tuple[float, float, float]] = []
        right: list[tuple[float, float, float]] = []

        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < self.min_slope:
                continue
            intercept = y1 - slope * x1
            length = float(np.hypot(y2 - y1, x2 - x1))

            if slope < 0 and max(x1, x2) < width * 0.6:
                left.append((slope, intercept, length))
            elif slope > 0 and min(x1, x2) > width * 0.4:
                right.append((slope, intercept, length))

        left_line = self._make_line(left, height, width)
        right_line = self._make_line(right, height, width)
        return left_line, right_line

    def _make_line(
        self,
        lines: list[tuple[float, float, float]],
        height: int,
        width: int,
    ) -> tuple[int, int, int, int] | None:
        if not lines:
            return None

        slopes, intercepts, weights = zip(*lines)
        weight_sum = float(np.sum(weights))
        slope = float(np.dot(slopes, weights) / weight_sum)
        intercept = float(np.dot(intercepts, weights) / weight_sum)
        if slope == 0:
            return None

        y1 = height
        y2 = int(height * self.roi_top_ratio)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        x1 = int(np.clip(x1, 0, width - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        return (x1, y1, x2, y2)

    def _smooth_line(
        self,
        new_line: tuple[int, int, int, int] | None,
        prev_line: tuple[int, int, int, int] | None,
    ) -> tuple[int, int, int, int] | None:
        if new_line is None:
            return prev_line
        if prev_line is None:
            return new_line

        smoothed = []
        for prev_val, new_val in zip(prev_line, new_line):
            value = prev_val * self.smooth_factor + new_val * (1 - self.smooth_factor)
            smoothed.append(int(value))
        return tuple(smoothed)

    def _draw_lanes(
        self,
        frame: np.ndarray,
        left_line: tuple[int, int, int, int] | None,
        right_line: tuple[int, int, int, int] | None,
    ) -> np.ndarray:
        line_img = np.zeros_like(frame)

        if left_line is not None and right_line is not None:
            x1, y1, x2, y2 = left_line
            x3, y3, x4, y4 = right_line
            polygon = np.array([[(x1, y1), (x2, y2), (x4, y4), (x3, y3)]], dtype=np.int32)
            cv2.fillPoly(line_img, polygon, (0, 255, 0))

        if left_line is not None:
            cv2.line(
                line_img,
                (left_line[0], left_line[1]),
                (left_line[2], left_line[3]),
                (0, 255, 0),
                self.line_thickness,
            )
        if right_line is not None:
            cv2.line(
                line_img,
                (right_line[0], right_line[1]),
                (right_line[2], right_line[3]),
                (0, 255, 0),
                self.line_thickness,
            )

        return cv2.addWeighted(frame, 1.0, line_img, self.overlay_alpha, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classical lane detection with Canny + Hough Transform.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input video file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path (default: outputs/<input_stem>_classical.mp4).",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable writing output video.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the processed video in a window.",
    )
    parser.add_argument(
        "--canny-low",
        type=int,
        default=50,
        help="Lower threshold for Canny edge detection.",
    )
    parser.add_argument(
        "--canny-high",
        type=int,
        default=150,
        help="Upper threshold for Canny edge detection.",
    )
    parser.add_argument(
        "--hough-threshold",
        type=int,
        default=50,
        help="Threshold for Hough line detection.",
    )
    return parser.parse_args()


def run_video(
    input_path: Path,
    output_path: Path | None,
    *,
    show: bool,
    canny_low: int,
    canny_high: int,
    hough_threshold: int,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-2:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    detector = LaneDetector(
        canny_low=canny_low,
        canny_high=canny_high,
        hough_threshold=hough_threshold,
    )

    start = time.perf_counter()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = detector.process(frame)
        frame_count += 1

        if writer is not None:
            writer.write(processed)

        if show:
            cv2.imshow("Lane Detection (Classical)", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.perf_counter() - start
    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    print(
        f"Processed {frame_count} frames in {elapsed:.2f}s "
        f"({avg_fps:.2f} FPS)."
    )
    if output_path is not None:
        print(f"Saved output to: {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path: Path | None
    if args.no_output:
        output_path = None
    else:
        output_path = Path(args.output) if args.output else Path("outputs") / f"{input_path.stem}_classical.mp4"

    run_video(
        input_path,
        output_path,
        show=args.show,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        hough_threshold=args.hough_threshold,
    )


if __name__ == "__main__":
    main()
