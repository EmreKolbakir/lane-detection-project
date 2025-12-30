#!/usr/bin/env python3
"""
Generate evaluation charts and visualizations for the report.
Uses TuSimple test set results.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results from our evaluations
RESULTS = {
    "U-Net": {
        "F1": 76.81,
        "Precision": 71.67,
        "Recall": 82.74,
        "mIoU": 62.35,
        "FPS": 15
    },
    "UFLDv2*": {  # Cross-dataset
        "F1": 50.62,
        "Precision": 56.39,
        "Recall": 45.92,
        "mIoU": 36.06,
        "FPS": 20
    },
    "Classical": {
        "F1": 7.20,
        "Precision": 64.43,
        "Recall": 3.81,
        "mIoU": 3.95,
        "FPS": 80
    }
}

def create_metrics_bar_chart(output_path: str):
    """Create bar chart comparing F1, Precision, Recall, IoU."""
    methods = list(RESULTS.keys())
    metrics = ["F1", "Precision", "Recall", "mIoU"]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    
    for i, method in enumerate(methods):
        values = [RESULTS[method][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=method, color=colors[i])
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Lane Detection Performance Comparison\n(TuSimple Test Set - 2782 images)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add footnote
    ax.text(0.02, -0.12, '*UFLDv2: Cross-dataset evaluation (trained on CULane, tested on TuSimple)', 
            transform=ax.transAxes, fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_fps_chart(output_path: str):
    """Create FPS comparison bar chart."""
    methods = list(RESULTS.keys())
    fps_values = [RESULTS[m]["FPS"] for m in methods]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(methods, fps_values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, val in zip(bars, fps_values):
        ax.annotate(f'{val} FPS',
                   xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add real-time line (30 FPS)
    ax.axhline(y=30, color='red', linestyle='--', linewidth=2, label='Real-time (30 FPS)')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Frames Per Second (FPS)', fontsize=12)
    ax.set_title('Processing Speed Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_accuracy_vs_speed_scatter(output_path: str):
    """Create scatter plot: Accuracy (F1) vs Speed (FPS)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    markers = ['o', 's', '^']
    
    for i, (method, data) in enumerate(RESULTS.items()):
        ax.scatter(data["FPS"], data["F1"], s=300, c=colors[i], marker=markers[i], 
                  label=method, edgecolors='black', linewidth=2, zorder=3)
        ax.annotate(method, (data["FPS"], data["F1"]), 
                   xytext=(10, 10), textcoords='offset points', fontsize=11, fontweight='bold')
    
    # Add real-time line
    ax.axvline(x=30, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Real-time threshold')
    
    ax.set_xlabel('Speed (FPS)', fontsize=12)
    ax.set_ylabel('F1 Score (%)', fontsize=12)
    ax.set_title('Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    
    # Add quadrant labels
    ax.text(60, 85, 'Ideal\n(Fast & Accurate)', ha='center', fontsize=10, 
            color='green', alpha=0.7, style='italic')
    ax.text(10, 85, 'Accurate\nbut Slow', ha='center', fontsize=10, 
            color='orange', alpha=0.7, style='italic')
    ax.text(60, 15, 'Fast but\nInaccurate', ha='center', fontsize=10, 
            color='orange', alpha=0.7, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_radar_chart(output_path: str):
    """Create radar/spider chart for multi-metric comparison."""
    metrics = ["F1", "Precision", "Recall", "mIoU", "Speed*"]
    
    # Normalize speed to 0-100 scale (max 100 FPS)
    data = {
        "U-Net": [76.81, 71.67, 82.74, 62.35, 15],
        "UFLDv2*": [50.62, 56.39, 45.92, 36.06, 20],
        "Classical": [7.20, 64.43, 3.81, 3.95, 80]
    }
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    for i, (method, values) in enumerate(data.items()):
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title('Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # Add footnote
    fig.text(0.5, 0.02, '*Speed normalized to percentage (100 FPS = 100%)', 
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(output_path: str):
    """Create a summary table as image."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Table data
    columns = ['Method', 'F1 (%)', 'Precision (%)', 'Recall (%)', 'mIoU (%)', 'FPS', 'Real-time']
    rows = [
        ['U-Net (TuSimple)', '76.81', '71.67', '82.74', '62.35', '~15', 'No'],
        ['UFLDv2 (CULane)*', '50.62', '56.39', '45.92', '36.06', '~20', 'No'],
        ['Classical (Hough)', '7.20', '64.43', '3.81', '3.95', '~80', 'Yes']
    ]
    
    # Create table
    table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i, col in enumerate(columns):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Color code rows
    colors_row = ['#d5f4e6', '#d6eaf8', '#fadbd8']  # Light green, blue, red
    for row_idx in range(1, 4):
        for col_idx in range(len(columns)):
            table[(row_idx, col_idx)].set_facecolor(colors_row[row_idx-1])
    
    # Highlight best values
    # Best F1
    table[(1, 1)].set_text_props(fontweight='bold')
    # Best Recall
    table[(1, 3)].set_text_props(fontweight='bold')
    # Best mIoU
    table[(1, 4)].set_text_props(fontweight='bold')
    # Best FPS
    table[(3, 5)].set_text_props(fontweight='bold')
    
    ax.set_title('Evaluation Results Summary\n(TuSimple Test Set - 2782 images)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Footnote
    fig.text(0.5, 0.05, '*UFLDv2: Cross-dataset evaluation (trained on CULane)', 
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = Path("outputs/report_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating report figures...")
    print("="*50)
    
    # Generate all charts
    create_metrics_bar_chart(str(output_dir / "metrics_comparison.png"))
    create_fps_chart(str(output_dir / "fps_comparison.png"))
    create_accuracy_vs_speed_scatter(str(output_dir / "accuracy_vs_speed.png"))
    create_radar_chart(str(output_dir / "radar_chart.png"))
    create_summary_table(str(output_dir / "summary_table.png"))
    
    print("="*50)
    print(f"All figures saved to: {output_dir}/")
    print("\nGenerated files:")
    for f in output_dir.glob("*.png"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
