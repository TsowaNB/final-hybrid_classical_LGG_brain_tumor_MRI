"""
Step 7 Both: Save Combined Metrics Bar Chart (Hybrid vs Classical)

This mirrors step7_both_metric (display) but saves the grouped bar chart.
It loads metrics JSONs from step7a (hybrid) and step7b (classical) and saves
Precision, Recall, Accuracy, F1, IoU, and Dice side-by-side.

Usage:
  python step7_both_save.py \
    --hybrid-json artifacts/eval_metrics_hybrid_test.json \
    --classical-json artifacts/eval_metrics_classical_test.json \
    --out-path artifacts/graphs/step7_both_metric.png \
    [--title "Hybrid vs Classical: Test Metrics"]
"""

from __future__ import annotations

import json
from pathlib import Path
import argparse

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


DEFAULT_HYBRID_JSON = Path("artifacts/eval_metrics_hybrid_test.json")
DEFAULT_CLASSICAL_JSON = Path("artifacts/eval_metrics_classical_test.json")
DEFAULT_OUT_PATH = Path("artifacts/graphs/step7_both_metric.png")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Step 7 Both: Save combined metrics bar chart")
    parser.add_argument("--hybrid-json", type=str, default=str(DEFAULT_HYBRID_JSON), help="Path to hybrid metrics JSON")
    parser.add_argument("--classical-json", type=str, default=str(DEFAULT_CLASSICAL_JSON), help="Path to classical metrics JSON")
    parser.add_argument("--out-path", type=str, default=str(DEFAULT_OUT_PATH), help="Output PNG path for the chart")
    parser.add_argument("--title", type=str, default="Hybrid vs Classical: Test Metrics", help="Chart title")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unknown CLI args: {unknown}")
    return args


def load_metrics_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    # Support either {"test": {...}} or direct metrics dict
    return data.get("test", data)


def save_combined_barchart(hybrid: dict, classical: dict, title: str, out_path: Path):
    if plt is None:
        raise RuntimeError("matplotlib is required to plot the combined bar chart.")

    keys = ["precision", "recall", "accuracy", "f1", "iou", "dice"]
    labels = ["Precision", "Recall", "Accuracy", "F1", "IoU", "Dice"]
    h_vals = [float(hybrid.get(k, 0.0)) for k in keys]
    c_vals = [float(classical.get(k, 0.0)) for k in keys]

    x = list(range(len(keys)))
    width = 0.38

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    hb = plt.bar([i - width/2 for i in x], h_vals, width=width, label="Hybrid", color="tab:cyan")
    cb = plt.bar([i + width/2 for i in x], c_vals, width=width, label="Classical", color="tab:orange")

    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="best")

    for b, v in zip(hb, h_vals):
        plt.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for b, v in zip(cb, c_vals):
        plt.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved combined metrics bar chart to: {out_path}")


def main(args=None):
    args = parse_args(args)
    hybrid = load_metrics_json(Path(args.hybrid_json))
    classical = load_metrics_json(Path(args.classical_json))
    out_path = Path(args.out_path)

    try:
        save_combined_barchart(hybrid, classical, args.title, out_path)
    except Exception as e:
        print(f"Error: could not save combined bar chart: {e}")


if __name__ == "__main__":
    main()