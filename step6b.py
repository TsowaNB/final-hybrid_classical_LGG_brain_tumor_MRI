"""
Step 6b: Display Training Curves — show plots instead of saving.

This script mirrors Step 6 (training curves visualization) but displays the
figures on screen using plt.show() and does not save any files.

It loads Keras CSVLogger outputs and shows:
- Dice (train/val), Loss (train/val)
- Optional combined overlay when multiple logs are provided

Usage:
  python step6b.py --log-csv <path_to_log.csv> [--log-csv <another_log.csv>] [--tag NAME]

If no --log-csv is provided, it will try defaults:
  logs/hybrid_quantum_unet_train_log.csv
  logs/classical_unet_train_log.csv
"""

from __future__ import annotations

from pathlib import Path
import csv

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import pandas as pd
except Exception:
    pd = None


def _load_training_log(csv_path: str | Path) -> dict:
    """Load Keras CSVLogger output into a dict of lists.

    Supports pandas if available; falls back to Python csv module.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Log CSV not found: {csv_path}")
    if pd is not None:
        df = pd.read_csv(csv_path)
        return {k: df[k].tolist() for k in df.columns}
    # Fallback: csv module
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        keys = reader.fieldnames or []
        data = {k: [] for k in keys}
        for row in reader:
            for k in keys:
                val = row.get(k)
                try:
                    data[k].append(float(val))
                except Exception:
                    data[k].append(val)
        return data


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def plot_training_curves_display(log_csv: str | Path, tag: str | None = None) -> None:
    """Create figures for Dice, IoU, Loss (train/val) and display them."""
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Please install matplotlib.")
    data = _load_training_log(log_csv)
    cols = list(data.keys())
    epochs = data.get("epoch") or list(range(1, len(data.get("loss", [])) + 1))

    # Resolve metric columns with flexible naming
    dice_tr = _find_col(cols, ["dice_coefficient", "dice", "train_dice"])
    dice_va = _find_col(cols, ["val_dice_coefficient", "val_dice", "dice_val", "validation_dice"])
    # IoU visualization removed per request
    loss_tr = _find_col(cols, ["loss", "train_loss"])
    loss_va = _find_col(cols, ["val_loss", "validation_loss", "loss_val"])

    tag = tag or Path(log_csv).stem.replace("_train_log", "")

    # 1) Dice curves
    if dice_tr is not None and dice_va is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, data[dice_tr], label="Dice (train)", color="tab:blue")
        plt.plot(epochs, data[dice_va], label="Dice (val)", color="tab:orange")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Coefficient")
        plt.title(f"Dice vs Epochs [{tag}]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    # IoU curves removed

    # 3) Loss curves
    if loss_tr is not None and loss_va is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, data[loss_tr], label="Loss (train)", color="tab:purple")
        plt.plot(epochs, data[loss_va], label="Loss (val)", color="tab:brown")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Epochs [{tag}]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    # Combined figure if at least one pair exists
    if any(x is not None for x in (dice_tr, dice_va, loss_tr, loss_va)):
        import numpy as np
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Training Curves [{tag}]")
        # Dice
        ax = axes[0]
        if dice_tr is not None and dice_va is not None:
            ax.plot(epochs, data[dice_tr], label="train", color="tab:blue")
            ax.plot(epochs, data[dice_va], label="val", color="tab:orange")
        ax.set_title("Dice")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        # Loss
        ax = axes[1]
        if loss_tr is not None and loss_va is not None:
            ax.plot(epochs, data[loss_tr], label="train", color="tab:purple")
            ax.plot(epochs, data[loss_va], label="val", color="tab:brown")
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout(rect=(0, 0, 1, 0.95))


def plot_combined_training_curves_display(log_csvs: list[str | Path], tag: str | None = None) -> None:
    """Overlay Dice, IoU, and Loss curves for multiple logs on single graphs and display."""
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Please install matplotlib.")
    tag = tag or "combined"

    # Load all logs
    logs_data: list[tuple[str, dict]] = []
    for lp in log_csvs:
        p = Path(lp)
        if not p.exists():
            print(f" Warning: log not found, skipping: {p}")
            continue
        data = _load_training_log(p)
        label = p.stem.replace("_train_log", "")
        logs_data.append((label, data))

    if not logs_data:
        raise RuntimeError("No valid logs to plot.")

    palette = [
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
        "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    ]

    def plot_metric(metric_name: str, candidates_train: list[str], candidates_val: list[str], colors: list[str], title_suffix: str):
        plt.figure(figsize=(10, 6))
        color_idx = 0
        for label, data in logs_data:
            cols = list(data.keys())
            epochs = data.get("epoch") or list(range(1, len(data.get("loss", [])) + 1))
            tr = _find_col(cols, candidates_train)
            va = _find_col(cols, candidates_val)
            if tr is None and va is None:
                continue
            base_color = colors[color_idx % len(colors)]
            color_idx += 1
            if tr is not None:
                plt.plot(epochs, data[tr], label=f"{label} {metric_name} (train)", color=base_color, linestyle="-")
            if va is not None:
                plt.plot(epochs, data[va], label=f"{label} {metric_name} (val)", color=base_color, linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs Epochs [{tag}] (overlay) {title_suffix}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    # Dice
    plot_metric(
        metric_name="Dice",
        candidates_train=["dice_coefficient", "dice", "train_dice"],
        candidates_val=["val_dice_coefficient", "val_dice", "dice_val", "validation_dice"],
        colors=palette,
        title_suffix="",
    )

    # IoU visualization removed

    # Loss
    plot_metric(
        metric_name="Loss",
        candidates_train=["loss", "train_loss"],
        candidates_val=["val_loss", "validation_loss", "loss_val"],
        colors=palette,
        title_suffix="",
    )


def main(args=None):
    import argparse

    parser = argparse.ArgumentParser(description="Step 6b: Display training curves (Dice, Loss)")
    parser.add_argument("--log-csv", action="append", help="Path to CSVLogger file. Repeat to plot multiple logs.")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag for figure titles")
    args, unknown = parser.parse_known_args(args)
    if unknown:
        print(f"Ignoring unknown CLI args: {unknown}")

    logs = args.log_csv or []
    if not logs:
        candidates = [
            "logs/hybrid_quantum_unet_train_log.csv",
            "logs/classical_unet_train_log.csv",
        ]
        logs = [p for p in candidates if Path(p).exists()]
    if not logs:
        raise SystemExit("No log CSV provided or found. Use --log-csv to specify one or more files.")

    print("Displaying training graphs (no files will be saved)...")
    # Per-log graphs
    for log in logs:
        print(f" Plotting per-log curves for: {log}")
        plot_training_curves_display(log_csv=log, tag=args.tag)

    # Combined overlay graphs when multiple logs provided
    if len(logs) > 1:
        print(" Plotting combined overlay curves across logs...")
        plot_combined_training_curves_display(log_csvs=logs, tag=(args.tag or "combined"))

    # Show all figures at once
    if plt is None:
        raise RuntimeError("matplotlib is required for display.")
    print(" Showing figures now...")
    plt.show()


if __name__ == "__main__":
    main()