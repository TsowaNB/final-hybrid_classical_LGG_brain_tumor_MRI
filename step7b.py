"""
Step 7B: Test Evaluation (Classical U-Net)

Evaluate the trained Classical U-Net (from step4b) on the test split and
report pixel-wise Precision, Recall, Accuracy, F1, IoU, and Dice, plus a
classification_report and a metrics bar chart PNG.

Usage:
  python step7b.py --dataset-path <path> \
                   --model-path models/classical_unet_best.h5 \
                   [--threshold 0.5] [--batch-size 16]

Outputs:
  - artifacts/eval_metrics_classical_test.json
  - artifacts/classification_report_classical_test.json and .txt
  - artifacts/graphs/classical_test_metrics_bar.png
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import argparse
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.models import model_from_json
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from step2_preprocessing import MRIDataGenerator, prepare_dataset
from step3_models import create_classical_unet
from step4a import CombinedBCEDiceLoss, dice_coefficient


# Base dirs (auto-detect Kaggle)
KAGGLE_WORKING = Path('/kaggle/working')
BASE_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path('.')
ARTIFACTS_DIR = BASE_DIR / 'artifacts'


def ensure_dirs():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _derive_arch_json_path(weights_path: str) -> Path:
    p = Path(weights_path)
    base = p.name
    if base.endswith("_best.weights.h5"):
        base = base.replace("_best.weights.h5", "")
    elif base.endswith("_final.weights.h5"):
        base = base.replace("_final.weights.h5", "")
    arch_name = f"{base}_arch.json"
    return p.parent / arch_name


def load_model(model_path: str, arch_json: str | None = None) -> tf.keras.Model:
    """Load model from a path.

    - If `model_path` ends with `.weights.h5`, rebuild the classical U-Net and load weights, optionally from JSON.
    - Otherwise, load as a serialized full H5 model with custom objects.
    """
    # Register a minimal Cast layer for environments that serialize tf.cast as 'Cast'.
    @tf.keras.utils.register_keras_serializable(package="HCCN", name="Cast")
    class Cast(tf.keras.layers.Layer):
        def __init__(self, dtype="float32", **kwargs):
            super().__init__(dtype=dtype, **kwargs)
            try:
                self.target_dtype = tf.as_dtype(dtype)
            except Exception:
                self.target_dtype = tf.float32

        def call(self, inputs):
            return tf.cast(inputs, self.target_dtype)

        def get_config(self):
            config = super().get_config()
            config.update({"dtype": tf.as_dtype(self.target_dtype).name})
            return config

    if model_path.endswith(".weights.h5"):
        candidate_json = Path(arch_json) if arch_json else _derive_arch_json_path(model_path)
        try:
            if candidate_json.exists():
                with open(candidate_json, "r", encoding="utf-8") as f:
                    arch_text = f.read()
                model = model_from_json(arch_text)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=CombinedBCEDiceLoss(),
                    metrics=[dice_coefficient, BinaryIoU()],
                )
                model.load_weights(model_path)
                print(f" Loaded architecture from JSON: {candidate_json}")
                return model
            else:
                print(f" Architecture JSON not found at {candidate_json}; falling back to classical factory.")
        except Exception as e:
            print(f" Warning: failed to load architecture JSON ({e}); falling back to classical factory.")

        # Fallback: rebuild via factory
        model = create_classical_unet()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=CombinedBCEDiceLoss(),
            metrics=[dice_coefficient, BinaryIoU()],
        )
        model.load_weights(model_path)
        return model

    # Full model load
    custom_objects = {
        "CombinedBCEDiceLoss": CombinedBCEDiceLoss,
        "dice_coefficient": dice_coefficient,
        "BinaryIoU": BinaryIoU,
        "Cast": Cast,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def aggregate_confusion(y_true: np.ndarray, y_pred_prob: np.ndarray, thresh: float) -> Tuple[int, int, int, int]:
    """Compute TP, FP, TN, FN sums across a batch."""
    y_true_b = (y_true > 0.5).astype(np.uint8).reshape(-1)
    y_pred_b = (y_pred_prob >= thresh).astype(np.uint8).reshape(-1)
    tp = int(np.sum((y_true_b == 1) & (y_pred_b == 1)))
    tn = int(np.sum((y_true_b == 0) & (y_pred_b == 0)))
    fp = int(np.sum((y_true_b == 0) & (y_pred_b == 1)))
    fn = int(np.sum((y_true_b == 1) & (y_pred_b == 0)))
    return tp, fp, tn, fn


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
    }


def build_classification_report(tp: int, fp: int, tn: int, fn: int) -> dict:
    """Create a classification report dict for binary classes (0/1)."""
    eps = 1e-8
    support_1 = tp + fn
    support_0 = tn + fp
    total = support_0 + support_1

    p1 = tp / (tp + fp + eps)
    r1 = tp / (tp + fn + eps)
    f1_1 = 2 * p1 * r1 / (p1 + r1 + eps)

    p0 = tn / (tn + fn + eps)
    r0 = tn / (tn + fp + eps)
    f1_0 = 2 * p0 * r0 / (p0 + r0 + eps)

    accuracy = (tp + tn) / (total + eps)

    macro_precision = (p0 + p1) / 2.0
    macro_recall = (r0 + r1) / 2.0
    macro_f1 = (f1_0 + f1_1) / 2.0
    weighted_precision = (p0 * support_0 + p1 * support_1) / (total + eps)
    weighted_recall = (r0 * support_0 + r1 * support_1) / (total + eps)
    weighted_f1 = (f1_0 * support_0 + f1_1 * support_1) / (total + eps)

    return {
        "0": {"precision": float(p0), "recall": float(r0), "f1-score": float(f1_0), "support": int(support_0)},
        "1": {"precision": float(p1), "recall": float(r1), "f1-score": float(f1_1), "support": int(support_1)},
        "accuracy": float(accuracy),
        "macro avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1-score": float(macro_f1),
            "support": int(total),
        },
        "weighted avg": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1-score": float(weighted_f1),
            "support": int(total),
        },
    }


def format_classification_report(report: dict) -> str:
    lines = []
    header = f"{'class':>10} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    lines.append(header)
    for cls in ["0", "1"]:
        row = report[cls]
        lines.append(f"{cls:>10} {row['precision']:10.4f} {row['recall']:10.4f} {row['f1-score']:10.4f} {row['support']:10d}")
    lines.append("")
    lines.append(f"{'accuracy':>10} {'':>10} {'':>10} {report['accuracy']:10.4f} {report['weighted avg']['support']:10d}")
    for key in ["macro avg", "weighted avg"]:
        row = report[key]
        lines.append(f"{key:>10} {row['precision']:10.4f} {row['recall']:10.4f} {row['f1-score']:10.4f} {row['support']:10d}")
    return "\n".join(lines)


def plot_metrics_barchart(metrics: dict, out_path: Path):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting the bar chart.")
    names = ["precision", "recall", "accuracy", "f1", "iou", "dice"]
    values = [metrics[n] for n in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=[
        "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"
    ])
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Test Metrics (Classical U-Net)")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2.0, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_test(model: tf.keras.Model, test_gen, threshold: float = 0.5) -> Tuple[dict, Tuple[int, int, int, int]]:
    """Run test evaluation and compute metrics and confusion over the entire test set."""
    if len(test_gen) > 0:
        X0, _ = test_gen[0]
        _ = model.predict_on_batch(X0)
    tp_sum, fp_sum, tn_sum, fn_sum = 0, 0, 0, 0
    for i in range(len(test_gen)):
        X_batch, y_batch = test_gen[i]
        y_pred_prob = model.predict_on_batch(X_batch)
        tp, fp, tn, fn = aggregate_confusion(y_batch, y_pred_prob, thresh=threshold)
        tp_sum += tp
        fp_sum += fp
        tn_sum += tn
        fn_sum += fn
    return compute_metrics(tp_sum, fp_sum, tn_sum, fn_sum), (tp_sum, fp_sum, tn_sum, fn_sum)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Step 7B: Test evaluation for Classical U-Net")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to LGG dataset (defaults to env LGG_DATASET_PATH)")
    parser.add_argument("--model-path", type=str, default=str(BASE_DIR / "models" / "classical_unet_best.h5"), help="Path to saved model (.h5) or weights (.weights.h5)")
    parser.add_argument("--arch-json", type=str, default=None, help="Optional path to architecture JSON to reconstruct weights-only")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing predictions")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Verbosity level")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unknown CLI args: {unknown}")
    return args


def main(args=None):
    args = parse_args(args)
    ensure_dirs()

    dataset_path = args.dataset_path or os.environ.get(
        "LGG_DATASET_PATH",
        "/kaggle/input/lgg-mri-segmentation/kaggle_3m",
    )

    print(" Preparing dataset for test evaluation (Classical U-Net)...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(dataset_path)
    print(f" Test samples: {len(X_test)}")

    test_gen = MRIDataGenerator(X_test, y_test, batch_size=args.batch_size, augment=False)

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        fallback = BASE_DIR / "models" / "classical_unet_final.h5"
        print(f" Model not found at {model_path}, falling back to {fallback}")
        model_path = fallback
    print(f" Loading model from: {model_path}")
    model = load_model(str(model_path), arch_json=args.arch_json)

    # Evaluate
    metrics, conf = evaluate_test(model, test_gen, threshold=args.threshold)
    tp_sum, fp_sum, tn_sum, fn_sum = conf
    out_json = ARTIFACTS_DIR / "eval_metrics_classical_test.json"
    with open(out_json, "w") as f:
        json.dump({"threshold": args.threshold, "test": metrics}, f, indent=2)
    print(" Test metrics (threshold={:.3f}):".format(args.threshold))
    for k, v in metrics.items():
        print(f"  - {k}: {v:.6f}")
    print(f" Saved metrics to {out_json}")

    # Classification report
    report_dict = build_classification_report(tp_sum, fp_sum, tn_sum, fn_sum)
    report_txt = format_classification_report(report_dict)
    out_rep_json = ARTIFACTS_DIR / "classification_report_classical_test.json"
    out_rep_txt = ARTIFACTS_DIR / "classification_report_classical_test.txt"
    with open(out_rep_json, "w") as f:
        json.dump(report_dict, f, indent=2)
    with open(out_rep_txt, "w") as f:
        f.write(report_txt + "\n")
    print(" Classification report:")
    print(report_txt)
    print(f" Saved classification report to {out_rep_json} and {out_rep_txt}")

    # Bar chart
    graphs_dir = ARTIFACTS_DIR / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    bar_path = graphs_dir / "classical_test_metrics_bar.png"
    try:
        plot_metrics_barchart(metrics, bar_path)
        print(f" Saved metrics bar chart to {bar_path}")
    except Exception as e:
        print(f" Warning: could not plot metrics bar chart: {e}")


if __name__ == "__main__":
    main()