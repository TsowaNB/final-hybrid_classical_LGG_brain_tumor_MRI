"""
Step 5: Evaluation Script

Evaluate the trained Hybrid or Classical U-Net on validation and test splits.
Loads the saved H5 weights and rebuilds the architecture, sweeps thresholds on
the validation set to find the best binary mask threshold, and reports metrics.

Usage:
  python step5.py --dataset-path <path> \
                  --model-path models/hybrid_quantum_unet_best.weights.h5 \
                  --arch hybrid \
                  [--arch-json models/hybrid_quantum_unet_arch.json]

Outputs:
  - artifacts/eval_metrics.json: Summary of val/test metrics and best threshold
  - artifacts/threshold_sweep.csv: Dice vs threshold on validation set
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import BinaryIoU

from step2_preprocessing import MRIDataGenerator, prepare_dataset
# Import custom objects and architectures for clean model loading
from step4a import CombinedBCEDiceLoss, dice_coefficient
from step3_models import create_hybrid_quantum_unet, create_classical_unet, QuantumLayer
from tensorflow.keras.models import model_from_json


# Base dirs (auto-detect Kaggle)
KAGGLE_WORKING = Path('/kaggle/working')
BASE_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path('.')
ARTIFACTS_DIR = BASE_DIR / 'artifacts'


def ensure_dirs():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def dice_thresh_aggregate(y_true: np.ndarray, y_pred_prob: np.ndarray, thresh: float, smooth: float = 1.0) -> Tuple[float, float]:
    """Return intersection and union sums for Dice at given threshold.

    This aggregates across a batch for stable dataset-level metrics.
    """
    y_true_f = y_true.astype(np.float32).reshape(-1)
    y_pred_bin = (y_pred_prob >= thresh).astype(np.float32).reshape(-1)
    intersection = np.sum(y_true_f * y_pred_bin)
    union = np.sum(y_true_f) + np.sum(y_pred_bin)
    return intersection, union


def sweep_thresholds(model: tf.keras.Model, val_gen, thresholds: List[float]) -> Tuple[float, dict]:
    """Sweep thresholds on validation set; return best threshold and per-threshold Dice.

    Uses index-based access with predict_on_batch to avoid generator predict issues.
    """
    inter_sums = {t: 0.0 for t in thresholds}
    union_sums = {t: 0.0 for t in thresholds}

    # Warmup pass to stabilize GPU timers and kernel launches
    if len(val_gen) > 0:
        X0, _ = val_gen[0]
        _ = model.predict_on_batch(X0)

    for i in range(len(val_gen)):
        X_batch, y_batch = val_gen[i]
        y_pred_prob = model.predict_on_batch(X_batch)
        for t in thresholds:
            inter, uni = dice_thresh_aggregate(y_batch, y_pred_prob, t)
            inter_sums[t] += float(inter)
            union_sums[t] += float(uni)

    dice_map = {}
    for t in thresholds:
        dice_map[t] = (2.0 * inter_sums[t] + 1.0) / (union_sums[t] + 1.0)

    best_t = max(dice_map, key=dice_map.get)
    return best_t, dice_map


def evaluate_dataset(model: tf.keras.Model, gen, verbose: int = 0):
    """Evaluate model on a generator; returns dict with loss and compiled metrics."""
    results = model.evaluate(gen, verbose=verbose)
    # Map results according to model.metrics_names order
    names = model.metrics_names
    return {name: float(val) for name, val in zip(["loss"] + names, results)} if len(names) + 1 == len(results) else {
        "loss": float(results[0]),
        **{names[i]: float(results[i + 1]) for i in range(len(names))}
    }


def dice_at_threshold(model: tf.keras.Model, gen, thresh: float) -> float:
    """Compute aggregated thresholded Dice over a dataset generator.

    Uses index-based access with predict_on_batch for reliability.
    """
    inter_sum, union_sum = 0.0, 0.0
    for i in range(len(gen)):
        X_batch, y_batch = gen[i]
        y_pred_prob = model.predict_on_batch(X_batch)
        inter, uni = dice_thresh_aggregate(y_batch, y_pred_prob, thresh)
        inter_sum += float(inter)
        union_sum += float(uni)
    return (2.0 * inter_sum + 1.0) / (union_sum + 1.0)


def _derive_arch_json_path(weights_path: str) -> Path:
    p = Path(weights_path)
    name = p.name
    base = name
    if base.endswith("_best.weights.h5"):
        base = base.replace("_best.weights.h5", "")
    elif base.endswith("_final.weights.h5"):
        base = base.replace("_final.weights.h5", "")
    arch_name = f"{base}_arch.json"
    return p.parent / arch_name


def load_model(model_path: str, arch: str = "hybrid", arch_json: str | None = None) -> tf.keras.Model:
    """Load model from a path.

    - If `model_path` ends with `.weights.h5`, build the architecture and load weights.
    - Otherwise, load as a serialized Keras model with custom objects.
    """
    if model_path.endswith(".weights.h5"):
        # Try to reconstruct from architecture JSON
        candidate_json = Path(arch_json) if arch_json else _derive_arch_json_path(model_path)
        try:
            if candidate_json.exists():
                with open(candidate_json, "r", encoding="utf-8") as f:
                    arch_text = f.read()
                model = model_from_json(arch_text, custom_objects={"QuantumLayer": QuantumLayer})
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=CombinedBCEDiceLoss(),
                    metrics=[dice_coefficient, BinaryIoU()],
                )
                model.load_weights(model_path)
                print(f" Loaded architecture from JSON: {candidate_json}")
                return model
            else:
                print(f" Architecture JSON not found at {candidate_json}; falling back to factory '{arch}'.")
        except Exception as e:
            print(f" Warning: failed to load architecture JSON ({e}); falling back to factory '{arch}'.")

        # Fallback: rebuild via factory
        if arch.lower() == "classical":
            model = create_classical_unet()
        else:
            model = create_hybrid_quantum_unet()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=CombinedBCEDiceLoss(),
            metrics=[dice_coefficient, BinaryIoU()],
        )
        model.load_weights(model_path)
        return model
    custom_objects = {
        "CombinedBCEDiceLoss": CombinedBCEDiceLoss,
        "dice_coefficient": dice_coefficient,
        "BinaryIoU": BinaryIoU,
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate U-Net (Step 5)")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to LGG dataset (defaults to env LGG_DATASET_PATH)")
    parser.add_argument("--model-path", type=str, default=str(BASE_DIR / "models" / "hybrid_quantum_unet_best.weights.h5"), help="Path to saved model or weights (.weights.h5)")
    parser.add_argument("--arch", type=str, default="hybrid", choices=["hybrid", "classical"], help="Model architecture when loading weights-only")
    parser.add_argument("--arch-json", type=str, default=None, help="Optional path to architecture JSON to reconstruct the exact model")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Verbosity level for evaluation")
    parser.add_argument("--thresh-start", type=float, default=0.1, help="Threshold sweep start")
    parser.add_argument("--thresh-stop", type=float, default=0.9, help="Threshold sweep stop (inclusive)")
    parser.add_argument("--thresh-step", type=float, default=0.05, help="Threshold sweep step")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f" Warning: ignoring unknown CLI args: {unknown}")
    return args


def main(args=None):
    args = parse_args(args)
    ensure_dirs()

    # Resolve dataset path
    dataset_path = args.dataset_path or os.environ.get(
        "LGG_DATASET_PATH",
        "/kaggle/input/lgg-mri-segmentation/kaggle_3m",
    )

    # Dataset splits
    print(" Preparing dataset for evaluation...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(dataset_path)
    print(f" Val samples: {len(X_val)} | Test samples: {len(X_test)}")

    # Generators
    val_gen = MRIDataGenerator(X_val, y_val, batch_size=args.batch_size, augment=False)
    test_gen = MRIDataGenerator(X_test, y_test, batch_size=args.batch_size, augment=False)

    # Load model (fallback to final weights if best not found)
    model_path = Path(args.model_path)
    if not model_path.exists():
        fallback = BASE_DIR / "models" / (
            "hybrid_quantum_unet_final.h5" if args.arch == "hybrid" else "classical_unet_final.h5"
        )
        print(f" Model not found at {model_path}, falling back to {fallback}")
        model_path = fallback
    print(f" Loading model from: {model_path}")
    model = load_model(str(model_path), arch=args.arch, arch_json=args.arch_json)

    # Evaluate on validation and test using compiled metrics
    val_metrics = evaluate_dataset(model, val_gen, verbose=args.verbose)
    test_metrics = evaluate_dataset(model, test_gen, verbose=args.verbose)

    # Threshold sweep on validation
    thresholds = [round(t, 5) for t in np.arange(args.thresh_start, args.thresh_stop + 1e-9, args.thresh_step)]
    best_t, dice_map = sweep_thresholds(model, val_gen, thresholds)
    print(f" Best threshold on validation: {best_t:.3f} (Dice={dice_map[best_t]:.4f})")

    # Dice at best threshold on test
    test_dice_thresh = dice_at_threshold(model, test_gen, best_t)

    # Save metrics
    metrics_out = {
        "val": {
            **val_metrics,
            "best_threshold": float(best_t),
            "val_dice_thresh_best": float(dice_map[best_t]),
        },
        "test": {
            **test_metrics,
            "test_dice_at_best_threshold": float(test_dice_thresh),
        },
    }
    with open(ARTIFACTS_DIR / "eval_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f" Saved metrics to {ARTIFACTS_DIR / 'eval_metrics.json'}")

    # Save threshold sweep CSV
    sweep_path = ARTIFACTS_DIR / "threshold_sweep.csv"
    with open(sweep_path, "w") as f:
        f.write("threshold,dice\n")
        for t in thresholds:
            f.write(f"{t},{dice_map[t]:.6f}\n")
    print(f" Saved threshold sweep to {sweep_path}")


if __name__ == "__main__":
    main()