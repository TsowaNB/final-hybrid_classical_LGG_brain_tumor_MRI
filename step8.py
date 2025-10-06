"""
Step 8B: Compare Predictions (Hybrid vs Classical)

Load both the Hybrid Quantum U-Net and Classical U-Net, sample random test
images, run predictions, and display a grid comparing:
  - Original image
  - Ground truth mask
  - Overlay with Hybrid prediction (GT=Green, Pred=Red, Both=Yellow)
  - Overlay with Classical prediction (GT=Green, Pred=Red, Both=Yellow)

No files are saved; the figure is displayed.

Usage:
  python step8b.py \
    --dataset-path <path> \
    [--hybrid-model models/hybrid_quantum_unet_best.h5] \
    [--classical-model models/classical_unet_best.h5] \
    [--hybrid-arch-json models/hybrid_quantum_unet_arch.json] \
    [--classical-arch-json models/classical_unet_arch.json] \
    [--num-samples 8] [--threshold 0.5] [--seed 42] \
    [--mark-index 5]
"""

from __future__ import annotations

import os
from pathlib import Path
import argparse
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.models import model_from_json
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import cv2
from step2_preprocessing import prepare_dataset, MRIPreprocessor
from step3_models import create_hybrid_quantum_unet, create_classical_unet, QuantumLayer
from step4a import CombinedBCEDiceLoss, dice_coefficient


# Base dirs (auto-detect Kaggle)
KAGGLE_WORKING = Path('/kaggle/working')
BASE_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path('.')


def _derive_arch_json_path(weights_path: str) -> Path:
    p = Path(weights_path)
    base = p.name
    if base.endswith("_best.weights.h5"):
        base = base.replace("_best.weights.h5", "")
    elif base.endswith("_final.weights.h5"):
        base = base.replace("_final.weights.h5", "")
    arch_name = f"{base}_arch.json"
    return p.parent / arch_name


def load_model_with_support(model_path: str, is_hybrid: bool, arch_json: str | None = None) -> tf.keras.Model:
    """Load model with robust H5/weights-only support and custom objects."""
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
                print(f" Architecture JSON not found at {candidate_json}; falling back to factory.")
        except Exception as e:
            print(f" Warning: failed to load architecture JSON ({e}); falling back to factory.")

        # Fallback: rebuild via factory
        model = create_hybrid_quantum_unet() if is_hybrid else create_classical_unet()
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
    if is_hybrid:
        custom_objects["QuantumLayer"] = QuantumLayer
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Step 8B: Compare predictions of Hybrid vs Classical")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to LGG dataset (defaults to env LGG_DATASET_PATH)")
    parser.add_argument("--hybrid-model", type=str, default=str(BASE_DIR / "models" / "hybrid_quantum_unet_best.h5"), help="Hybrid model path (.h5 or .weights.h5)")
    parser.add_argument("--classical-model", type=str, default=str(BASE_DIR / "models" / "classical_unet_best.h5"), help="Classical model path (.h5 or .weights.h5)")
    parser.add_argument("--hybrid-arch-json", type=str, default=None, help="Optional Hybrid architecture JSON (for weights-only)")
    parser.add_argument("--classical-arch-json", type=str, default=None, help="Optional Classical architecture JSON (for weights-only)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing predictions")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of random test images to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mark-index", type=int, default=None, help="Dataset index to highlight in the grid (e.g., 5)")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"Ignoring unknown CLI args: {unknown}")
    return args


def make_overlay(image: np.ndarray, mask_true: np.ndarray, mask_pred: np.ndarray) -> np.ndarray:
    img = np.squeeze(image)
    gt = (np.squeeze(mask_true) > 0.5).astype(np.uint8)
    pr = (np.squeeze(mask_pred) > 0.5).astype(np.uint8)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    rgb = np.stack([img, img, img], axis=-1)
    rgb[gt == 1, 1] = 1.0  # GT green
    rgb[pr == 1, 0] = 1.0  # Pred red
    both = (gt == 1) & (pr == 1)
    rgb[both] = [1.0, 1.0, 0.0]  # Yellow where both
    return rgb


def visualize_comparison(hybrid_model: tf.keras.Model, classical_model: tf.keras.Model, X_test, y_test, indices: List[int], threshold: float, mark_index: int | None):
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization.")
    preprocessor = MRIPreprocessor()
    rows = len(indices)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    # Add color legend at the top of the figure
    try:
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color='green', label='Ground Truth'),
            Patch(color='red', label='Prediction'),
            Patch(color='yellow', label='Overlap'),
        ]
        fig.legend(
            handles=legend_handles,
            loc='upper center',
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
            title='Color Legend',
        )
    except Exception:
        pass
    if rows == 1:
        axes = np.array([axes])
    for r, idx in enumerate(indices):
        # Load image and mask paths
        img_path = str(X_test[idx])
        mask_path = str(y_test[idx]) if y_test[idx] else ""

        # Read grayscale image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f" Failed to load image: {img_path}")
        # Read mask if present; otherwise create empty mask
        if mask_path and mask_path != "":
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros_like(image)
        else:
            mask = np.zeros_like(image)

        # Preprocess to match model input shape (128x128x1, float32)
        image_pp, mask_pp = preprocessor.preprocess_pair(image, mask, augment=False)
        x = np.expand_dims(image_pp, axis=0)  # (1, 128, 128, 1)
        y = mask_pp                             # (128, 128, 1)

        # Predict probabilities from both models
        y_h_prob = hybrid_model.predict_on_batch(x)[0]
        y_c_prob = classical_model.predict_on_batch(x)[0]
        y_h_pred = (y_h_prob >= threshold).astype(np.float32)
        y_c_pred = (y_c_prob >= threshold).astype(np.float32)

        ax0 = axes[r, 0]
        ax0.imshow(np.squeeze(image_pp), cmap='gray')
        ax0.set_title(f"Image #{idx}" + (" (marked)" if mark_index is not None and idx == mark_index else ""))
        ax0.axis('off')

        ax1 = axes[r, 1]
        ax1.imshow(np.squeeze(y), cmap='gray')
        ax1.set_title("Ground Truth")
        ax1.axis('off')

        ax2 = axes[r, 2]
        overlay_h = make_overlay(image_pp, y, y_h_pred)
        ax2.imshow(overlay_h)
        ax2.set_title("Hybrid Overlay")
        ax2.axis('off')

        ax3 = axes[r, 3]
        overlay_c = make_overlay(image_pp, y, y_c_pred)
        ax3.imshow(overlay_c)
        ax3.set_title("Classical Overlay")
        ax3.axis('off')

        # Highlight the entire row if this is the marked index
        if mark_index is not None and idx == mark_index:
            for c in range(cols):
                ax = axes[r, c]
                for spine in ax.spines.values():
                    spine.set_edgecolor('yellow')
                    spine.set_linewidth(2.5)

    # Leave space for the legend at the top
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    print(" Displaying comparison visualization window...")
    plt.show()


def main(args=None):
    args = parse_args(args)

    dataset_path = args.dataset_path or os.environ.get(
        "LGG_DATASET_PATH",
        "/kaggle/input/lgg-mri-segmentation/kaggle_3m",
    )
    print(" Preparing dataset for comparison...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(dataset_path)
    print(f" Test samples: {len(X_test)}")

    # Hybrid model path with fallback
    h_path = Path(args.hybrid_model)
    if not h_path.exists():
        fallback_h = BASE_DIR / "models" / "hybrid_quantum_unet_final.h5"
        print(f" Hybrid model not found at {h_path}, falling back to {fallback_h}")
        h_path = fallback_h
    print(f" Loading Hybrid model from: {h_path}")
    hybrid_model = load_model_with_support(str(h_path), is_hybrid=True, arch_json=args.hybrid_arch_json)

    # Classical model path with fallback
    c_path = Path(args.classical_model)
    if not c_path.exists():
        fallback_c = BASE_DIR / "models" / "classical_unet_final.h5"
        print(f" Classical model not found at {c_path}, falling back to {fallback_c}")
        c_path = fallback_c
    print(f" Loading Classical model from: {c_path}")
    classical_model = load_model_with_support(str(c_path), is_hybrid=False, arch_json=args.classical_arch_json)

    # Choose random indices
    rng = np.random.default_rng(args.seed)
    n = min(args.num_samples, len(X_test))
    indices = sorted(rng.choice(len(X_test), size=n, replace=False).tolist())
    # Ensure marked index is present if requested
    if args.mark_index is not None and 0 <= args.mark_index < len(X_test):
        if args.mark_index not in indices:
            # Replace the last index with the marked one to keep size n
            indices[-1] = args.mark_index
            indices = sorted(indices)
            print(f" Included marked index {args.mark_index} in visualization.")
    elif args.mark_index is not None:
        print(f" Warning: mark-index {args.mark_index} is out of range (0..{len(X_test)-1}).")
    print(f" Visualizing indices: {indices}")

    visualize_comparison(hybrid_model, classical_model, X_test, y_test, indices, args.threshold, args.mark_index)


if __name__ == "__main__":
    main()