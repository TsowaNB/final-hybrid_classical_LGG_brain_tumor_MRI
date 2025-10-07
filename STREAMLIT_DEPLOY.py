"""
Streamlit Deploy: Hybrid Quantum U-Net — Train, Evaluate, Visualize

This app lets you:
- Train the Hybrid Quantum U-Net (step4a.train).
- Evaluate on the test split (metrics from step7a).
- Visualize predictions on test samples with color overlays.

Run:
  streamlit run STREAMLIT_DEPLOY.py
"""

from __future__ import annotations

import os
from pathlib import Path
import csv
from typing import List

import numpy as np
import streamlit as st
import tensorflow as tf

from step2_preprocessing import prepare_dataset, MRIPreprocessor, MRIDataGenerator
from step3_models import create_hybrid_quantum_unet, QuantumLayer
from step4a import train as hybrid_train, CombinedBCEDiceLoss, dice_coefficient
from step7a import load_model as load_hybrid_model, evaluate_test as evaluate_hybrid_test


st.set_page_config(page_title="Hybrid Quantum U-Net", layout="wide", initial_sidebar_state="collapsed")


def make_overlay(image: np.ndarray, mask_true: np.ndarray, mask_pred: np.ndarray) -> np.ndarray:
    img = np.squeeze(image)
    gt = (np.squeeze(mask_true) > 0.5).astype(np.uint8)
    pr = (np.squeeze(mask_pred) > 0.5).astype(np.uint8)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    rgb = np.stack([img, img, img], axis=-1)
    rgb[gt == 1, 1] = 1.0  # Ground Truth = Green
    rgb[pr == 1, 0] = 1.0  # Prediction = Red
    both = (gt == 1) & (pr == 1)
    rgb[both] = [1.0, 1.0, 0.0]  # Overlap = Yellow
    return (rgb * 255).astype(np.uint8)


def load_test_generators(dataset_path: str, batch_size: int = 16):
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(dataset_path)
    val_gen = MRIDataGenerator(X_val, y_val, batch_size=batch_size, augment=False)
    test_gen = MRIDataGenerator(X_test, y_test, batch_size=batch_size, augment=False)
    return (X_test, y_test, val_gen, test_gen)


def _best_threshold_from_csv() -> float | None:
    """Read threshold sweep and return best threshold by Dice, if available."""
    candidates = [Path("threshold_sweep_hybrid.csv"), Path("artifacts/threshold_sweep.csv")]
    for csv_path in candidates:
        try:
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    best_t, best_d = None, float("-inf")
                    for row in reader:
                        t_raw, d_raw = row.get("threshold"), row.get("dice")
                        if t_raw is None or d_raw is None:
                            continue
                        try:
                            t = float(t_raw)
                            d = float(d_raw)
                        except Exception:
                            continue
                        if d > best_d:
                            best_d, best_t = d, t
                    if best_t is not None:
                        return best_t
        except Exception:
            # fail silent; return None to use default
            pass
    return None


def _default_model_path() -> Path:
    """Prefer an existing model file: best.h5 -> best.weights.h5 -> final.h5."""
    best_h5 = Path("models/hybrid_quantum_unet_best.h5")
    best_weights = Path("models/hybrid_quantum_unet_best.weights.h5")
    final_h5 = Path("models/hybrid_quantum_unet_final.h5")
    if best_h5.exists():
        return best_h5
    if best_weights.exists():
        return best_weights
    return final_h5


def _resolve_model_path(user_path: str) -> str:
    """Return a usable model path, falling back to default if the user path is missing."""
    up = Path(user_path) if user_path else Path("")
    if up and up.exists():
        return str(up)
    dp = _default_model_path()
    return str(dp)


def sidebar_controls():
    st.sidebar.header("Configuration")
    # Prefer a local dataset folder if present; otherwise use env var or Kaggle default
    default_dataset = (
        str(Path("kaggle_3m")) if Path("kaggle_3m").exists()
        else os.environ.get("LGG_DATASET_PATH", "/kaggle/input/lgg-mri-segmentation/kaggle_3m")
    )
    dataset_path = st.sidebar.text_input(
        "Dataset path",
        value=default_dataset,
        help="Root folder containing images and masks (LGG dataset).",
    )
    if Path(dataset_path).exists():
        st.sidebar.success("Dataset folder found.")
    else:
        st.sidebar.warning("Dataset folder not found. Please check the path.")
    batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=64, value=16)
    epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=200, value=10)
    augment = st.sidebar.checkbox("Augment", value=True)
    learning_rate = st.sidebar.number_input("Learning rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%f")
    best_t = _best_threshold_from_csv()
    threshold = st.sidebar.slider(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(best_t) if best_t is not None else 0.5,
        step=0.01,
        help=("Auto-set from sweep: " + f"{best_t:.2f}" if best_t is not None else "Default 0.5; adjust as needed"),
    )

    st.sidebar.markdown("---")
    # Prefer existing model files
    default_model_path = _default_model_path()

    model_path = st.sidebar.text_input(
        "Hybrid model path",
        value=str(default_model_path),
        help="Full .h5 or weights-only .weights.h5",
    )
    if not Path(model_path).exists():
        fallback = _default_model_path()
        if fallback.exists():
            st.sidebar.info(f"Model not found; will use fallback: {fallback}")
        else:
            st.sidebar.warning("No model files found under models/. Training may be required.")

    # Set JSON only if present; otherwise leave blank (handled gracefully)
    arch_json_default = Path("models/hybrid_quantum_unet_arch.json")
    arch_json = st.sidebar.text_input(
        "Architecture JSON (optional)",
        value=str(arch_json_default) if arch_json_default.exists() else "",
        help="Required only for weights-only models if JSON exists",
    )
    return dataset_path, batch_size, epochs, augment, learning_rate, threshold, model_path, arch_json


def section_train(dataset_path: str, batch_size: int, epochs: int, augment: bool, learning_rate: float):
    st.header("Train Hybrid Quantum U-Net")
    st.write("Runs `step4a.train` and exports deployment artifacts under `models/`.")
    train_btn = st.button("Start Training")
    if train_btn:
        with st.spinner("Training hybrid model..."):
            try:
                history = hybrid_train(
                    dataset_path=dataset_path,
                    batch_size=batch_size,
                    epochs=epochs,
                    augment=augment,
                    learning_rate=learning_rate,
                    verbose=1,
                )
                st.success("Training completed. Artifacts saved under `models/`. Logs under `artifacts/`.\nCheck console for details.")
                # Show simple curves if available in history
                if isinstance(history, dict):
                    col1, col2 = st.columns(2)
                    with col1:
                        if "dice_coefficient" in history and "val_dice_coefficient" in history:
                            st.line_chart({
                                "Dice (train)": history["dice_coefficient"],
                                "Dice (val)": history["val_dice_coefficient"],
                            })
                    with col2:
                        if "loss" in history and "val_loss" in history:
                            st.line_chart({
                                "Loss (train)": history["loss"],
                                "Loss (val)": history["val_loss"],
                            })
            except Exception as e:
                st.error(f"Training failed: {e}")


def section_evaluate(dataset_path: str, batch_size: int, threshold: float, model_path: str, arch_json: str):
    st.header("Evaluate on Test Split")
    st.write("Loads the hybrid model and computes metrics over the test set.")
    eval_btn = st.button("Run Evaluation")
    if eval_btn:
        try:
            with st.spinner("Preparing dataset..."):
                X_test, y_test, val_gen, test_gen = load_test_generators(dataset_path, batch_size=batch_size)
            # Resolve model path with fallback
            effective_model_path = _resolve_model_path(model_path)
            if not Path(effective_model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            with st.spinner("Loading model..."):
                model = load_hybrid_model(effective_model_path, arch_json=arch_json if arch_json else None)
            with st.spinner("Evaluating on test split..."):
                metrics = evaluate_hybrid_test(model, test_gen, threshold=threshold)
            st.success("Evaluation completed.")
            st.subheader("Metrics")
            cols = st.columns(3)
            morder = ["precision", "recall", "accuracy", "f1", "iou", "dice"]
            for i, key in enumerate(morder):
                with cols[i % 3]:
                    st.metric(label=key.upper(), value=f"{metrics.get(key, 0.0):.4f}")

            # Bar chart
            st.bar_chart({k.upper(): metrics.get(k, 0.0) for k in morder})
        except Exception as e:
            st.error(f"Evaluation failed: {e}")


def section_visualize(dataset_path: str, threshold: float, model_path: str, arch_json: str):
    st.header("Visualize Prediction (Test Sample)")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(dataset_path)
    except Exception as e:
        st.error(f"Dataset preparation failed: {e}")
        return

    # Filter to indices that have masks
    masked_indices = [i for i, m in enumerate(y_test) if m and str(m) != ""]
    if not masked_indices:
        st.warning("No masked samples found in test set.")
        return

    idx = st.number_input("Test index (with mask)", min_value=0, max_value=len(X_test) - 1, value=masked_indices[0])

    vis_btn = st.button("Predict and Show")
    if vis_btn:
        try:
            # Resolve model path with fallback
            effective_model_path = _resolve_model_path(model_path)
            if not Path(effective_model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = load_hybrid_model(effective_model_path, arch_json=arch_json if arch_json else None)

            # Load image/mask
            import cv2
            img_path = str(X_test[int(idx)])
            mask_path = str(y_test[int(idx)]) if y_test[int(idx)] else ""
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else np.zeros_like(image)
            preprocessor = MRIPreprocessor()
            image_pp, mask_pp = preprocessor.preprocess_pair(image, mask, augment=False)
            x = np.expand_dims(image_pp, axis=0)
            y_prob = model.predict_on_batch(x)[0]
            y_pred = (y_prob >= threshold).astype(np.float32)

            # Show image, ground truth, and overlay
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption("Image")
                st.image(np.squeeze(image_pp), clamp=True)
            with col2:
                st.caption("Ground Truth")
                st.image(np.squeeze(mask_pp), clamp=True)
            with col3:
                st.caption("Overlay (GT=Green, Pred=Red, Overlap=Yellow)")
                st.image(make_overlay(image_pp, mask_pp, y_pred))
        except Exception as e:
            st.error(f"Visualization failed: {e}")


def _read_uploaded_grayscale(uploaded_file):
    import numpy as np
    import cv2
    data = uploaded_file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


def _binary_metrics(pred: np.ndarray, gt: np.ndarray):
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    dice = (2.0 * inter) / (pred_b.sum() + gt_b.sum() + 1e-8)
    iou = inter / (union + 1e-8)
    return float(dice), float(iou)


def section_upload_predict(threshold: float, model_path: str, arch_json: str):
    import numpy as np
    import streamlit as st
    import cv2
    st.header("Upload Image + Ground Truth")
    st.caption("Upload both files together (image first, mask second), or use the separate inputs.")

    files = st.file_uploader(
        "Upload image and mask (2 files)", type=["tif", "png", "jpg", "jpeg"], accept_multiple_files=True
    )
    col_a, col_b = st.columns(2)
    with col_a:
        image_file = st.file_uploader("Image", type=["tif", "png", "jpg", "jpeg"], key="upload_image_single")
    with col_b:
        mask_file = st.file_uploader("Ground Truth Mask", type=["tif", "png", "jpg", "jpeg"], key="upload_mask_single")

    image_up = files[0] if files and len(files) >= 1 else image_file
    mask_up = files[1] if files and len(files) >= 2 else mask_file

    run = st.button("Run Prediction on Uploaded Pair")
    if run:
        effective_model_path = _resolve_model_path(model_path)
        if not Path(effective_model_path).exists():
            st.error(f"Model file not found: {model_path}")
            return

        if not image_up or not mask_up:
            st.warning("Please upload both an image and a ground truth mask.")
            return

        img = _read_uploaded_grayscale(image_up)
        msk = _read_uploaded_grayscale(mask_up)
        if img is None or msk is None:
            st.error("Could not read uploaded files. Ensure they are valid images.")
            return

        preprocessor = MRIPreprocessor(target_size=(128, 128))
        image_pp, mask_pp = preprocessor.preprocess_pair(img, msk, augment=False)
        x = np.expand_dims(np.expand_dims(image_pp, axis=-1), axis=0)

        model = load_hybrid_model(effective_model_path, arch_json=arch_json if arch_json else None)
        y_prob = model.predict_on_batch(x)[0, ...]
        y_pred = (y_prob >= threshold).astype(np.uint8)
        gt_bin = (mask_pp >= 0.5).astype(np.uint8)

        dice, iou = _binary_metrics(y_pred, gt_bin)

        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(image_pp, caption="Preprocessed Image", clamp=True)
        with c2:
            st.image(gt_bin * 255, caption="Ground Truth", clamp=True)
        with c3:
            st.image(y_pred * 255, caption=f"Prediction @ threshold={threshold}", clamp=True)

        st.image(make_overlay(image_pp, gt_bin, y_pred), caption="Overlay")

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Dice", f"{dice:.4f}")
        with m2:
            st.metric("IoU", f"{iou:.4f}")

def main():
    st.title("Hybrid Quantum U-Net: Streamlit Deployment")
    st.caption("Train, evaluate, and visualize segmentation with the Hybrid Quantum U-Net.")

    (
        dataset_path,
        batch_size,
        epochs,
        augment,
        learning_rate,
        threshold,
        model_path,
        arch_json,
    ) = sidebar_controls()

    # Sections
    section_train(dataset_path, batch_size, epochs, augment, learning_rate)
    st.divider()
    section_evaluate(dataset_path, batch_size, threshold, model_path, arch_json)
    st.divider()
    section_visualize(dataset_path, threshold, model_path, arch_json)
    st.divider()
    section_upload_predict(threshold, model_path, arch_json)


if __name__ == "__main__":
    main()