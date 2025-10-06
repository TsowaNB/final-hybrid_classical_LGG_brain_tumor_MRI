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
from typing import List

import numpy as np
import streamlit as st
import tensorflow as tf

from step2_preprocessing import prepare_dataset, MRIPreprocessor, MRIDataGenerator
from step3_models import create_hybrid_quantum_unet, QuantumLayer
from step4a import train as hybrid_train, CombinedBCEDiceLoss, dice_coefficient
from step7a import load_model as load_hybrid_model, evaluate_test as evaluate_hybrid_test


st.set_page_config(page_title="Hybrid Quantum U-Net", layout="wide")


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
    batch_size = st.sidebar.number_input("Batch size", min_value=1, max_value=64, value=16)
    epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=200, value=10)
    augment = st.sidebar.checkbox("Augment", value=True)
    learning_rate = st.sidebar.number_input("Learning rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%f")
    threshold = st.sidebar.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    st.sidebar.markdown("---")
    model_path = st.sidebar.text_input(
        "Hybrid model path",
        value=str(Path("models/hybrid_quantum_unet_final.h5")),
        help="Full .h5 or weights-only .weights.h5",
    )
    arch_json = st.sidebar.text_input(
        "Architecture JSON (optional)",
        value=str(Path("models/hybrid_quantum_unet_arch.json")),
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
            with st.spinner("Preparing dataset and loading model..."):
                X_test, y_test, val_gen, test_gen = load_test_generators(dataset_path, batch_size=batch_size)
                model = load_hybrid_model(model_path, arch_json=arch_json if arch_json else None)
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
            model = load_hybrid_model(model_path, arch_json=arch_json if arch_json else None)

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


if __name__ == "__main__":
    main()