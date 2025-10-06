"""
Minimal Streamlit App: Upload, Preprocess, Predict, Postprocess

Outputs:
- Predicted mask (binary)
- Predicted overlay (on image)
- Dice score vs uploaded ground truth

Requires:
- Trained hybrid model at `models/hybrid_quantum_unet_best.h5` (default)
- `step2_preprocessing.py` for MRIPreprocessor
- Streamlit installed (`pip install streamlit`)
"""

from __future__ import annotations

import os
from pathlib import Path
import csv

import numpy as np
import streamlit as st
import cv2

from step2_preprocessing import MRIPreprocessor
from step7a import load_model as load_hybrid_model


st.set_page_config(
    page_title="Hybrid Quantum U-Net — Quick Predict",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/streamlit/streamlit/issues",
        "About": "Hybrid Quantum U-Net quick predictor. Upload, predict, and download results.",
    },
)


def _best_threshold_from_csv() -> float | None:
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
            pass
    return None


def _default_model_path() -> Path:
    best_h5 = Path("models/hybrid_quantum_unet_best.h5")
    best_weights = Path("models/hybrid_quantum_unet_best.weights.h5")
    final_h5 = Path("models/hybrid_quantum_unet_final.h5")
    if best_h5.exists():
        return best_h5
    if best_weights.exists():
        return best_weights
    return final_h5


def _resolve_model_path(user_path: str) -> str:
    up = Path(user_path) if user_path else Path("")
    if up and up.exists():
        return str(up)
    dp = _default_model_path()
    return str(dp)


def _read_uploaded_grayscale(uploaded_file):
    # Use getvalue() when available to avoid consuming the stream
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
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


def _overlay(image: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    img = np.squeeze(image)
    pr = (np.squeeze(pred_mask) > 0.5).astype(np.uint8)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    rgb = np.stack([img, img, img], axis=-1)
    rgb[pr == 1, 0] = 1.0  # Prediction = Red
    return (rgb * 255).astype(np.uint8)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    x = np.squeeze(image).astype(np.float32)
    minv, maxv = float(np.min(x)), float(np.max(x))
    if maxv - minv < 1e-8:
        x = np.zeros_like(x)
    else:
        x = (x - minv) / (maxv - minv)
    return (x * 255).astype(np.uint8)


def _encode_png(image: np.ndarray) -> bytes:
    """Encode a grayscale or RGB uint8 image to PNG bytes."""
    img = image
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    success, buf = cv2.imencode(".png", img)
    return buf.tobytes() if success else b""


def _postprocess(mask_bin: np.ndarray, open_ksize: int, close_ksize: int) -> np.ndarray:
    m = mask_bin.astype(np.uint8)
    if open_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


@st.cache_resource
def _get_model(path: str, arch_json: str | None):
    return load_hybrid_model(path, arch_json=arch_json if arch_json else None)


def sidebar():
    st.sidebar.header("Settings")
    best_t = _best_threshold_from_csv()
    threshold = st.sidebar.slider(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(best_t) if best_t is not None else 0.5,
        step=0.01,
        help=("Auto-set from sweep: " + f"{best_t:.2f}" if best_t is not None else "Default 0.5; adjust as needed"),
    )
    default_model = _default_model_path()
    model_path = st.sidebar.text_input("Model path", value=str(default_model))
    arch_json_default = Path("models/hybrid_quantum_unet_arch.json")
    arch_json = st.sidebar.text_input(
        "Architecture JSON (optional)",
        value=str(arch_json_default) if arch_json_default.exists() else "",
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Postprocessing")
    open_ksize = st.sidebar.number_input("Opening kernel size", min_value=0, max_value=21, value=0)
    close_ksize = st.sidebar.number_input("Closing kernel size", min_value=0, max_value=21, value=0)
    return threshold, model_path, arch_json, open_ksize, close_ksize


def _find_gt_mask_in_dataset(image_name: str, dataset_root: str) -> np.ndarray | None:
    """Try to locate a ground truth mask for the given image name within the dataset.

    Looks through case folders under `dataset_root` for a file with `image_name`.
    If found, attempts to read a sibling `<stem>_mask.tif`.
    Returns mask array if found, else None.
    """
    try:
        root = Path(dataset_root)
        if not root.exists():
            return None
        # Search immediate subdirectories for the image name
        for case_folder in root.iterdir():
            if not case_folder.is_dir():
                continue
            candidate = case_folder / image_name
            if candidate.exists():
                mask_path = case_folder / (Path(image_name).stem + "_mask.tif")
                if mask_path.exists():
                    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    return m if m is not None else None
        return None
    except Exception:
        return None


def main():
    st.markdown(
        """
        <style>
        /* Polished UI tweaks */
        .app-header { font-size: 28px; font-weight: 700; color: #2b8a3e; margin-bottom: 0.25rem; }
        .app-subheader { color: #415a77; margin-bottom: 1rem; }
        .card { background: #ffffff; border: 1px solid #eaeaea; border-radius: 10px; padding: 1rem; }
        .accent { color: #2b8a3e; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="app-header">Hybrid Quantum U-Net — Quick Predict</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subheader">Single-image upload, prediction and downloads</div>', unsafe_allow_html=True)

    threshold, model_path, arch_json, open_ksize, close_ksize = sidebar()
    effective_model = _resolve_model_path(model_path)
    if not Path(effective_model).exists():
        st.error(f"Model file not found: {effective_model}")
        return

    tabs = st.tabs(["Welcome", "Predict", "App Info"])

    with tabs[0]:
        st.subheader("Welcome")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### Student Details")
            student_name = st.text_input("Student Name", value="Your Name")
            matric_no = st.text_input("Matric No", value="ABC/1234")
            supervisor_name = st.text_input("Supervisor Name", value="Dr. Supervisor")
            st.markdown(
                f"""
                <div class="card">
                    <p><strong>Student:</strong> <span class="accent">{student_name}</span></p>
                    <p><strong>Matric No:</strong> <span class="accent">{matric_no}</span></p>
                    <p><strong>Supervisor:</strong> <span class="accent">{supervisor_name}</span></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("### App Illustration")
            # Try to show an existing figure as illustration
            illustration_candidates = [
                Path("hybrid_quantum_unet_dice.png"),
                Path("combined_dice_all.png"),
                Path("hybrid_quantum_unet_curves_combined.png"),
            ]
            shown = False
            for p in illustration_candidates:
                if p.exists():
                    st.image(str(p), caption="App Illustration", use_column_width=True)
                    shown = True
                    break
            if not shown:
                st.info("Upload and predict to see the app in action.")

    with tabs[1]:
        st.subheader("Upload Image")
        image_file = st.file_uploader("Image", type=["tif", "png", "jpg", "jpeg"], key="img_single")
        # Limit input size (e.g., 8 MB) to avoid overly large files
        max_upload_mb = 8
        if image_file and getattr(image_file, "size", 0) > max_upload_mb * 1024 * 1024:
            st.warning(f"File too large (> {max_upload_mb} MB). Please upload a smaller image.")
            image_file = None
        if image_file:
            preview_img = _read_uploaded_grayscale(image_file)
            if preview_img is not None:
                st.image(preview_img, caption=f"Uploaded Image Preview — {image_file.name}", clamp=True, width=256)
            else:
                st.warning("Unable to preview uploaded image; it may be invalid.")
        st.markdown("---")
        st.subheader("Optional: Auto-fetch Ground Truth")
        use_auto_gt = st.checkbox("Try to find ground truth in dataset (by filename)", value=True)
        dataset_root = st.text_input("Dataset root (for auto GT)", value=str(Path("kaggle_3m")))

        run = st.button("Run Prediction")
        if run:
            if not image_file:
                st.warning("Please upload an image.")
                return

            # Read image
            img = _read_uploaded_grayscale(image_file)
            if img is None:
                st.error("Could not read the uploaded image. Ensure it is a valid file.")
                return

            # Optionally resolve ground truth from dataset
            msk = None
            if use_auto_gt and image_file.name:
                msk = _find_gt_mask_in_dataset(image_file.name, dataset_root)

            # Preprocess using real GT if found; else use zero-mask placeholder
            pre = MRIPreprocessor(target_size=(128, 128))
            if msk is not None:
                image_pp, mask_pp = pre.preprocess_pair(img, msk, augment=False)
                have_gt = True
            else:
                zero_msk = np.zeros_like(img, dtype=np.uint8)
                image_pp, mask_pp = pre.preprocess_pair(img, zero_msk, augment=False)
                have_gt = False

            # Predict
            x = np.expand_dims(np.expand_dims(image_pp, axis=-1), axis=0)
            model = _get_model(effective_model, arch_json if arch_json else None)
            y_prob = model.predict_on_batch(x)[0, ...]
            y_bin = (y_prob >= threshold).astype(np.uint8)

            # Postprocess
            y_pp = _postprocess(y_bin, open_ksize=open_ksize, close_ksize=close_ksize)

            # Metrics (only if GT available)
            dice = None
            if have_gt:
                gt_bin = (mask_pp >= 0.5).astype(np.uint8)
                dice, _ = _binary_metrics(y_pp, gt_bin)

            # Outputs (ordered: original image -> predicted mask -> overlay), equal display sizes
            st.subheader("Outputs")
            col1, col2, col3 = st.columns(3)
            overlay = _overlay(image_pp, y_pp)
            display_width = 256
            with col1:
                st.image(_to_uint8(image_pp), caption="Original Image (preprocessed)", clamp=True, width=display_width)
                st.download_button(
                    label="Download Original (PNG)",
                    data=_encode_png(_to_uint8(image_pp)),
                    file_name="input_image.png",
                    mime="image/png",
                )
            with col2:
                st.image(y_pp * 255, caption=f"Predicted Mask (threshold={threshold:.2f})", clamp=True, width=display_width)
                st.download_button(
                    label="Download Predicted Mask (PNG)",
                    data=_encode_png(y_pp * 255),
                    file_name="pred_mask.png",
                    mime="image/png",
                )
            with col3:
                st.image(overlay, caption="Predicted Overlay", clamp=True, width=display_width)
                st.download_button(
                    label="Download Overlay (PNG)",
                    data=_encode_png(overlay),
                    file_name="pred_overlay.png",
                    mime="image/png",
                )

            # Dice score
            if dice is not None:
                st.metric("Dice", f"{dice:.4f}")
            else:
                st.info("Dice unavailable: no matching ground truth found in dataset.")

    with tabs[2]:
        st.subheader("App Info")
        st.markdown(
            """
            - Model: Hybrid Quantum U-Net trained on LGG MRI segmentation.
            - Preprocessing: Resize to 128x128, z-score normalization; masks binarized.
            - Postprocessing: Optional morphological opening/closing via sidebar settings.
            - Threshold: Adjustable via sidebar; used to binarize probabilities.
            - Auto GT: Matches `<stem>_mask.tif` under the selected dataset root.
            - Outputs: Original (preprocessed), Predicted mask, Overlay; all downloadable as PNG.
            - Metrics: Dice computed when GT found; IoU omitted on single-input flow for simplicity.
            - UI/UX: Light theme, polished visuals, consistent image display sizes.
            """
        )


if __name__ == "__main__":
    main()