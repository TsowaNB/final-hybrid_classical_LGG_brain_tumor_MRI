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
import io
import base64
import zipfile

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


def _load_threshold_sweep() -> list[tuple[float, float]]:
    """Load threshold sweep (threshold, dice) pairs sorted by threshold."""
    path = Path("threshold_sweep_hybrid.csv")
    data: list[tuple[float, float]] = []
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t_raw, d_raw = row.get("threshold"), row.get("dice")
                    if t_raw is None or d_raw is None:
                        continue
                    try:
                        t = float(t_raw)
                        d = float(d_raw)
                    except Exception:
                        continue
                    data.append((t, d))
    except Exception:
        pass
    return sorted(data, key=lambda x: x[0])


def _snap_threshold(value: float, sweep: list[tuple[float, float]]) -> float:
    """Snap to the closest threshold available in sweep, fallback to value."""
    if not sweep:
        return value
    arr = np.array([t for t, _ in sweep], dtype=np.float32)
    idx = int(np.argmin(np.abs(arr - value)))
    return float(arr[idx])


def _dice_for_threshold(value: float, sweep: list[tuple[float, float]]) -> float | None:
    """Return dice for exact threshold match in sweep, else None."""
    for t, d in sweep:
        if abs(t - value) < 1e-9:
            return d
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

def _encode_zip(files: dict[str, bytes]) -> bytes:
    """Create a ZIP archive from a mapping of filename -> bytes."""
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return bio.getvalue()

def _data_uri(mime: str, data: bytes) -> str:
    """Return a data: URI for inline downloads in HTML."""
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


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
        value=0.85,
        step=0.01,
        help=("Default 0.85; adjust as needed. You can optionally use the sweep-recommended threshold via the checkbox below."),
    )
    use_recommended = st.sidebar.checkbox(
        "Use recommended threshold from sweep",
        value=False,
        help="If enabled, uses best threshold from threshold_sweep_hybrid.csv",
    )
    default_model = _default_model_path()
    model_path = st.sidebar.text_input("Model path", value=str(default_model))
    arch_json_default = Path("models/hybrid_quantum_unet_arch.json")
    arch_json = st.sidebar.text_input(
        "Architecture JSON (optional)",
        value=str(arch_json_default) if arch_json_default.exists() else "",
        help=(
            "Supply a Keras architecture JSON (from model.to_json) when using a weights-only file (.weights.h5). "
            "Leave blank if you load a full .h5 model file with embedded architecture."
        ),
    )
    st.sidebar.caption(
        "Use this only if your model path points to weights-only (no architecture). Otherwise, leave it empty."
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Postprocessing")
    open_ksize = st.sidebar.number_input("Opening kernel size", min_value=0, max_value=21, value=0)
    close_ksize = st.sidebar.number_input("Closing kernel size", min_value=0, max_value=21, value=0)
    return threshold, model_path, arch_json, open_ksize, close_ksize, use_recommended


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
    # Session state flags for skeletons and output lifecycle
    st.session_state.setdefault("is_running", False)
    st.session_state.setdefault("outputs_ready", False)
    st.session_state.setdefault("last_image_name", None)
    st.session_state.setdefault("last_threshold_used", None)
    st.session_state.setdefault("last_use_recommended", None)
    st.session_state.setdefault("last_effective_model", None)

    st.markdown(
        """
        <style>
        :root {
            --accent: #2563eb; /* blue */
            --accent2: #7c3aed; /* purple */
            --ink: #0f172a;
            --muted: #64748b;
            --bg: #f8fafc;
        }
        .stApp { background: linear-gradient(180deg, var(--bg), #ffffff); }
        .app-header { font-size: 30px; font-weight: 700; color: var(--accent); margin-bottom: 0.25rem; }
        .app-subheader { color: var(--muted); margin-bottom: 1rem; }
        .card { background: #ffffff; border: 1px solid #e6edf5; border-radius: 14px; padding: 1rem; box-shadow: 0 2px 12px rgba(37,99,235,0.08); }
        .accent { color: var(--accent); }
        /* Hero */
        .hero { background: linear-gradient(90deg, var(--accent), var(--accent2)); color: #ffffff; border-radius: 16px; padding: 16px 20px; box-shadow: 0 6px 18px rgba(37,99,235,0.25); margin-bottom: 18px; }
        .hero h1 { margin: 0; font-size: 28px; font-weight: 800; letter-spacing: 0.2px; }
        .hero p { margin: 6px 0 10px; font-size: 15px; opacity: 0.95; }
        .badge { display: inline-block; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.25); color: #fff; padding: 4px 10px; border-radius: 999px; font-size: 12px; margin-right: 8px; }
        /* Tabs styling (button-like, with icons) */
        .stTabs [data-baseweb="tab"] {
            height: 44px; border-radius: 12px; background: #f1f5f9; color: var(--ink);
            border: 1px solid #e0e6ef; margin-right: 8px; padding: 0 16px;
            font-weight: 600; transition: all 0.15s ease-in-out;
        }
        .stTabs [data-baseweb="tab"]:hover { filter: brightness(1.02); }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, var(--accent), var(--accent2)); color: #ffffff;
            border-color: transparent; box-shadow: 0 4px 12px rgba(37,99,235,0.20);
        }
        /* Button styling */
        .stButton>button { background: linear-gradient(90deg, var(--accent), var(--accent2)); color: white; border: none; border-radius: 12px; padding: 0.6rem 1rem; font-weight: 600; }
        .stButton>button:hover { filter: brightness(1.05); }
        /* Inputs and sliders */
        .stTextInput>div>div>input,
        .stNumberInput input,
        .stTextArea textarea,
        .stSelectbox>div>div>select {
            border-radius: 10px !important;
            border: 1px solid #e0e6ef !important;
        }
        .stTextInput>div>div>input:focus,
        .stNumberInput input:focus,
        .stTextArea textarea:focus,
        .stSelectbox>div>div>select:focus {
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.25) !important;
            border-color: var(--accent) !important;
        }
        /* Slider accent */
        .stSlider [data-baseweb="slider"]>div>div>div {
            background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
        }
        /* Skeleton */
        .skeleton { position: relative; overflow: hidden; background: #e5e7eb; border-radius: 12px; }
        .skeleton::after { content: ""; position: absolute; inset: 0; transform: translateX(-100%);
            background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.6), rgba(255,255,255,0));
            animation: shimmer 1.2s infinite; }
        @keyframes shimmer { 100% { transform: translateX(100%); } }
        /* Responsive tweaks */
        @media (max-width: 900px) {
            .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
            .stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; }
            .stTabs [data-baseweb="tab"] { margin-bottom: 8px; }
        }
        /* Images */
        .stImage img { border-radius: 12px; border: 1px solid #e6edf5; }
        /* Captions */
        .stCaption { color: var(--muted) !important; }
        /* Downloads toolbar */
        .dl-toolbar { display:flex; gap:10px; flex-wrap:wrap; margin: 6px 0 16px; }
        .pill-btn { display:inline-block; padding:10px 14px; border-radius:999px; font-weight:600; text-decoration:none; }
        .pill-btn.primary { color:#fff; background: linear-gradient(90deg, var(--accent), var(--accent2)); box-shadow: 0 6px 14px rgba(59,130,246,0.28); }
        .pill-btn { color:#0f172a; background:#f1f5f9; border:1px solid #e2e8f0; }
        .pill-btn:hover { filter: brightness(1.06); transform: translateY(-1px); transition: all 0.15s ease; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Remove textual headers; keep only the top banner
    # Global hero banner at the top
    st.markdown(
        """
        <div class="hero">
            <h1>MRI segmentation with Hybrid U‑Net</h1>
            <p>Brain MRI lesion segmentation — upload, predict, and download clean results.</p>
            <span class="badge">Fast Inference</span>
            <span class="badge">Robust Segmentation</span>
            <span class="badge">Simple Downloads</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    threshold_sel, model_path, arch_json, open_ksize, close_ksize, use_recommended = sidebar()
    effective_model = _resolve_model_path(model_path)
    if not Path(effective_model).exists():
        st.error(f"Model file not found: {effective_model}")
        return

    # Load sweep and determine effective threshold
    sweep = _load_threshold_sweep()
    if use_recommended:
        t_best = _best_threshold_from_csv()
        threshold_effective = float(t_best) if t_best is not None else float(threshold_sel)
    else:
        threshold_effective = float(threshold_sel)
    expected_dice = _dice_for_threshold(threshold_effective, sweep)
    # Invalidate outputs if controls changed since last run
    if st.session_state.get("outputs_ready", False):
        if (st.session_state.get("last_threshold_used") is not None and abs(float(threshold_effective) - float(st.session_state.get("last_threshold_used"))) > 1e-9) \
            or (bool(use_recommended) != bool(st.session_state.get("last_use_recommended"))) \
            or (str(effective_model) != str(st.session_state.get("last_effective_model"))):
            st.session_state["outputs_ready"] = False

    tabs = st.tabs(["🏠 Welcome", "🔬 Predict", "ℹ️ App Info"])

    with tabs[0]:
        st.subheader("Welcome")
        st.markdown(
            """
            <div class="card">
              <div style="display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap;">
                <div style="flex:1; min-width:280px;">
                  <p><strong>Student:</strong> <span class="accent">Tsowa Blessing Nnawonchiko</span></p>
                  <p><strong>Matric No:</strong> <span class="accent">2019/1/76577PP</span></p>
                  <p><strong>Supervisor:</strong> <span class="accent">Dr. O. M. Dada</span></p>
                  <p>Clean, responsive interface for brain MRI lesion segmentation using a Hybrid Quantum U‑Net.</p>
                </div>
                <div style="flex:1; min-width:280px;">
                  <p><strong>Quick Start</strong></p>
                  <ol style="margin:0; padding-left:18px;">
                    <li>Open the Predict tab.</li>
                    <li>Upload a single MRI slice.</li>
                    <li>Optionally enable auto ground truth.</li>
                    <li>Adjust threshold or use recommended.</li>
                    <li>Run Prediction to view and download outputs.</li>
                  </ol>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tabs[1]:
        st.subheader("Upload Image")
        # st.markdown('<div class="card">', unsafe_allow_html=True)
        image_file = st.file_uploader("Image", type=["tif", "png", "jpg", "jpeg"], key="img_single")
        # Limit input size (e.g., 8 MB) to avoid overly large files
        max_upload_mb = 8
        if image_file and getattr(image_file, "size", 0) > max_upload_mb * 1024 * 1024:
            st.warning(f"File too large (> {max_upload_mb} MB). Please upload a smaller image.")
            image_file = None
        preview_placeholder = st.empty()
        if image_file:
            # Show skeleton only when an image has been selected and before preview renders
            preview_placeholder.markdown('<div class="skeleton" style="height: 180px;"></div>', unsafe_allow_html=True)
            preview_img = _read_uploaded_grayscale(image_file)
            if preview_img is not None:
                preview_placeholder.image(preview_img, caption=f"Uploaded Image Preview — {image_file.name}", clamp=True, width=256)
            else:
                preview_placeholder.empty()
                st.warning("Unable to preview uploaded image; it may be invalid.")
            # Invalidate outputs if a new image is selected
            if image_file.name != st.session_state.get("last_image_name"):
                st.session_state["outputs_ready"] = False
                st.session_state["last_image_name"] = image_file.name
        else:
            st.caption("Upload an image to preview.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("Optional: Auto-fetch Ground Truth")
        # st.markdown('<div class="card">', unsafe_allow_html=True)
        use_auto_gt = st.checkbox("Try to find ground truth in dataset (by filename)", value=True)
        dataset_root = st.text_input(
            "Dataset root (for auto GT)", value=str(Path("kaggle_3m")), disabled=True
        )

        # Show threshold info from sweep
        th_info = f"Using threshold: {threshold_effective:.2f}"
        if expected_dice is not None:
            th_info += f" — sweep Dice: {expected_dice:.4f}"
        st.caption(th_info)
        st.markdown('</div>', unsafe_allow_html=True)

        run = st.button("Run Prediction", disabled=st.session_state.get("is_running", False), key="run_predict")
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

            # Mark running and show spinner during compute
            st.session_state["is_running"] = True
            with st.spinner("Running prediction..."):
                # Predict
                x = np.expand_dims(np.expand_dims(image_pp, axis=-1), axis=0)
                model = _get_model(effective_model, arch_json if arch_json else None)
                y_prob = model.predict_on_batch(x)[0, ...]
                y_bin = (y_prob >= threshold_effective).astype(np.uint8)

                # Postprocess
                y_pp = _postprocess(y_bin, open_ksize=open_ksize, close_ksize=close_ksize)

                # Metrics (only if GT available)
                dice = None
                if have_gt:
                    gt_bin = (mask_pp >= 0.5).astype(np.uint8)
                    dice, _ = _binary_metrics(y_pp, gt_bin)

                # Prepare overlay and store outputs in session
                overlay = _overlay(image_pp, y_pp)
                st.session_state["out_image_pp"] = _to_uint8(image_pp)
                st.session_state["out_mask"] = (y_pp * 255).astype(np.uint8)
                st.session_state["out_overlay"] = overlay
                st.session_state["out_dice"] = dice
                st.session_state["out_threshold_used"] = float(threshold_effective)
                st.session_state["last_threshold_used"] = float(threshold_effective)
                st.session_state["last_use_recommended"] = bool(use_recommended)
                st.session_state["last_effective_model"] = str(effective_model)
                st.session_state["last_image_name"] = image_file.name
                st.session_state["outputs_ready"] = True
            st.session_state["is_running"] = False

        # Unified Outputs rendering based on state (render only after predictions)
        if st.session_state.get("outputs_ready", False):
            st.subheader("Results")

            # Dice first
            dice_val = st.session_state.get("out_dice")
            if dice_val is not None:
                st.metric("Dice", f"{float(dice_val):.4f}")
            else:
                st.info("Dice unavailable: no matching ground truth found in dataset.")

            # Cool download toolbar (ZIP + individual PNGs)
            img_png = _encode_png(st.session_state.get("out_image_pp"))
            mask_png = _encode_png(st.session_state.get("out_mask"))
            overlay_png = _encode_png(st.session_state.get("out_overlay"))
            zip_bytes = _encode_zip({
                "input_image.png": img_png,
                "pred_mask.png": mask_png,
                "pred_overlay.png": overlay_png,
                "metrics.txt": (f"Dice={float(dice_val):.4f}".encode("utf-8") if dice_val is not None else b"Dice=NA"),
            })
            img_uri = _data_uri("image/png", img_png)
            mask_uri = _data_uri("image/png", mask_png)
            overlay_uri = _data_uri("image/png", overlay_png)
            zip_uri = _data_uri("application/zip", zip_bytes)

            st.markdown(f"""
            <div class=\"dl-toolbar\">
              <a class=\"pill-btn primary\" href=\"{zip_uri}\" download=\"results.zip\">📦 Download All</a>
              <a class=\"pill-btn\" href=\"{mask_uri}\" download=\"pred_mask.png\">🟣 Mask PNG</a>
              <a class=\"pill-btn\" href=\"{overlay_uri}\" download=\"pred_overlay.png\">🔴 Overlay PNG</a>
              <a class=\"pill-btn\" href=\"{img_uri}\" download=\"input_image.png\">🖼️ Original PNG</a>
            </div>
            """, unsafe_allow_html=True)

            # Images grid
            col1, col2, col3 = st.columns(3)
            display_width = 256
            with col1:
                st.image(st.session_state.get("out_image_pp"), caption="Original Image (preprocessed)", clamp=True, width=display_width)
            with col2:
                st.image(
                    st.session_state.get("out_mask"),
                    caption=f"Predicted Mask (threshold={st.session_state.get('out_threshold_used', threshold_effective):.2f})",
                    clamp=True,
                    width=display_width,
                )
            with col3:
                st.image(st.session_state.get("out_overlay"), caption="Predicted Overlay", clamp=True, width=display_width)

    with tabs[2]:
        st.subheader("App Info")

        # Topic paragraph and fixed About image first
        # st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            """
            **Hybrid Classical–Quantum U‑Net for LGG MRI Segmentation** — This approach combines a U‑Net’s classical convolutional encoder‑decoder with a compact variational quantum layer at the bottleneck to capture expressive, non‑classical feature interactions. Applied to lower‑grade glioma MRI, it targets improved boundary precision and robustness. The figure below shows multi‑modal inputs (a–c) and the resulting lesion mask (d).
            """
        )
        about_path = Path("illustration.png")
        if about_path.exists():
            st.image(
                str(about_path),
                caption="Example MRI modalities (a–c) and LGG mask (d)",
                use_container_width=True,
            )
        else:
            st.error("illustration.png not found in project root. Please add the attached image as 'illustration.png'.")
        st.markdown('</div>', unsafe_allow_html=True)

        # How the app works (existing details)
        st.markdown(
            """
            ### How the App Works
            1) Upload: Load a single MRI slice (`.tif/.png/.jpg`) via Predict tab.
            2) Preprocessing: Image resizes to 128×128 and is z-score normalized.
               If Ground Truth is found (optional auto-fetch), it is binarized and aligned.
            3) Model Loading: The Hybrid Quantum U-Net loads from the model path. If using a weights-only file,
               you can provide an Architecture JSON to reconstruct the model graph.
            4) Prediction: The model outputs per-pixel probabilities; these are binarized using the chosen threshold.
               You can set a fixed threshold or enable the sweep-recommended value.
            5) Postprocessing: Optional morphological opening/closing clean up small artifacts and fill small holes.
            6) Outputs: You can view and download the preprocessed image, predicted mask, and overlay.
            7) Metrics: If Ground Truth is found, Dice is computed on the postprocessed mask.

            ### Key Settings
            - Threshold: Slider in the sidebar controls mask binarization.
              Toggle "Use recommended threshold from sweep" to use the best Dice threshold from `threshold_sweep_hybrid.csv`.
            - Postprocessing: Opening removes small noise; Closing fills small holes (set to 0 for none).
            - Model Path: Select a full `.h5` model or `.weights.h5` + Architecture JSON.

            ### Notes
            - Ground truth auto-fetch looks for a sibling file named `<stem>_mask.tif` in your dataset folders.
            - All outputs render at equal sizes and can be downloaded as PNGs.
            - The UI uses a blue→purple accent gradient for a modern, professional look.
            """
        )


if __name__ == "__main__":
    main()