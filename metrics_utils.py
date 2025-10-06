"""
Binary classification metrics utilities.

Provides functions to compute accuracy, precision, recall, specificity,
F1-score, ROC-AUC, PR-AUC, and confusion matrix with minimal dependencies.
Falls back gracefully if sklearn is not available.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tn, fp, fn, tp) counts for binary labels and predictions."""
    y_true = (y_true.astype(int) > 0).astype(int)
    y_pred = (y_pred.astype(int) > 0).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)


def binary_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute common binary classification metrics.

    - y_true: binary labels {0,1}
    - y_score: predicted probabilities [0,1]
    - threshold: decision threshold for converting scores to labels
    Returns dict with accuracy, precision, recall, specificity, f1, auc_roc, auc_pr, confusion_matrix
    """
    y_true = (y_true.astype(int) > 0).astype(int)
    y_score = y_score.astype(float)
    y_pred = (y_score >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    acc = safe_div(tn + tp, tn + fp + fn + tp)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    f1 = safe_div(2 * prec * rec, prec + rec)

    # ROC-AUC
    if roc_auc_score is not None:
        try:
            auc_roc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc_roc = float('nan')
    else:
        # Simple ROC-AUC approximation via threshold sweep
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        tpr = [0.0]
        fpr = [0.0]
        pos = float(np.sum(y_true))
        neg = float(len(y_true) - np.sum(y_true))
        tp_cum = 0.0
        fp_cum = 0.0
        for yt in y_true_sorted:
            if yt == 1:
                tp_cum += 1.0
            else:
                fp_cum += 1.0
            tpr.append(tp_cum / (pos + 1e-12))
            fpr.append(fp_cum / (neg + 1e-12))
        # Trapezoidal integration
        auc_roc = 0.0
        for i in range(1, len(tpr)):
            auc_roc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
        auc_roc = float(auc_roc)

    # PR-AUC (Average Precision)
    if average_precision_score is not None:
        try:
            auc_pr = float(average_precision_score(y_true, y_score))
        except Exception:
            auc_pr = float('nan')
    else:
        # Basic PR-AUC approximation via precision-recall points
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        tp, fp = 0.0, 0.0
        precisions, recalls = [], []
        pos = float(np.sum(y_true))
        for yt in y_true_sorted:
            if yt == 1:
                tp += 1.0
            else:
                fp += 1.0
            precisions.append(tp / (tp + fp + 1e-12))
            recalls.append(tp / (pos + 1e-12))
        # Integrate PR curve with step-wise approximation
        auc_pr = 0.0
        for i in range(1, len(recalls)):
            auc_pr += (recalls[i] - recalls[i - 1]) * precisions[i]
        auc_pr = float(auc_pr)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "threshold": float(threshold),
    }


def labels_from_masks(masks: np.ndarray) -> np.ndarray:
    """Convert segmentation masks to per-image presence labels (any pixel > 0 => 1)."""
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    return (np.any(masks > 0, axis=(1, 2))).astype(int)


if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description="Binary classification metrics demo")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for labels")
    parser.add_argument("--y-true", type=str, default="", help="Path to .npy array of true labels (0/1)")
    parser.add_argument("--y-score", type=str, default="", help="Path to .npy array of predicted scores [0,1]")
    args = parser.parse_args()

    if args.y_true and os.path.isfile(args.y_true) and args.y_score and os.path.isfile(args.y_score):
        y_true = np.load(args.y_true)
        y_score = np.load(args.y_score)
        print(f"Loaded y_true from {args.y_true} and y_score from {args.y_score}")
    else:
        # Fallback demo data
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_score = np.array([0.1, 0.9, 0.65, 0.3, 0.8, 0.2, 0.55, 0.7])
        print("Using built-in demo labels and scores.")

    m = binary_classification_metrics(y_true, y_score, threshold=args.threshold)
    print("Classification metrics:")
    print(f" - accuracy:   {m['accuracy']:.4f}")
    print(f" - precision:  {m['precision']:.4f}")
    print(f" - recall:     {m['recall']:.4f}")
    print(f" - specificity:{m['specificity']:.4f}")
    print(f" - f1:         {m['f1']:.4f}")
    print(f" - auc_roc:    {m['auc_roc']:.4f}")
    print(f" - auc_pr:     {m['auc_pr']:.4f}")
    cm = m["confusion_matrix"]
    print(f" - confusion_matrix: tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}")