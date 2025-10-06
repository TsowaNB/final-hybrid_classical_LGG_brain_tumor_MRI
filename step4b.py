"""
Step 4B: Training Script (Classical U-Net)
Train the Classical U-Net for LGG MRI segmentation.

Usage:
  - Ensure the dataset path is set via the env var `LGG_DATASET_PATH`,
    or pass `--dataset-path` when running this script.
  - Example:
      python step4b.py --epochs 50 --batch-size 16 --lr 1e-3
"""

import os
from pathlib import Path
import argparse
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.metrics import BinaryIoU

from step2_preprocessing import MRIDataGenerator, prepare_dataset
from step3_models import create_classical_unet
from step4a import CombinedBCEDiceLoss, dice_coefficient
from step6_graph_train import plot_training_curves

"""Clean, deployment-ready training script for Classical U-Net."""

# Base dirs (auto-detect Kaggle)
KAGGLE_WORKING = Path('/kaggle/working')
BASE_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path('.')
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'
 


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)


def create_callbacks(model_name: str, patience: int = 10):
    """Create training callbacks (ModelCheckpoint/EarlyStopping/LR scheduler/CSVLogger)."""
    ensure_dirs()
    return [
        # Checkpoint: save best full model in HDF5 (.h5)
        callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}_best.h5"),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # Checkpoint: save best weights in HDF5 (.weights.h5)
        callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}_best.weights.h5"),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.CSVLogger(str(LOGS_DIR / f"{model_name}_train_log.csv"), append=False),
    ]


def setup_gpu_memory_growth():
    """Enable GPU memory growth to avoid OOM errors."""
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU config warning: {e}")


def export_for_deployment(model, model_name: str):
    """Export final artifacts: full H5 model (architecture + weights) and architecture JSON."""
    ensure_dirs()
    # Full H5 model (includes architecture + weights)
    h5_full_path = MODELS_DIR / f"{model_name}_final.h5"
    try:
        model.save(str(h5_full_path))
        print(f" Saved final full H5 model to: {h5_full_path}")
    except Exception as e:
        print(f" Warning: full H5 save failed: {e}")
    # Optional: Save architecture JSON for reproducible reconstruction
    try:
        arch_json = model.to_json()
        arch_path = MODELS_DIR / f"{model_name}_arch.json"
        with open(arch_path, "w", encoding="utf-8") as f:
            f.write(arch_json)
        print(f" Saved model architecture JSON to: {arch_path}")
    except Exception as e:
        print(f" Warning: architecture JSON save failed: {e}")


def compile_model(model, learning_rate: float = 1e-3):
    """Compile model with optimizer, loss, and metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=CombinedBCEDiceLoss(),
        metrics=[dice_coefficient, BinaryIoU()],
    )
    return model


def train_model(model, train_gen, val_gen, model_name: str, epochs: int = 50, verbose: int = 1):
    """Train model with generators and callbacks."""
    print(f" Training {model_name}...")
    callbacks_list = create_callbacks(model_name)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=verbose,
    )
    print(f" {model_name} training completed!")
    return history


def train(
    dataset_path: str | None,
    batch_size: int = 16,
    epochs: int = 50,
    augment: bool = True,
    learning_rate: float = 1e-3,
    verbose: int = 1,
):
    """Simple end-to-end training pipeline, deployment-ready."""
    ensure_dirs()
    setup_gpu_memory_growth()

    # Prepare paths
    if dataset_path is None:
        dataset_path = os.environ.get(
            "LGG_DATASET_PATH",
            "/kaggle/input/lgg-mri-segmentation/kaggle_3m",
        )

    # Load and prepare data
    print(" Preparing dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(dataset_path)
    print(f" Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Generators
    train_gen = MRIDataGenerator(X_train, y_train, batch_size=batch_size, augment=augment)
    val_gen = MRIDataGenerator(X_val, y_val, batch_size=batch_size, augment=False)

    # Model
    model_name = "classical_unet"
    model = create_classical_unet()
    model = compile_model(model, learning_rate=learning_rate)

    # Train
    history = train_model(model, train_gen, val_gen, model_name=model_name, epochs=epochs, verbose=verbose)

    # Export final artifacts for deployment
    export_for_deployment(model, model_name=model_name)
    # Plot training curves inline (Dice, IoU, Loss)
    try:
        log_csv_path = LOGS_DIR / f"{model_name}_train_log.csv"
        out_dir = BASE_DIR / 'artifacts' / 'graphs'
        saved = plot_training_curves(log_csv=str(log_csv_path), out_dir=out_dir, tag=model_name)
        print(" Saved training graphs:")
        for p in saved:
            print(f"  - {p}")
    except Exception as e:
        print(f" Warning: plotting training curves failed: {e}")
    return history


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Classical U-Net (simple pipeline)")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to LGG dataset (defaults to env LGG_DATASET_PATH)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (P100-friendly)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], help="Training verbosity (0, 1, or 2)")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    # TFLite export removed for H5-only workflow
    # In Jupyter/Colab, the kernel injects extra args (e.g., -f <connection.json>).
    # Use parse_known_args to ignore unknown args gracefully.
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f" Warning: ignoring unknown CLI args: {unknown}")
    return args


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        augment=(not args.no_augment),
        learning_rate=args.lr,
        verbose=args.verbose,
    )