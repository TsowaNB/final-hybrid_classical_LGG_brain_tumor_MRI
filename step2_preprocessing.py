"""
Step 2: Data Preprocessing
Hybrid U-Net Model for Lower-grade Glioma Segmentation in MRI
"""

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import albumentations as A

class MRIPreprocessor:
    """Efficient MRI preprocessing with normalization and augmentation."""
    
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
        ])
    
    def normalize_image(self, image):
        """Z-score normalization."""
        return (image - np.mean(image)) / (np.std(image) + 1e-8)
    
    def preprocess_pair(self, image, mask, augment=False):
        """Preprocess image-mask pair."""
        # Resize
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize
        image = self.normalize_image(image.astype(np.float32))
        mask = (mask > 0).astype(np.float32)
        
        # Augment if requested
        if augment:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        return np.expand_dims(image, -1), np.expand_dims(mask, -1)

class MRIDataGenerator(tf.keras.utils.Sequence):
    """Efficient data generator for training."""
    
    def __init__(self, image_paths, mask_paths, batch_size=16, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.augment = augment
        self.preprocessor = MRIPreprocessor()
        self.indices = np.arange(len(image_paths))
        np.random.shuffle(self.indices)
    
    def __len__(self):
        return len(self.image_paths) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images, batch_masks = [], []
        
        for i in batch_indices:
            # Load real images
            image = cv2.imread(str(self.image_paths[i]), cv2.IMREAD_GRAYSCALE)
            mask_path = str(self.mask_paths[i]) if self.mask_paths[i] else None
            if mask_path and mask_path != "":
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.zeros_like(image) if image is not None else np.zeros((128, 128), dtype=np.uint8)
            
            # Handle case where image loading fails
            if image is None:
                raise ValueError(f" Failed to load image: {self.image_paths[i]}")
            if mask is None:
                mask = np.zeros_like(image)
            
            image, mask = self.preprocessor.preprocess_pair(image, mask, self.augment)
            batch_images.append(image)
            batch_masks.append(mask)
        
        return np.array(batch_images), np.array(batch_masks)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def prepare_dataset(dataset_path=None, test_size=0.15, val_size=0.15):
    """Prepare dataset splits (70/15/15)."""
    if dataset_path is None:
        dataset_path = os.environ.get("LGG_DATASET_PATH", "/kaggle/input/lgg-mri-segmentation/kaggle_3m")
    dataset_path = Path(dataset_path)
    image_paths, mask_paths = [], []
    
    # Check if dataset path exists
    if not dataset_path.exists():
        raise FileNotFoundError(f" Dataset path not found: {dataset_path}")
    
    for case_folder in dataset_path.iterdir():
        if not case_folder.is_dir():
            continue
        
        for file_path in case_folder.glob("*.tif"):
            if "_mask" not in file_path.name:
                image_paths.append(str(file_path))
                mask_path = case_folder / (file_path.stem + "_mask.tif")
                mask_paths.append(str(mask_path) if mask_path.exists() else "")
    
    if not image_paths:
        raise ValueError(f" No image files found in dataset path: {dataset_path}")
    
    # Split dataset
    X_temp, X_test, y_temp, y_test = train_test_split(image_paths, mask_paths, test_size=test_size, random_state=42)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)
    
    # Return as tuple for backward compatibility
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Main preprocessing demonstration."""
    DATASET_PATH = os.environ.get("LGG_DATASET_PATH", "/kaggle/input/lgg-mri-segmentation/kaggle_3m")
    
    print(" Preparing dataset splits...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(DATASET_PATH)
        
        print(f" Dataset prepared:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
    except (FileNotFoundError, ValueError) as e:
        print(f" Error: {e}")
        print(" Please ensure the dataset path is correct and contains .tif files")
        print(" You can download the LGG MRI dataset and place it in './dataset' folder")

if __name__ == "__main__":
    main()