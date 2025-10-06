""" 
Step 1: Data Loading and Visualization 
""" 

import os 
import random 
import cv2 
import matplotlib.pyplot as plt 

DATASET_PATH = "/kaggle/input/lgg-mri-segmentation/kaggle_3m" 

def load_random_sample(dataset_path): 
    """Load a random image and mask from the dataset.""" 
    case_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))] 
    case = random.choice(case_folders) 
    case_path = os.path.join(dataset_path, case) 
    
    image_files = [f for f in os.listdir(case_path) if f.endswith('.tif') and not f.endswith('_mask.tif')] 
    image_file = random.choice(image_files) 
    mask_file = image_file.replace('.tif', '_mask.tif') 
    
    image = cv2.imread(os.path.join(case_path, image_file)) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    # Resize image and mask to 128x128 for consistent visualization
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    mask = cv2.imread(os.path.join(case_path, mask_file), cv2.IMREAD_GRAYSCALE) 
    mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)
    
    return image, mask, case, image_file 

def overlay_mask(image, mask): 
    """Overlay mask on image.""" 
    overlay = image.copy() 
    overlay[mask > 0] = [255, 0, 0] 
    return overlay 

if __name__ == "__main__": 
    print("Loading random sample...") 
    image, mask, case, image_file = load_random_sample(DATASET_PATH) 
    overlay = overlay_mask(image, mask) 

    fig, axs = plt.subplots(1, 3, figsize=(15, 5)) 
    axs[0].imshow(image) 
    axs[0].set_title(f'Image\n{case}/{image_file}') 
    axs[1].imshow(mask, cmap='gray') 
    axs[1].set_title('Mask') 
    axs[2].imshow(overlay) 
    axs[2].set_title('Overlay') 
    
    for ax in axs: 
        ax.axis('off') 
    plt.tight_layout() 
    plt.show()