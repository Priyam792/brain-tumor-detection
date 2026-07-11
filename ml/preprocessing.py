import os
import random
import shutil
from pathlib import Path

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def clean_directory(path: Path):
    """Remove directory if it exists and recreate it."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def split_dataset(base_dir: Path, output_dir: Path, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """
    Combines the images from raw validation/test directories and splits them into
    train, validation, and test sets with the specified ratios.
    """
    print("=" * 60)
    print("REORGANIZING & SPLITTING DATASET")
    print("=" * 60)
    
    # Identify raw input folders
    raw_val_dir = base_dir / "validation"
    raw_test_dir = base_dir / "test"
    
    if not raw_val_dir.exists() or not raw_test_dir.exists():
        raise FileNotFoundError(
            f"Expected directories '{raw_val_dir}' and '{raw_test_dir}' to exist."
        )
        
    # Define processed output folders
    processed_train = output_dir / "train"
    processed_val = output_dir / "validation"
    processed_test = output_dir / "test"
    
    # Recreate output directories
    for folder in [processed_train, processed_val, processed_test]:
        for cls in CLASSES:
            clean_directory(folder / cls)
            
    # Process each class
    for cls in CLASSES:
        # Collect all image paths for this class from test and validation
        cls_images = []
        for src_dir in [raw_val_dir, raw_test_dir]:
            cls_src = src_dir / cls
            if cls_src.exists():
                for file in cls_src.iterdir():
                    if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                        cls_images.append(file)
                        
        # Shuffle images
        random.shuffle(cls_images)
        total_images = len(cls_images)
        
        # Calculate split boundaries
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        # Test count takes the remainder to avoid rounding issues
        test_count = total_images - train_count - val_count
        
        train_imgs = cls_images[:train_count]
        val_imgs = cls_images[train_count:train_count + val_count]
        test_imgs = cls_images[train_count + val_count:]
        
        print(f"Class: {cls} | Total: {total_images}")
        print(f"  - Train      : {len(train_imgs)}")
        print(f"  - Validation : {len(val_imgs)}")
        print(f"  - Test       : {len(test_imgs)}")
        
        # Copy files to destinations
        for img, dest_dir in [
            (train_imgs, processed_train),
            (val_imgs, processed_val),
            (test_imgs, processed_test)
        ]:
            for img_path in img:
                shutil.copy2(img_path, dest_dir / cls / img_path.name)
                
    print("\nDataset split and copy completed successfully!")
    print(f"Processed dataset saved to: {output_dir.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    base_dataset_dir = project_dir / "dataset"
    processed_dataset_dir = base_dataset_dir / "processed"
    
    split_dataset(base_dataset_dir, processed_dataset_dir)
