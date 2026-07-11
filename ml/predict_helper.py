import os
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

from ml.gradcam import find_last_conv_layer, make_gradcam_heatmap, save_and_display_gradcam

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Class names mapping from indices
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor"
}

def download_model_if_missing(filename: str, models_dir: Path):
    """
    Downloads a missing model file from GitHub Releases.
    """
    model_path = models_dir / filename
    if model_path.exists():
        return
        
    models_dir.mkdir(parents=True, exist_ok=True)
    # Target URL of the v1.0.0 GitHub release
    url = f"https://github.com/Priyam792/brain-tumor-detection/releases/download/v1.0.0/{filename}"
    print(f"Model file {filename} not found locally.")
    print(f"Attempting to download from GitHub Releases: {url}...")
    
    try:
        # Disable SSL warnings or verify depending on environment. Github uses valid SSL.
        urllib.request.urlretrieve(url, str(model_path))
        print(f"Successfully downloaded {filename} to {model_path}")
    except Exception as e:
        print(f"Warning: Could not download {filename} from remote release: {e}")
        print("The system will attempt to look for fallback local model files.")

def get_best_model_name(models_dir: Path):
    """
    Finds the model name with the highest accuracy from model_comparison.json,
    defaulting to 'MobileNetV2' if not found.
    """
    comparison_path = models_dir / "model_comparison.json"
    if comparison_path.exists():
        try:
            with open(comparison_path, "r") as f:
                data = json.load(f)
            best_model = None
            best_accuracy = -1.0
            for model_name, metrics in data.items():
                if metrics["accuracy"] > best_accuracy:
                    best_accuracy = metrics["accuracy"]
                    best_model = model_name
            if best_model:
                return best_model
        except Exception as e:
            print(f"Error reading model comparison: {e}")
            
    return "MobileNetV2"

def load_prediction_model(model_name: str, models_dir: Path):
    """
    Loads a specific trained model by name.
    """
    model_mapping = {
        "Custom_CNN": "best_custom_cnn.keras",
        "MobileNetV2": "best_mobilenetv2_transfer.keras",
        "EfficientNetB0": "best_efficientnetb0_transfer.keras"
    }
    
    filename = model_mapping.get(model_name, "best_mobilenetv2_transfer.keras")
    
    # Attempt download if missing
    download_model_if_missing(filename, models_dir)
    
    model_path = models_dir / filename
    
    # Fallback to any available keras file if the requested one doesn't exist
    if not model_path.exists():
        print(f"Model {model_path.name} not found, searching for fallbacks...")
        keras_files = list(models_dir.glob("*.keras"))
        if keras_files:
            model_path = keras_files[0]
            print(f"Using fallback model: {model_path.name}")
        else:
            raise FileNotFoundError(f"No trained Keras models found in {models_dir.resolve()}. Please download the models or run training first.")
            
    return tf.keras.models.load_model(str(model_path))

def preprocess_single_image(image_path: str):
    """
    Loads and preprocesses a single image for TensorFlow prediction.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    return img_array

def predict_brain_tumor(image_path: str, model_name: str = None, models_dir: Path = None, outputs_dir: Path = None):
    """
    Executes end-to-end prediction:
    1. Loads the specified model (or the best model).
    2. Runs inference and measures execution time.
    3. Generates the Grad-CAM heatmap.
    4. Formats predictions and returns the results.
    """
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
    if outputs_dir is None:
        outputs_dir = Path(__file__).resolve().parent.parent / "static" / "uploads"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Determine model to load
    if not model_name or model_name == "Best_Model":
        model_name = get_best_model_name(models_dir)
        print(f"Auto-selected best model: {model_name}")
        
    model = load_prediction_model(model_name, models_dir)
    
    # 2. Preprocess image
    img_array = preprocess_single_image(image_path)
    
    # 3. Inference & Time measurement
    start_time = time.time()
    preds = model.predict(img_array)
    processing_time = time.time() - start_time
    
    # Probabilities mapping
    probabilities = {}
    for i, name in enumerate(CLASS_NAMES):
        prob = float(preds[0][i])
        probabilities[CLASS_DISPLAY_NAMES[name]] = round(prob, 4)
        
    pred_idx = int(np.argmax(preds[0]))
    predicted_class_raw = CLASS_NAMES[pred_idx]
    predicted_class = CLASS_DISPLAY_NAMES[predicted_class_raw]
    confidence = float(preds[0][pred_idx])
    
    # 4. Generate Grad-CAM Heatmap
    last_conv_layer_name = find_last_conv_layer(model)
    print(f"Generating Grad-CAM using layer: {last_conv_layer_name}")
    
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_idx)
    
    # Create output paths
    img_filename = Path(image_path).name
    gradcam_filename = f"gradcam_{img_filename}"
    gradcam_path = outputs_dir / gradcam_filename
    
    save_and_display_gradcam(image_path, heatmap, str(gradcam_path))
    
    return {
        "model_used": model_name,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "processing_time": processing_time,
        "gradcam_image_path": str(gradcam_path.relative_to(outputs_dir.parent.parent)) # Relative to static/
    }
