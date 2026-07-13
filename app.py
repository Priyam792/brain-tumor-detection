import os
import json
import secrets
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration paths
PROJECT_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_DIR / "static" / "uploads"
MODELS_DIR = PROJECT_DIR / "models"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure directories exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Global variables to store model and class index mapping
model = None
class_indices = None
invert_class_indices = None

def load_ml_model():
    global model, class_indices, invert_class_indices
    model_path = MODELS_DIR / "best_brain_tumor_cnn.keras"
    if not model_path.exists():
        model_path = MODELS_DIR / "brain_tumor_cnn_final.keras"
        
    class_map_path = MODELS_DIR / "class_indices.json"
    
    if model_path.exists():
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(str(model_path))
    else:
        print(f"Warning: Model not found at {model_path}.")
        
    if class_map_path.exists():
        print(f"Loading class mapping from {class_map_path}...")
        with open(class_map_path, "r", encoding="utf-8") as file:
            class_indices = json.load(file)
            # Invert class indices to mapping index -> class name
            invert_class_indices = {v: k for k, v in class_indices.items()}
    else:
        # Fallback to standard order from notebook
        class_indices = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
        invert_class_indices = {v: k for k, v in class_indices.items()}

# Load on startup
load_ml_model()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    # Preprocess image according to the notebook logic
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize to match network input shape
    img = img.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # Run prediction
    predictions = model.predict(img_array)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_class = invert_class_indices.get(predicted_idx, "Unknown")
    confidence = float(predictions[predicted_idx])
    
    # Format class probabilities
    probabilities = []
    for idx, prob in enumerate(predictions):
        class_name = invert_class_indices.get(idx, f"Class {idx}")
        probabilities.append({
            "class_name": class_name,
            "probability": float(prob),
            "percentage": f"{prob * 100:.2f}%"
        })
        
    # Sort probabilities by highest confidence
    probabilities = sorted(probabilities, key=lambda x: x["probability"], reverse=True)
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "probabilities": probabilities
    }

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if file is provided
        if "mri_image" not in request.files:
            flash("No file part in the request", "error")
            return redirect(request.url)
            
        file = request.files["mri_image"]
        if file.filename == "":
            flash("No file selected", "error")
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            # Save file securely
            filename = secure_filename(file.filename)
            unique_filename = f"{secrets.token_hex(8)}_{filename}"
            filepath = UPLOAD_FOLDER / unique_filename
            file.save(str(filepath))
            
            # Reload model if it was missing on startup
            global model
            if model is None:
                load_ml_model()
                
            if model is None:
                flash("Model is not available. Please train the model first.", "error")
                return redirect(request.url)
                
            try:
                # Run prediction
                result = predict_image(str(filepath))
                image_url = url_for("static", filename=f"uploads/{unique_filename}")
                
                return render_template(
                    "result.html",
                    image_url=image_url,
                    result=result
                )
            except Exception as e:
                if filepath.exists():
                    os.remove(filepath)
                flash(f"Inference failed: {str(e)}", "error")
                return redirect(request.url)
        else:
            flash("Invalid file type. Supported formats: PNG, JPG, JPEG", "error")
            return redirect(request.url)
            
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
