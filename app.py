import os
import json
import secrets
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename

from db import init_db, insert_prediction, get_prediction_by_id, get_all_predictions, delete_prediction, get_analytics_summary
from ml.predict_helper import predict_brain_tumor, get_best_model_name
from report_generator import generate_pdf_report

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Configuration paths
PROJECT_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = PROJECT_DIR / "static" / "uploads"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "static" / "reports"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024 # 16 MB limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Ensure directories exist
for folder in [UPLOAD_FOLDER, MODELS_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Initialize database
init_db()

@app.context_processor
def inject_now():
    from datetime import datetime
    return {'now': datetime.utcnow()}

@app.route("/")
def index():
    """Home / Landing page."""
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Upload and predict brain tumor."""
    if request.method == "POST":
        # Check if file is in request
        if "mri_image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        file = request.files["mri_image"]
        model_name = request.form.get("model_name", "Best_Model")
        
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add unique prefix to avoid collisions
            unique_filename = f"{secrets.token_hex(8)}_{filename}"
            filepath = UPLOAD_FOLDER / unique_filename
            file.save(str(filepath))
            
            try:
                # Run ML Prediction and Grad-CAM
                result = predict_brain_tumor(
                    image_path=str(filepath),
                    model_name=model_name,
                    models_dir=MODELS_DIR,
                    outputs_dir=UPLOAD_FOLDER
                )
                
                # Save details in SQLite database
                original_img_rel = f"static/uploads/{unique_filename}"
                prediction_id = insert_prediction(
                    image_name=filename,
                    predicted_class=result["predicted_class"],
                    confidence=result["confidence"],
                    probabilities=result["probabilities"],
                    processing_time=result["processing_time"],
                    original_image_path=original_img_rel,
                    gradcam_image_path=result["gradcam_image_path"]
                )
                
                return jsonify({
                    "success": True,
                    "prediction_id": prediction_id,
                    "redirect_url": url_for("result", prediction_id=prediction_id)
                })
                
            except Exception as e:
                # Cleanup file if prediction failed
                if filepath.exists():
                    os.remove(filepath)
                import traceback
                print(traceback.format_exc())
                return jsonify({"error": f"Model inference failed: {str(e)}"}), 500
        else:
            return jsonify({"error": "Invalid file type. Supported: JPG, JPEG, PNG"}), 400
            
    # GET request
    best_model = get_best_model_name(MODELS_DIR)
    
    # Check if models exist to list them
    models_available = []
    if (MODELS_DIR / "best_custom_cnn.keras").exists():
        models_available.append(("Custom_CNN", "Custom CNN"))
    if (MODELS_DIR / "best_mobilenetv2_transfer.keras").exists():
        models_available.append(("MobileNetV2", "MobileNetV2 (Transfer Learning)"))
    if (MODELS_DIR / "best_efficientnetb0_transfer.keras").exists():
        models_available.append(("EfficientNetB0", "EfficientNetB0 (Transfer Learning)"))
        
    return render_template("predict.html", best_model=best_model, models_available=models_available)

@app.route("/result/<int:prediction_id>")
def result(prediction_id):
    """Detailed result page."""
    record = get_prediction_by_id(prediction_id)
    if not record:
        flash("Record not found", "error")
        return redirect(url_for("predict"))
        
    return render_template("result.html", record=record)

@app.route("/dashboard")
def dashboard():
    """Analytics dashboard page."""
    analytics = get_analytics_summary()
    
    # Model comparison info
    comparison_path = MODELS_DIR / "model_comparison.json"
    comparison_data = None
    if comparison_path.exists():
        with open(comparison_path, "r") as f:
            comparison_data = json.load(f)
            
    return render_template("dashboard.html", analytics=analytics, comparison_data=comparison_data)

@app.route("/history")
def history():
    """Searchable prediction history table."""
    page = request.args.get("page", 1, type=int)
    search_query = request.args.get("q", "", type=str)
    limit = 10
    offset = (page - 1) * limit
    
    records, total_count = get_all_predictions(search_query, limit, offset)
    total_pages = (total_count + limit - 1) // limit
    
    return render_template(
        "history.html",
        records=records,
        page=page,
        total_pages=total_pages,
        total_count=total_count,
        search_query=search_query
    )

@app.route("/delete/<int:prediction_id>", methods=["POST"])
def delete(prediction_id):
    """Delete prediction record and its files."""
    row = delete_prediction(prediction_id)
    if row:
        orig_path = PROJECT_DIR / row["original_image_path"]
        gradcam_path = PROJECT_DIR / row["gradcam_image_path"]
        
        # Remove files from disk if they exist
        if orig_path.exists():
            os.remove(orig_path)
        if gradcam_path.exists():
            os.remove(gradcam_path)
            
        flash("Scan record deleted successfully.", "success")
    else:
        flash("Record not found.", "error")
        
    return redirect(url_for("history"))

@app.route("/download_report/<int:prediction_id>")
def download_report(prediction_id):
    """Generate and download PDF report."""
    record = get_prediction_by_id(prediction_id)
    if not record:
        return "Record not found", 404
        
    pdf_filename = f"brain_tumor_report_{record['id']}.pdf"
    pdf_path = REPORTS_DIR / pdf_filename
    
    try:
        generate_pdf_report(record, str(pdf_path))
        return send_file(str(pdf_path), as_attachment=True, download_name=pdf_filename)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Failed to generate report: {str(e)}", 500

@app.route("/about")
def about():
    """About page."""
    return render_template("about.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
