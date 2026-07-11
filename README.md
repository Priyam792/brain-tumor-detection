# Brain Tumor Detection and Analysis System

An AI-powered, production-quality medical imaging web application that allows users to upload brain MRI scans and predicts the presence of brain tumors (Glioma, Meningioma, Pituitary Tumor, or No Tumor). The system displays classification probabilities, computes inference latency, visualizes attention areas using **Grad-CAM (Explainable AI)**, logs prediction histories to **SQLite**, and automatically generates downloadable **PDF reports**.

---

## Key Features

- **Multi-Model Classifiers:** Combines a Custom CNN, MobileNetV2, and EfficientNetB0 to compare classification metrics (Accuracy, Precision, Recall, F1-Score).
- **Explainable AI (XAI):** Generates Grad-CAM visual heatmaps overlaying raw grayscale MRI scans to highlight regions of interest.
- **Analytics Dashboard:** Visualizes diagnostic class distributions (Chart.js) and model metrics.
- **Searchable History:** Interactive, paginated prediction histories with search filters and record deletion.
- **PDF Report Generation:** Creates professional medical reports detailing findings, class probabilities, and side-by-side scans.
- **Jupyter Notebooks:** Contains 7 structured educational notebooks for dataset analysis, training, and XAI.

---

## Project Structure

```text
brain-tumor-detection/
├── app.py                      # Flask Application Entrypoint
├── db.py                       # SQLite Database Management
├── report_generator.py         # PDF Report Generation Module
├── requirement.txt             # Python Package Dependencies
├── README.md                   # Project Documentation
├── ml/                         # Machine Learning Modules
│   ├── preprocessing.py        # Dataset Stratified Split Utility
│   ├── train.py                # Model Architectures & Training
│   ├── evaluate.py             # Model Evaluation on Test Set
│   ├── gradcam.py              # Grad-CAM Heatmap Calculations
│   └── predict_helper.py       # Single Scan Inference Helper
├── models/                     # Saved Models & Configuration
│   ├── class_indices.json      # Class Names to Indices Mapping
│   ├── model_comparison.json   # Model Evaluation Metrics Summary
│   ├── best_custom_cnn.keras   # Saved Custom CNN
│   ├── best_mobilenetv2_transfer.keras  # Saved MobileNetV2
│   └── best_efficientnetb0_transfer.keras # Saved EfficientNetB0
├── notebooks/                  # Educational Jupyter Notebooks
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_custom_cnn_training.ipynb
│   ├── 04_transfer_learning.ipynb
│   ├── 05_model_evaluation.ipynb
│   ├── 06_gradcam_visualization.ipynb
│   └── 07_final_testing.ipynb
├── static/                     # Web Application Assets
│   ├── css/
│   │   └── style.css           # Premium Glassmorphic Dark Styles
│   ├── uploads/                # Saved Scans & Grad-CAM Heatmaps
│   ├── images/
│   │   └── model_comparison/   # Confusion Matrices & Comparison Charts
│   └── reports/                # Temporary PDF Files
└── templates/                  # HTML Views (Jinja2 Templates)
    ├── index.html              # Landing Page
    ├── predict.html            # MRI File Upload
    ├── result.html             # Diagnostic Result Visualizer
    ├── dashboard.html          # Analytics & Chart.js Visuals
    ├── history.html            # Paginated History List
    └── about.html              # Tumor Details & XAI Intuition
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 - 3.14
- Virtualenv (`python3 -m venv`)

### 2. Configure Virtual Environment
Clone the repository and initialize the Python virtual environment:
```bash
git clone https://github.com/Priyam792/brain-tumor-detection.git
cd brain-tumor-detection
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install packages listed in `requirement.txt`.
```bash
pip install -r requirement.txt
```
*Note: If using Python 3.14+, `tf-nightly` is automatically pulled to ensure compatibility with experimental Python releases.*

---

## Dataset Reorganization & Training Pipeline

The system is equipped with modular machine learning scripts that preprocess the dataset, train three architectures, and save the best weights.

### Step 1: Preprocess and Split Dataset
Runs a stratified 70/15/15 split of the dataset to create `dataset/processed/`:
```bash
python ml/preprocessing.py
```

### Step 2: Train Model Architectures
Trains the Custom CNN, MobileNetV2 (Transfer Learning), and EfficientNetB0 (Transfer Learning) models:
```bash
python ml/train.py
```

### Step 3: Run Model Evaluations
Evaluates the saved models on the test set, computing metrics and plotting confusion matrices:
```bash
python ml/evaluate.py
```

---

## Running the Web Application

To run the Flask development server locally:
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

## Database Schema (SQLite)

The prediction history is stored in an SQLite database file `brain_tumor.db` containing the `predictions` table:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER` | Primary Key, Auto-incremented ID |
| `image_name` | `TEXT` | File name of the uploaded MRI scan |
| `predicted_class` | `TEXT` | Predicted diagnosis (Glioma, Meningioma, Pituitary, No Tumor) |
| `confidence` | `REAL` | Prediction confidence (score between 0.0 and 1.0) |
| `probabilities` | `TEXT` | JSON string storing distribution scores for all classes |
| `date` | `TEXT` | YYYY-MM-DD date stamp |
| `time` | `TEXT` | HH:MM:SS time stamp |
| `processing_time` | `REAL` | Inference duration in seconds |
| `original_image_path`| `TEXT` | Filepath to original uploaded MRI |
| `gradcam_image_path` | `TEXT` | Filepath to generated Grad-CAM heatmap overlay |

---

## API Documentation

### 1. Upload & Predict MRI Scan
- **Endpoint:** `/predict`
- **Method:** `POST`
- **Payload:** `multipart/form-data`
  - `mri_image`: Binary image file (PNG, JPG, JPEG)
  - `model_name`: Classifier model name (`Best_Model`, `Custom_CNN`, `MobileNetV2`, `EfficientNetB0`)
- **Response (Success):**
  ```json
  {
    "success": true,
    "prediction_id": 12,
    "redirect_url": "/result/12"
  }
  ```
- **Response (Error):**
  ```json
  {
    "error": "Invalid file type. Supported: JPG, JPEG, PNG"
  }
  ```

### 2. Download Report
- **Endpoint:** `/download_report/<prediction_id>`
- **Method:** `GET`
- **Response:** PDF file attachment containing visual overlays and analysis metrics.

---

## Deployment Instructions

### 1. Deploying on Render (or Railway)
Create a Web Service on Render and point it to your GitHub repository:
- **Build Command:**
  ```bash
  pip install -r requirement.txt && python ml/preprocessing.py && python ml/train.py && python ml/evaluate.py
  ```
- **Start Command:**
  ```bash
  gunicorn app:app
  ```
- **Environment Variables:** Set `PYTHON_VERSION` to `3.10` or higher.

---

## Educational Disclaimer

**NeuroScan is built strictly for educational, research, and demonstration purposes.** The machine learning models and visualizations are not clinically validated and must not be used as medical diagnostic tools.
