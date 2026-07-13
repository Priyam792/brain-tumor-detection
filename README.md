# Brain Tumor Detection Webapp

A simplified, AI-powered brain MRI classification web application. It allows users to upload a brain MRI scan and predicts the presence of brain tumors (Glioma, Meningioma, Pituitary Tumor, or No Tumor) with a custom Convolutional Neural Network (CNN) trained using Keras/TensorFlow.

This is a college project rebuilt directly around the `01_brain_tumor_cnn.ipynb` notebook as the single source of truth.

---

## Features

- **Custom CNN Classifier:** Replicates the 4-layer Custom CNN model from the notebook.
- **Sleek Web Interface:** Clean drag-and-drop file upload zone and dynamic classification result progress bars styled with a premium dark theme and glassmorphism.
- **Command-Line Training:** Includes a Python training script so the model can be retrained directly from the CLI.

---

## Project Structure

```text
brain-tumor-detection/
├── app.py                      # Flask Web Application
├── train.py                    # Command-line Model Training script
├── requirement.txt             # Python Package Dependencies
├── README.md                   # Project Documentation
├── dataset/                    # Dataset directory
│   └── processed/
│       ├── train/              # Training images per class
│       ├── validation/         # Validation images per class
│       └── test/               # Test images per class
├── models/                     # Saved Keras models & config
│   ├── class_indices.json      # Class index mapping
│   ├── best_brain_tumor_cnn.keras  # Best model checkpoint
│   └── brain_tumor_cnn_final.keras # Final saved model
├── notebooks/                  # Project Notebooks
│   └── 01_brain_tumor_cnn.ipynb    # Single Source of Truth
├── static/                     # Web Application Assets (Uploads & Styles)
│   ├── css/
│   │   └── style.css           # Premium Dark Mode Glassmorphism Stylesheet
│   └── uploads/                # Temporary uploaded scan directory
└── templates/                  # HTML Views (Jinja2 Templates)
    ├── index.html              # Drag-and-drop Scan Upload
    └── result.html             # Diagnostic Result Visualizer
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 - 3.14
- Virtualenv (`python3 -m venv`)

### 2. Configure Virtual Environment & Dependencies
Clone the repository, create a virtual environment, and install dependencies:
```bash
git clone https://github.com/Priyam792/brain-tumor-detection.git
cd brain-tumor-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

---

## Training the Model

The project comes pre-populated with trained weights (`best_brain_tumor_cnn.keras`), but you can train the model from scratch at any time:
```bash
python train.py
```
This runs the entire preprocessing, validation split, CNN model building, 20-epoch training, and test set evaluation pipeline as demonstrated in the notebook.

---

## Running the Web Application

To run the Flask development server locally:
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser. Drag and drop any MRI image from your test dataset to predict its classification and view confidence levels.

---

## Educational Disclaimer

**This system is built strictly for educational, research, and demonstration purposes.** The machine learning models and visualizations are not clinically validated and must not be used as medical diagnostic tools.
