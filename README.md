# 🧠 Brain Tumor Detection Web Application

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask)
![HTML5](https://img.shields.io/badge/HTML5-orange?logo=html5)
![CSS3](https://img.shields.io/badge/CSS3-blue?logo=css3)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Made with ❤️](https://img.shields.io/badge/Made%20with-Python-blue)

**An AI-powered Brain MRI classification web app built with TensorFlow/Keras, Flask, and a custom CNN.**

[Overview](#-overview) •
[Demo](#-demo) •
[Features](#-features) •
[Installation](#-installation) •
[Usage](#️-running-the-application) •
[Model](#-model-architecture) •
[Contributing](#-contributing)

</div>

---

## 📖 Overview

This application allows users to upload a brain MRI scan and receive an instant prediction of one of four classes, along with a confidence score:

| Class | Description |
|-------|-------------|
| 🔴 Glioma | Tumor arising from glial cells |
| 🟠 Meningioma | Tumor arising from the meninges |
| 🟡 Pituitary Tumor | Tumor located in the pituitary gland |
| 🟢 No Tumor | No abnormality detected |

The project demonstrates a complete deep learning workflow — from dataset preprocessing and CNN training to deploying the trained model behind a Flask web interface.

> **Note:** This project was developed as a college project for educational and learning purposes.

> ⚠️ **Disclaimer:** Predictions are **not medically certified** and must **never** be used for real clinical diagnosis or treatment decisions. Always consult a qualified medical professional.

---

## 🎥 Demo

<div align="center">

*Add a screenshot or GIF of the upload page and prediction result here, e.g.*

`docs/screenshots/upload.png` · `docs/screenshots/result.png`

</div>

---

## ✨ Features

- 🧠 Custom CNN model built with TensorFlow/Keras
- 🌐 Lightweight Flask backend for real-time inference
- 📤 Drag-and-drop MRI image upload
- 📊 Per-class confidence score visualization
- 🎨 Modern dark UI with glassmorphism design
- ⚡ Fast, real-time predictions on CPU or GPU
- 🔁 One-command retraining via `train.py`
- 📚 Fully documented Jupyter Notebook walkthrough
- 🖥️ Responsive, mobile-friendly interface

---

## 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Backend | Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Deep Learning | TensorFlow, Keras |
| Data Processing | NumPy |
| Image Processing | Pillow |
| Notebook | Jupyter Notebook |
| Version Control | Git & GitHub |

---

## 🧩 Model Architecture

The classifier is a custom Convolutional Neural Network trained on labeled brain MRI scans, using standard techniques for medical image classification:

- Convolution + ReLU + MaxPooling blocks for hierarchical feature extraction
- Batch normalization for stable, faster convergence
- Dropout layers to reduce overfitting
- Dense classification head with softmax output over 4 classes
- Data augmentation (rotation, flipping, zoom) during training to improve generalization

Full architecture details, training curves, and evaluation metrics are available in `notebooks/01_brain_tumor_cnn.ipynb`.

> 💡 **Tip:** Add your actual accuracy / precision / recall / F1 numbers and a confusion matrix image here once available, e.g.:
>
> | Metric | Score |
> |--------|-------|
> | Accuracy | xx.x% |
> | Precision | xx.x% |
> | Recall | xx.x% |
> | F1-score | xx.x% |

---

## 📦 Download Pre-trained Model

GitHub enforces a **100 MB file size limit**, so the trained `.keras` model files are **not included** in this repository.

**📥 Download from Google Drive:**
👉 https://drive.google.com/drive/folders/10B6OCV832H0Jr5vBFptZ06ILJtCiJMWa?usp=drive_link

After downloading, place the model files inside:

```text
models/
├── best_brain_tumor_cnn.keras
└── brain_tumor_cnn_final.keras
```

Without these files, the web application will not be able to make predictions.

---

## 📁 Project Structure

```text
brain-tumor-detection/
├── app.py                      # Flask web application
├── train.py                    # Model training script
├── requirement.txt             # Python dependencies
├── README.md
│
├── dataset/
│   └── processed/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── models/
│   ├── class_indices.json
│   ├── best_brain_tumor_cnn.keras
│   └── brain_tumor_cnn_final.keras
│
├── notebooks/
│   └── 01_brain_tumor_cnn.ipynb
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
│
└── templates/
    ├── index.html
    └── result.html
```

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Priyam792/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Create a Virtual Environment

**Linux/macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirement.txt
```

### 4. Download the Trained Model

Download the model files from the [Google Drive link](https://drive.google.com/drive/folders/10B6OCV832H0Jr5vBFptZ06ILJtCiJMWa?usp=drive_link) above and place them inside `models/`.

---

## 🏋️ Training the Model

To train the CNN model from scratch:

```bash
python train.py
```

This script handles:

- Dataset loading
- Image preprocessing
- Training
- Validation
- Model checkpointing
- Model evaluation
- Saving trained weights

---

## ▶️ Running the Application

Start the Flask development server:

```bash
python app.py
```

Then open your browser at:

```
http://localhost:5000
```

Upload an MRI image and the application will return the predicted class along with a confidence score.

---

## 📒 Notebook

A complete, self-contained Jupyter Notebook is included, covering:

- Dataset preprocessing
- CNN architecture design
- Model training
- Validation
- Performance evaluation

📂 `notebooks/01_brain_tumor_cnn.ipynb`

---

## 🗺️ Roadmap

- [ ] Grad-CAM visualization for model explainability
- [ ] Transfer learning (MobileNetV2 / EfficientNet)
- [ ] Prediction history database
- [ ] Docker support
- [ ] Cloud deployment (Render / AWS / GCP)
- [ ] REST API for programmatic access
- [ ] User authentication

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please open an issue first for major changes to discuss what you'd like to add.

---

## ⚠️ Educational Disclaimer

This project is developed **strictly for educational, research, and demonstration purposes only**.

The predictions generated by this application are **not medically certified** and **must not** be used for clinical diagnosis, treatment planning, or any healthcare decision.

**Always consult qualified medical professionals for diagnosis and treatment.**

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future improvements.

---

<div align="center">
Made with ❤️ and TensorFlow
</div>
