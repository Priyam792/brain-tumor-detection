import json
from pathlib import Path

def create_notebook_json(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def make_markdown_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    }

def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    }

def build_all_notebooks(notebooks_dir: Path):
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    # ----------------------------------------------------
    # Notebook 1: Dataset Analysis
    # ----------------------------------------------------
    nb1_cells = [
        make_markdown_cell([
            "# Notebook 1: Dataset Analysis",
            "This notebook performs an exploratory data analysis (EDA) on the Brain Tumor MRI dataset.",
            "We will count the files, check class balance, visualize sample scans, and identify image dimensions."
        ]),
        make_code_cell([
            "import os",
            "from pathlib import Path",
            "import matplotlib.pyplot as plt",
            "import cv2",
            "from PIL import Image",
            "import random",
            "",
            "BASE_DIR = Path('../dataset')",
            "CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']",
            "print('Dataset path:', BASE_DIR.resolve())"
        ]),
        make_markdown_cell([
            "### Count Images in Validation and Test sets"
        ]),
        make_code_cell([
            "for split in ['validation', 'test']:",
            "    print(f'=== Split: {split} ===')",
            "    split_dir = BASE_DIR / split",
            "    total = 0",
            "    for cls in CLASSES:",
            "        cls_dir = split_dir / cls",
            "        count = len(list(cls_dir.glob('*'))) if cls_dir.exists() else 0",
            "        total += count",
            "        print(f'  Class {cls:<12}: {count} images')",
            "    print(f'  Total {split:<12}: {total} images\\n')"
        ]),
        make_markdown_cell([
            "### Visualize Sample MRI Scans from Each Class"
        ]),
        make_code_cell([
            "fig, axes = plt.subplots(1, 4, figsize=(16, 4))",
            "for i, cls in enumerate(CLASSES):",
            "    # Load a random image from validation set",
            "    cls_dir = BASE_DIR / 'validation' / cls",
            "    images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))",
            "    img_path = random.choice(images)",
            "    ",
            "    # Read using PIL",
            "    img = Image.open(img_path)",
            "    axes[i].imshow(img, cmap='gray')",
            "    axes[i].set_title(f'{cls.upper()}\\n{img.size} px', fontsize=12, fontweight='bold')",
            "    axes[i].axis('off')",
            "plt.tight_layout()",
            "plt.show()"
        ])
    ]
    
    # ----------------------------------------------------
    # Notebook 2: Data Preprocessing
    # ----------------------------------------------------
    nb2_cells = [
        make_markdown_cell([
            "# Notebook 2: Data Preprocessing & Augmentation",
            "This notebook details our preprocessing and data augmentation pipeline.",
            "We demonstrate how to split raw data and augment scans using `ImageDataGenerator`."
        ]),
        make_code_cell([
            "import os",
            "import numpy as np",
            "from pathlib import Path",
            "import matplotlib.pyplot as plt",
            "import tensorflow as tf",
            "from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img",
            "",
            "# Import our preprocessing split function",
            "import sys",
            "sys.path.append('..')",
            "from ml.preprocessing import split_dataset",
            "",
            "BASE_DIR = Path('../dataset')",
            "PROCESSED_DIR = BASE_DIR / 'processed'",
            "print('Processed folder:', PROCESSED_DIR.resolve())"
        ]),
        make_markdown_cell([
            "### Step 1: Split and Reorganize Dataset",
            "If you haven't run the split script, let's run it here. It combines validation and test sets and performs a stratified 70/15/15 split."
        ]),
        make_code_cell([
            "if not PROCESSED_DIR.exists():",
            "    split_dataset(BASE_DIR, PROCESSED_DIR)",
            "else:",
            "    print('Processed dataset already split and ready.')"
        ]),
        make_markdown_cell([
            "### Step 2: Configure Image Data Generator (Data Augmentation)",
            "Augmentations help prevent overfitting on small datasets."
        ]),
        make_code_cell([
            "train_datagen = ImageDataGenerator(",
            "    rescale=1.0 / 255,",
            "    rotation_range=15,",
            "    width_shift_range=0.1,",
            "    height_shift_range=0.1,",
            "    zoom_range=0.1,",
            "    horizontal_flip=True,",
            "    fill_mode='nearest'",
            ")",
            "",
            "print('Training generator configured with Rescaling, Rotations, Shifts, Zoom, and Flips.')"
        ]),
        make_markdown_cell([
            "### Step 3: Visualize Augmented Images for a Single Scan"
        ]),
        make_code_cell([
            "# Pick a random image",
            "sample_path = list((PROCESSED_DIR / 'train' / 'glioma').glob('*'))[0]",
            "img = load_img(sample_path, target_size=(224, 224))",
            "x = img_to_array(img) / 255.0",
            "x = np.expand_dims(x, axis=0)",
            "",
            "fig, axes = plt.subplots(1, 4, figsize=(16, 4))",
            "axes[0].imshow(img)",
            "axes[0].set_title('Original Image', fontweight='bold')",
            "axes[0].axis('off')",
            "",
            "# Generate 3 random augmented versions",
            "i = 1",
            "for batch in train_datagen.flow(x, batch_size=1):",
            "    axes[i].imshow(batch[0])",
            "    axes[i].set_title(f'Augmented {i}', fontweight='bold')",
            "    axes[i].axis('off')",
            "    i += 1",
            "    if i > 3:",
            "        break",
            "plt.tight_layout()",
            "plt.show()"
        ])
    ]
    
    # ----------------------------------------------------
    # Notebook 3: Custom CNN Training
    # ----------------------------------------------------
    nb3_cells = [
        make_markdown_cell([
            "# Notebook 3: Custom CNN Training",
            "This notebook builds, compiles, and trains a custom convolutional neural network from scratch."
        ]),
        make_code_cell([
            "import os",
            "import json",
            "from pathlib import Path",
            "import matplotlib.pyplot as plt",
            "import tensorflow as tf",
            "from tensorflow.keras import layers, models",
            "",
            "sys.path.append('..')",
            "from ml.train import get_data_generators, build_custom_cnn, train_model",
            "",
            "PROCESSED_DIR = Path('../dataset/processed')",
            "MODELS_DIR = Path('../models')",
            "",
            "train_gen, val_gen, test_gen = get_data_generators(PROCESSED_DIR)",
            "num_classes = train_gen.num_classes"
        ]),
        make_markdown_cell([
            "### Step 1: Model Architecture Definition",
            "Our custom CNN features 4 Conv2D blocks with MaxPool2D, followed by a Flatten layer, Dense layer with Dropout, and Softmax output."
        ]),
        make_code_cell([
            "model = build_custom_cnn(num_classes)",
            "model.summary()"
        ]),
        make_markdown_cell([
            "### Step 2: Compile & Train",
            "We use the Adam optimizer, categorical crossentropy loss, and early stopping / checkpoint callbacks."
        ]),
        make_code_cell([
            "# We compile and fit the model",
            "history = train_model(model, train_gen, val_gen, MODELS_DIR)"
        ]),
        make_markdown_cell([
            "### Step 3: Visualize Loss & Accuracy Curves"
        ]),
        make_code_cell([
            "acc = history.history['accuracy']",
            "val_acc = history.history['val_accuracy']",
            "loss = history.history['loss']",
            "val_loss = history.history['val_loss']",
            "epochs_range = range(len(acc))",
            "",
            "plt.figure(figsize=(12, 5))",
            "plt.subplot(1, 2, 1)",
            "plt.plot(epochs_range, acc, label='Training Accuracy')",
            "plt.plot(epochs_range, val_acc, label='Validation Accuracy')",
            "plt.legend(loc='lower right')",
            "plt.title('Training and Validation Accuracy')",
            "",
            "plt.subplot(1, 2, 2)",
            "plt.plot(epochs_range, loss, label='Training Loss')",
            "plt.plot(epochs_range, val_loss, label='Validation Loss')",
            "plt.legend(loc='upper right')",
            "plt.title('Training and Validation Loss')",
            "plt.show()"
        ])
    ]
    
    # ----------------------------------------------------
    # Notebook 4: Transfer Learning
    # ----------------------------------------------------
    nb4_cells = [
        make_markdown_cell([
            "# Notebook 4: Transfer Learning (MobileNetV2 & EfficientNetB0)",
            "This notebook demonstrates transfer learning using pre-trained ImageNet architectures.",
            "We train and save MobileNetV2 and EfficientNetB0, keeping base layers frozen."
        ]),
        make_code_cell([
            "import os",
            "from pathlib import Path",
            "import tensorflow as tf",
            "from tensorflow.keras import layers, models",
            "",
            "sys.path.append('..')",
            "from ml.train import get_data_generators, build_mobilenetv2, build_efficientnetb0, train_model",
            "",
            "PROCESSED_DIR = Path('../dataset/processed')",
            "MODELS_DIR = Path('../models')",
            "",
            "train_gen, val_gen, test_gen = get_data_generators(PROCESSED_DIR)",
            "num_classes = train_gen.num_classes"
        ]),
        make_markdown_cell([
            "### Step 1: MobileNetV2 Transfer Learning Model"
        ]),
        make_code_cell([
            "mobilenet = build_mobilenetv2(num_classes)",
            "mobilenet.summary()",
            "",
            "print('Training MobileNetV2 (with frozen base layer weights)...')",
            "train_model(mobilenet, train_gen, val_gen, MODELS_DIR)"
        ]),
        make_markdown_cell([
            "### Step 2: EfficientNetB0 Transfer Learning Model"
        ]),
        make_code_cell([
            "efficientnet = build_efficientnetb0(num_classes)",
            "efficientnet.summary()",
            "",
            "print('Training EfficientNetB0 (with frozen base layer weights)...')",
            "train_model(efficientnet, train_gen, val_gen, MODELS_DIR)"
        ])
    ]
    
    # ----------------------------------------------------
    # Notebook 5: Model Evaluation
    # ----------------------------------------------------
    nb5_cells = [
        make_markdown_cell([
            "# Notebook 5: Model Evaluation & Comparison",
            "In this notebook, we load the trained models, evaluate them on the test set,",
            "calculate classification metrics, print a comparison table, and plot confusion matrices."
        ]),
        make_code_cell([
            "import os",
            "import json",
            "from pathlib import Path",
            "import numpy as np",
            "import tensorflow as tf",
            "import matplotlib.pyplot as plt",
            "",
            "sys.path.append('..')",
            "from ml.evaluate import evaluate_all",
            "",
            "print('Evaluating all models and saving statistics...')"
        ]),
        make_markdown_cell([
            "### Step 1: Run End-to-End Evaluation",
            "We execute the evaluation script. It automatically plots confusion matrices and saves comparison data in `models/model_comparison.json`."
        ]),
        make_code_cell([
            "evaluate_all()"
        ]),
        make_markdown_cell([
            "### Step 2: Load and Display Model Accuracy comparison Chart",
            "Let's display the metrics comparison chart generated by the evaluation script."
        ]),
        make_code_cell([
            "chart_path = Path('../static/images/model_comparison/metrics_comparison.png')",
            "if chart_path.exists():",
            "    img = plt.imread(str(chart_path))",
            "    plt.figure(figsize=(10, 6))",
            "    plt.imshow(img)",
            "    plt.axis('off')",
            "    plt.show()"
        ]),
        make_markdown_cell([
            "### Step 3: Load and Display Confusion Matrices"
        ]),
        make_code_cell([
            "fig, axes = plt.subplots(1, 3, figsize=(18, 6))",
            "models = ['custom_cnn', 'mobilenetv2_transfer', 'efficientnetb0_transfer']",
            "titles = ['Custom CNN', 'MobileNetV2', 'EfficientNetB0']",
            "",
            "for i, model in enumerate(models):",
            "    cm_path = Path(f'../static/images/model_comparison/cm_{model}.png')",
            "    if cm_path.exists():",
            "        img = plt.imread(str(cm_path))",
            "        axes[i].imshow(img)",
            "        axes[i].axis('off')",
            "        axes[i].set_title(titles[i], fontweight='bold')",
            "plt.tight_layout()",
            "plt.show()"
        ])
    ]
    
    # ----------------------------------------------------
    # Notebook 6: Grad-CAM Visualization
    # ----------------------------------------------------
    nb6_cells = [
        make_markdown_cell([
            "# Notebook 6: Explainable AI with Grad-CAM",
            "This notebook illustrates how Grad-CAM extracts gradients and superimposes",
            "colored heatmaps onto raw grayscale MRI inputs to make model classifications transparent."
        ]),
        make_code_cell([
            "import os",
            "from pathlib import Path",
            "import numpy as np",
            "import tensorflow as tf",
            "import cv2",
            "import matplotlib.pyplot as plt",
            "from PIL import Image",
            "",
            "sys.path.append('..')",
            "from ml.predict_helper import load_prediction_model, preprocess_single_image",
            "from ml.gradcam import find_last_conv_layer, make_gradcam_heatmap, save_and_display_gradcam",
            "",
            "MODELS_DIR = Path('../models')",
            "TEST_DIR = Path('../dataset/processed/test')"
        ]),
        make_markdown_cell([
            "### Step 1: Load Model & Test Image",
            "We load our MobileNetV2 model and select a pituitary tumor MRI."
        ]),
        make_code_cell([
            "model = load_prediction_model('MobileNetV2', MODELS_DIR)",
            "img_path = list((TEST_DIR / 'pituitary').glob('*'))[0]",
            "print('Selected Image:', img_path.name)",
            "",
            "img_array = preprocess_single_image(str(img_path))",
            "preds = model.predict(img_array)",
            "pred_idx = np.argmax(preds[0])",
            "print(f'Prediction index: {pred_idx} | Confidence: {preds[0][pred_idx]:.4f}')"
        ]),
        make_markdown_cell([
            "### Step 2: Generate Grad-CAM Heatmap",
            "We find the last conv layer dynamically and run our gradient calculator."
        ]),
        make_code_cell([
            "last_conv_layer = find_last_conv_layer(model)",
            "print('Last conv layer:', last_conv_layer)",
            "",
            "heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_idx)",
            "print('Heatmap shape:', heatmap.shape)",
            "",
            "# Plot raw heatmap",
            "plt.figure(figsize=(4, 4))",
            "plt.imshow(heatmap, cmap='viridis')",
            "plt.title('Raw Activation Heatmap')",
            "plt.colorbar()",
            "plt.show()"
        ]),
        make_markdown_cell([
            "### Step 3: Superimpose Heatmap onto Original Image"
        ]),
        make_code_cell([
            "output_path = 'gradcam_temp.jpg'",
            "save_and_display_gradcam(str(img_path), heatmap, output_path, alpha=0.4)",
            "",
            "# Display side-by-side",
            "fig, axes = plt.subplots(1, 2, figsize=(10, 5))",
            "orig_img = Image.open(img_path)",
            "axes[0].imshow(orig_img, cmap='gray')",
            "axes[0].set_title('Original MRI Scan', fontweight='bold')",
            "axes[0].axis('off')",
            "",
            "overlay_img = Image.open(output_path)",
            "axes[1].imshow(overlay_img)",
            "axes[1].set_title('Grad-CAM Overlay', fontweight='bold')",
            "axes[1].axis('off')",
            "",
            "plt.tight_layout()",
            "plt.show()",
            "",
            "# Cleanup temporary file",
            "if os.path.exists(output_path):",
            "    os.remove(output_path)"
        ])
    ]
    
    # ----------------------------------------------------
    # Notebook 7: Final Testing
    # ----------------------------------------------------
    nb7_cells = [
        make_markdown_cell([
            "# Notebook 7: Final End-to-End Prediction Pipeline Testing",
            "This notebook tests our single prediction pipeline end-to-end, loading models,",
            "classifying random MRI files, and plotting probability lists."
        ]),
        make_code_cell([
            "import os",
            "import random",
            "from pathlib import Path",
            "import matplotlib.pyplot as plt",
            "from PIL import Image",
            "",
            "sys.path.append('..')",
            "from ml.predict_helper import predict_brain_tumor",
            "",
            "TEST_DIR = Path('../dataset/processed/test')",
            "MODELS_DIR = Path('../models')",
            "OUTPUTS_DIR = Path('temp_outputs')"
        ]),
        make_markdown_cell([
            "### Step 1: Select a Random MRI from the Test Dataset"
        ]),
        make_code_cell([
            "classes = ['glioma', 'meningioma', 'notumor', 'pituitary']",
            "selected_class = random.choice(classes)",
            "class_dir = TEST_DIR / selected_class",
            "image_path = random.choice(list(class_dir.glob('*')))",
            "print(f'True Label: {selected_class.upper()} | File: {image_path.name}')"
        ]),
        make_markdown_cell([
            "### Step 2: Run End-to-End Prediction and Grad-CAM"
        ]),
        make_code_cell([
            "result = predict_brain_tumor(",
            "    image_path=str(image_path),",
            "    model_name='Best_Model',",
            "    models_dir=MODELS_DIR,",
            "    outputs_dir=OUTPUTS_DIR",
            ")",
            "",
            "print('\\nPrediction Details:')",
            "print(f'  Model Used      : {result[\"model_used\"]}')",
            "print(f'  Predicted Class : {result[\"predicted_class\"]}')",
            "print(f'  Confidence      : {result[\"confidence\"] * 100:.2f}%')",
            "print(f'  Processing Time : {result[\"processing_time\"]:.4f} seconds')",
            "print(f'  Probabilities   :')",
            "for cls, prob in result['probabilities'].items():",
            "    print(f'    - {cls:<15}: {prob * 100:.2f}%')"
        ]),
        make_markdown_cell([
            "### Step 3: Visualize Results Side-by-Side"
        ]),
        make_code_cell([
            "fig, axes = plt.subplots(1, 2, figsize=(10, 5))",
            "",
            "# Original",
            "axes[0].imshow(Image.open(image_path), cmap='gray')",
            "axes[0].set_title(f'Original Scan (True: {selected_class.upper()})', fontweight='bold')",
            "axes[0].axis('off')",
            "",
            "# Grad-CAM",
            "gradcam_img_path = OUTPUTS_DIR / f'gradcam_{image_path.name}'",
            "axes[1].imshow(Image.open(gradcam_img_path))",
            "axes[1].set_title(f'Predicted: {result[\"predicted_class\"]} ({result[\"confidence\"] * 100:.1f}%)', fontweight='bold')",
            "axes[1].axis('off')",
            "",
            "plt.tight_layout()",
            "plt.show()",
            "",
            "# Cleanup",
            "import shutil",
            "if OUTPUTS_DIR.exists():",
            "    shutil.rmtree(OUTPUTS_DIR)"
        ])
    ]
    
    # Write all notebooks to files
    all_nbs = [
        ("01_dataset_analysis.ipynb", nb1_cells),
        ("02_data_preprocessing.ipynb", nb2_cells),
        ("03_custom_cnn_training.ipynb", nb3_cells),
        ("04_transfer_learning.ipynb", nb4_cells),
        ("05_model_evaluation.ipynb", nb5_cells),
        ("06_gradcam_visualization.ipynb", nb6_cells),
        ("07_final_testing.ipynb", nb7_cells)
    ]
    
    for filename, cells in all_nbs:
        filepath = notebooks_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(create_notebook_json(cells), f, indent=2)
        print(f"Created notebook: {filename}")

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    notebooks_dir = project_dir / "notebooks"
    build_all_notebooks(notebooks_dir)
