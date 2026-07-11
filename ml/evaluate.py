import os
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from train import get_data_generators

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def calculate_metrics(y_true, y_pred, num_classes):
    """
    Computes confusion matrix, precision, recall, and F1-score using NumPy.
    Returns:
        confusion_matrix: np.ndarray
        metrics: dict containing class-wise and macro average precision, recall, and f1
    """
    # 1. Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    class_metrics = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[CLASSES[i]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }
        
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        
    macro_precision /= num_classes
    macro_recall /= num_classes
    macro_f1 /= num_classes
    
    overall_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    
    return cm, {
        "accuracy": round(overall_accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "class_wise": class_metrics
    }

def plot_confusion_matrix(cm, classes, title, save_path):
    """Plot confusion matrix and save as image."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Labeling cells
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12, fontweight='bold')

    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def plot_metrics_comparison(comparison_data, save_path):
    """Plot bar chart comparing model metrics."""
    models = list(comparison_data.keys())
    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    x = np.arange(len(models))
    width = 0.2
    
    plt.figure(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):
        values = [comparison_data[model][metric] for model in models]
        plt.bar(x + i * width, values, width, label=metric_labels[i])
        
    plt.title("Model Performance Comparison", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(x + 1.5 * width, models, fontsize=12)
    plt.ylim(0, 1.1)
    plt.ylabel("Score", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def evaluate_all():
    project_dir = Path(__file__).resolve().parent.parent
    processed_dir = project_dir / "dataset" / "processed"
    models_dir = project_dir / "models"
    comparison_img_dir = project_dir / "static" / "images" / "model_comparison"
    comparison_img_dir.mkdir(parents=True, exist_ok=True)
    
    _, _, test_gen = get_data_generators(processed_dir)
    num_classes = test_gen.num_classes
    y_true = test_gen.classes
    
    model_files = {
        "Custom_CNN": "best_custom_cnn.keras",
        "MobileNetV2": "best_mobilenetv2_transfer.keras",
        "EfficientNetB0": "best_efficientnetb0_transfer.keras"
    }
    
    comparison_results = {}
    
    for model_name, filename in model_files.items():
        model_path = models_dir / filename
        if not model_path.exists():
            print(f"Model file not found: {model_path.resolve()}. Skipping...")
            continue
            
        print(f"\nEvaluating Model: {model_name}...")
        model = tf.keras.models.load_model(str(model_path))
        
        # Run prediction
        test_gen.reset()
        preds = model.predict(test_gen)
        y_pred = np.argmax(preds, axis=1)
        
        # Calculate metrics
        cm, metrics = calculate_metrics(y_true, y_pred, num_classes)
        comparison_results[model_name] = metrics
        
        # Plot and save CM
        cm_path = comparison_img_dir / f"cm_{model_name.lower()}.png"
        plot_confusion_matrix(cm, CLASSES, f"Confusion Matrix - {model_name}", cm_path)
        print(f"Saved confusion matrix for {model_name} to {cm_path.name}")
        
    if comparison_results:
        # Save comparison results JSON
        comparison_path = models_dir / "model_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison_results, f, indent=4)
        print(f"\nSaved performance comparison JSON to {comparison_path.resolve()}")
        
        # Plot comparison metrics
        comparison_chart_path = comparison_img_dir / "metrics_comparison.png"
        plot_metrics_comparison(comparison_results, comparison_chart_path)
        print(f"Saved metrics comparison chart to {comparison_chart_path.name}")
        
        # Print Markdown Table Comparison
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE COMPARISON TABLE")
        print("=" * 80)
        print(f"{'Model Name':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("-" * 71)
        for m_name, m_val in comparison_results.items():
            print(f"{m_name:<20} | {m_val['accuracy']:<10.4f} | {m_val['macro_precision']:<10.4f} | {m_val['macro_recall']:<10.4f} | {m_val['macro_f1']:<10.4f}")
        print("=" * 80)

if __name__ == "__main__":
    evaluate_all()
