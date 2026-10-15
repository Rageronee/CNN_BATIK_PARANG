
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import time
import sys

# Add the 'src' directory to the Python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
from data_loader import get_data_generators

# Automatically resolve BASE_DIR to the root directory (parent of src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_model_size(file_path):
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024) # Convert to MB

def evaluate_model(model_name, test_gen, model_dir, plots_dir):
    print(f"--- Evaluating {model_name} ---")
    model_path = os.path.join(model_dir, f"{model_name}_best.h5")
    
    if not os.path.exists(model_path):
        print(f"Model {model_name} not found at {model_path}. Skipping.")
        return None

    model = tf.keras.models.load_model(model_path)
    
    # Inference Latency Measurement
    start_time = time.time()
    predictions = model.predict(test_gen)
    end_time = time.time()
    latency_per_image = ((end_time - start_time) / test_gen.n) * 1000 # ms
    
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    model_size = get_model_size(model_path)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(plots_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

    print(f"Results for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Latency: {latency_per_image:.2f} ms/img")

    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'size_mb': model_size,
        'latency_ms': latency_per_image
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN Models for Batik Parang Classification")
    parser.add_argument('--model', type=str, choices=['resnet50', 'mobilenetv2', 'all'], default='all', help="Model to evaluate")
    parser.add_argument('--data_dir', type=str, default=BASE_DIR, help="Base directory of the dataset")
    parser.add_argument('--model_dir', type=str, default=os.path.join(BASE_DIR, 'models'), help="Directory where models are saved")
    parser.add_argument('--plots_dir', type=str, default=os.path.join(BASE_DIR, 'plots'), help="Directory to save evaluation plots")

    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)
    _, test_gen = get_data_generators(args.data_dir)
    
    results = []
    models_to_evaluate = ['mobilenetv2', 'resnet50'] if args.model == 'all' else [args.model]

    for m in models_to_evaluate:
        res = evaluate_model(m, test_gen, args.model_dir, args.plots_dir)
        if res:
            results.append(res)
    
    # Save comparison text
    with open(os.path.join(BASE_DIR, 'evaluation_results.txt'), 'w') as f:
        for r in results:
            f.write(str(r) + "\n")
