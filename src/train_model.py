
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os
import time
import sys

# Add the 'src' directory to the Python path to allow absolute imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import get_data_generators

import argparse

# Automatically resolve BASE_DIR to the root directory (parent of src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_model(model_name, num_classes):
    base_model = None
    if model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'mobilenetv2':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Invalid model name. Choose 'resnet50' or 'mobilenetv2'.")

    # Freeze base model layers
    base_model.trainable = False

    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train(model_name, epochs, learning_rate, data_dir):
    print(f"--- Starting training for {model_name} ---")
    train_gen, test_gen = get_data_generators(data_dir)
    num_classes = len(train_gen.class_indices)

    model = build_model(model_name, num_classes)
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training finished in {training_time:.2f} seconds.")

    # Save model
    save_path = os.path.join(BASE_DIR, 'models')
    os.makedirs(save_path, exist_ok=True)
    model.save(os.path.join(save_path, f"{model_name}_best.h5"))
    print(f"Model saved to {save_path}")

    # Save training time for efficiency metric
    with open(os.path.join(save_path, f"{model_name}_time.txt"), "w") as f:
        f.write(str(training_time))

    # Plot and save training history
    import matplotlib.pyplot as plt
    plots_dir = os.path.join(BASE_DIR, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'learning_curve_{model_name}.png'))
    plt.close()

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Models for Batik Parang Classification")
    parser.add_argument('--model', type=str, choices=['resnet50', 'mobilenetv2', 'all'], default='all', help="Model to train")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--data_dir', type=str, default=BASE_DIR, help="Base directory of the dataset")

    args = parser.parse_args()

    if args.model == 'all':
        train('mobilenetv2', args.epochs, args.lr, args.data_dir)
        train('resnet50', args.epochs, args.lr, args.data_dir)
    else:
        train(args.model, args.epochs, args.lr, args.data_dir)
