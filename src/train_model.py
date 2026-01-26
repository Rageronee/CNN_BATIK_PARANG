
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os
import time
from data_loader import get_data_generators

# Configuration
EPOCHS = 20 # Can be adjusted
LEARNING_RATE = 0.0001
BASE_DIR = r"d:\Data Main\Portfolio_afnan\CNN_Batik_Parang"

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

def train(model_name):
    print(f"--- Starting training for {model_name} ---")
    train_gen, test_gen = get_data_generators(BASE_DIR)
    num_classes = len(train_gen.class_indices)

    model = build_model(model_name, num_classes)
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    start_time = time.time()
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
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

    return history

if __name__ == "__main__":
    # Train both models
    train('mobilenetv2')
    train('resnet50')
