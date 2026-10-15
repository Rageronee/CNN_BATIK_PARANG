
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def get_data_generators(data_dir):
    """
    Creates training and validation/testing data generators.
    Assumes structure:
    parang_1/
        citra latih/ (Training)
            ClassA/
            ClassB/
        citra uji/ (Testing)
            ClassA/
            ClassB/
    """
    # Point to 'citra latih' for both, using split
    base_dir = os.path.join(data_dir, 'parang_1', 'citra latih')
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    # Data Augmentation for Training
    # Note: We use the same seed to ensuring splitting consistency if possible, 
    # but since we are using two different IDG instances, strict split identity isn't guaranteed 
    # without identical args. 
    # However, standard practice with Keras IDG split is usually safe enough for this scale.
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Use 20% for validation/testing
    )

    # Validation Generator (No Augmentation, just rescale)
    # We must use the same validation_split argument to ensure indices align conceptually
    # But to be perfectly safe against leakage/mismatch, we will accept Training with Augmentation 
    # and Validation with Augmentation (Train-Time Val) OR No Augmentation (but potentially different split).
    #
    # FOR ROBUSTNESS: We will use the SAME generator config for split, then wrap it?
    # No, let's stick to the separation. 
    # To minimize mismatch risk: use `rescale` only for val, but keep `validation_split`.
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    print(f"Loading Training Data (80%) from: {base_dir}")
    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training', # Set as training data
        seed=42,
        shuffle=True
    )

    print(f"Loading Testing/Validation Data (20%) from: {base_dir}")
    test_generator = test_datagen.flow_from_directory(
        base_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation', # Set as validation data
        seed=42,
        shuffle=False
    )

    return train_generator, test_generator

if __name__ == "__main__":
    # Test the loader
    # Using the parent directory of 'src' as the base path
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        train_gen, test_gen = get_data_generators(base_path)
        print("Class indices:", train_gen.class_indices)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please ensure the dataset is placed at: {os.path.join(base_path, 'parang_1', 'citra latih')}")
