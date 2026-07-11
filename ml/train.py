import os
import json
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Random seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)

# Hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 3

def get_data_generators(processed_dir: Path):
    """Create data generators for train, validation, and test splits."""
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "validation"
    test_dir = processed_dir / "test"
    
    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    
    train_gen = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED
    )
    
    val_gen = val_datagen.flow_from_directory(
        str(val_dir),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    
    test_gen = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    
    return train_gen, val_gen, test_gen

def build_custom_cnn(num_classes):
    """Build Custom CNN architecture."""
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ], name="Custom_CNN")
    return model

def build_mobilenetv2(num_classes):
    """Build MobileNetV2 Transfer Learning model."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base layers
    
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ], name="MobileNetV2_Transfer")
    return model

def build_efficientnetb0(num_classes):
    """Build EfficientNetB0 Transfer Learning model."""
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base layers
    
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ], name="EfficientNetB0_Transfer")
    return model

def train_model(model, train_gen, val_gen, models_dir: Path):
    """Compile and train a model, saving the best version."""
    print("\n" + "=" * 50)
    print(f"TRAINING MODEL: {model.name}")
    print("=" * 50)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    best_model_path = models_dir / f"best_{model.name.lower()}.keras"
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save the final model just in case
    final_model_path = models_dir / f"{model.name.lower()}_final.keras"
    model.save(str(final_model_path))
    print(f"Model {model.name} trained. Best saved to {best_model_path.name}")
    
    # Save history to JSON for later plotting/comparison
    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    history_path = models_dir / f"history_{model.name.lower()}.json"
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=4)
        
    return history

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    processed_dir = project_dir / "dataset" / "processed"
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    train_gen, val_gen, test_gen = get_data_generators(processed_dir)
    num_classes = train_gen.num_classes
    
    # Save class indices mapping
    class_mapping_path = models_dir / "class_indices.json"
    with open(class_mapping_path, "w") as f:
        json.dump(train_gen.class_indices, f, indent=4)
    print("Saved class indices mapping to class_indices.json")
    
    # 1. Custom CNN
    custom_cnn = build_custom_cnn(num_classes)
    train_model(custom_cnn, train_gen, val_gen, models_dir)
    
    # 2. MobileNetV2
    mobilenet = build_mobilenetv2(num_classes)
    train_model(mobilenet, train_gen, val_gen, models_dir)
    
    # 3. EfficientNetB0
    efficientnet = build_efficientnetb0(num_classes)
    train_model(efficientnet, train_gen, val_gen, models_dir)
    
    print("\nAll models trained and saved successfully!")
