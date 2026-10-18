
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent

TRAIN_DIR = ROOT / "data" / "iris_images" / "train"
VALIDATION_DIR = ROOT / "data" / "iris_images" / "validation"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "iris_image_model.keras"
CLASSES_PATH = MODEL_DIR / "image_class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10


def main():
    if not TRAIN_DIR.exists() or not VALIDATION_DIR.exists():
        raise FileNotFoundError(
            "Check that data/iris_images contains train "
            "and validation folders."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training images...")

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )

    class_names = train_ds.class_names

    print("Classes:", class_names)

    print("Loading validation images...")

    validation_ds = keras.utils.image_dataset_from_directory(
        VALIDATION_DIR,
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    train_ds = train_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )
    validation_ds = validation_ds.prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    # Normalize pixel values inside the model.
    # This matches image_classifier.py, which passes pixels
    # in the 0-255 range during prediction.
    model = keras.Sequential([
        keras.Input(shape=(224, 224, 3)),
        layers.Rescaling(1.0 / 255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),

        layers.Conv2D(
            32, 3, activation="relu", padding="same"
        ),
        layers.MaxPooling2D(),

        layers.Conv2D(
            64, 3, activation="relu", padding="same"
        ),
        layers.MaxPooling2D(),

        layers.Conv2D(
            128, 3, activation="relu", padding="same"
        ),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training image model...")

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=EPOCHS,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
        ],
    )

    print("Saving trained model...")

    model.save(MODEL_PATH)

    with open(CLASSES_PATH, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=4)

    print("\nTraining completed!")
    print("Model saved:", MODEL_PATH)
    print("Class names saved:", CLASSES_PATH)
    print("Classes:", class_names)


if __name__ == "__main__":
    main()
