import json
from pathlib import Path
from typing import Union, BinaryIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "iris_image_model.keras"
CLASSES_PATH = ROOT / "models" / "image_class_names.json"

_model = None
_class_names = None

CLASS_DISPLAY_NAMES = {
    "iris_setosa": "Iris setosa",
    "iris_versicolor": "Iris versicolor",
    "iris_virginica": "Iris virginica",
    "setosa": "Iris setosa",
    "versicolor": "Iris versicolor",
    "virginica": "Iris virginica",
}


def load_image_model():
    """Load and cache the trained Keras model and class names.
    
    Returns:
        tuple: (model, class_names)
    Raises:
        FileNotFoundError: If model or class names file is missing.
    """
    global _model, _class_names

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Image model file not found at: {MODEL_PATH}")
        if not CLASSES_PATH.exists():
            raise FileNotFoundError(f"Class names file not found at: {CLASSES_PATH}")

        _model = keras.models.load_model(MODEL_PATH)

        with open(CLASSES_PATH, "r", encoding="utf-8") as file:
            _class_names = json.load(file)

    return _model, _class_names


def format_species_name(raw_name: str) -> str:
    """Format raw class name into a clean user-facing title."""
    cleaned = raw_name.strip()
    if cleaned in CLASS_DISPLAY_NAMES:
        return CLASS_DISPLAY_NAMES[cleaned]
    return cleaned.replace("_", " ").title()


def get_expected_image_size(model) -> tuple[int, int]:
    """Retrieve expected (width, height) from model input shape."""
    shape = getattr(model, "input_shape", None)
    if shape and len(shape) == 4 and shape[1] is not None and shape[2] is not None:
        return (int(shape[2]), int(shape[1]))
    return (224, 224)


def predict_image(image_input: Union[str, Path, BinaryIO, Image.Image]) -> dict:
    """Classify an Iris flower image using the trained deep learning model.

    Args:
        image_input: File path, file-like binary stream, or PIL Image.

    Returns:
        dict: Prediction results including species name, confidence score,
              and probability breakdown.
    Raises:
        FileNotFoundError: If model files cannot be found.
        ValueError: If the input cannot be decoded as a valid image.
    """
    model, class_names = load_image_model()
    target_size = get_expected_image_size(model)

    try:
        if isinstance(image_input, Image.Image):
            pil_image = image_input.copy()
        else:
            pil_image = Image.open(image_input)

        # Ensure correct EXIF orientation and RGB channels
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        pil_image = pil_image.resize(target_size)

        # Model expects batch dimension and float32 in [0, 255] range
        array = np.asarray(pil_image, dtype=np.float32)
        array = np.expand_dims(array, axis=0)

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid or unreadable image file: {exc}") from exc

    # Run inference
    predictions = model.predict(array, verbose=0)[0]
    best_index = int(np.argmax(predictions))
    best_raw = class_names[best_index]
    best_confidence = float(predictions[best_index]) * 100.0

    probabilities = [
        {
            "raw_class": name,
            "species": format_species_name(name),
            "confidence": round(float(prob) * 100.0, 2),
        }
        for name, prob in zip(class_names, predictions)
    ]

    return {
        "raw_class": best_raw,
        "species": format_species_name(best_raw),
        "confidence": round(best_confidence, 2),
        "probabilities": probabilities,
    }
