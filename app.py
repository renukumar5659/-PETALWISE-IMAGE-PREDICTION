from pathlib import Path
import json
import uuid
import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from image_classifier import predict_image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "metrics.json"
UPLOAD_FOLDER = ROOT / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    return joblib.load(MODEL_PATH)

def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}

@app.get("/")
def index():
    metrics = load_metrics()
    return render_template("index.html", metrics=metrics, result=None, error=None)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_PATH.exists(), "service": "PetalWise Iris API"})

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    metrics = load_metrics()
    if request.method == "GET":
        return jsonify({
            "endpoint": "/prediction",
            "method": "POST",
            "required_fields": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "example": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        })
    try:
        data = request.get_json(silent=True) if request.is_json else request.form
        values = [float(data[k]) for k in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        if not all(np.isfinite(values)):
            raise ValueError("Measurements must be finite numbers.")
        limits = [(0.1, 10), (0.1, 10), (0.1, 10), (0.1, 10)]
        for value, (low, high) in zip(values, limits):
            if not low <= value <= high:
                raise ValueError(f"Measurements must be between {low} and {high} cm.")
        model = load_model()
        pred_id = int(model.predict([values])[0])
        names = ["Iris setosa", "Iris versicolor", "Iris virginica"]
        probabilities = model.predict_proba([values])[0].tolist() if hasattr(model, "predict_proba") else []
        result = {"prediction": names[pred_id], "class_id": pred_id, "probabilities": dict(zip(names, probabilities)), "model": metrics.get("selected_model", "trained classifier")}
        if request.is_json:
            return jsonify(result)
        return render_template("index.html", metrics=metrics, result=result, error=None, values=values)
    except (KeyError, TypeError, ValueError) as exc:
        message = f"Invalid input: {exc}"
        if request.is_json:
            return jsonify({"error": message}), 400
        return render_template("index.html", metrics=metrics, result=None, error=message), 400
    except FileNotFoundError as exc:
        if request.is_json:
            return jsonify({"error": str(exc)}), 500
        return render_template("index.html", metrics=metrics, result=None, error=str(exc)), 500

@app.route("/image-prediction", methods=["GET", "POST"])
def image_prediction():
    if request.method == "GET":
        return render_template("image_prediction.html", result=None, image_url=None, error=None)

    image_url = None
    try:
        if "flower_image" not in request.files:
            error_msg = "Please select an image file to upload."
            if request.is_json:
                return jsonify({"error": error_msg}), 400
            return render_template("image_prediction.html", result=None, image_url=None, error=error_msg), 400

        file = request.files["flower_image"]
        if not file or not file.filename or file.filename.strip() == "":
            error_msg = "No image selected. Please choose a flower photo."
            if request.is_json:
                return jsonify({"error": error_msg}), 400
            return render_template("image_prediction.html", result=None, image_url=None, error=error_msg), 400

        if not allowed_file(file.filename):
            error_msg = "Unsupported file type. Please upload a JPG, JPEG, PNG, or WEBP image."
            if request.is_json:
                return jsonify({"error": error_msg}), 400
            return render_template("image_prediction.html", result=None, image_url=None, error=error_msg), 400

        # Validate image integrity using Pillow
        try:
            with Image.open(file.stream) as img:
                img.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            error_msg = "The uploaded file is corrupt or not a valid image."
            if request.is_json:
                return jsonify({"error": error_msg}), 400
            return render_template("image_prediction.html", result=None, image_url=None, error=error_msg), 400

        # Rewind stream and save sanitized JPEG
        file.stream.seek(0)
        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = UPLOAD_FOLDER / filename

        with Image.open(file.stream) as img:
            img.convert("RGB").save(save_path, format="JPEG")

        image_url = url_for("static", filename=f"uploads/{filename}")

        # Run inference using the trained image model
        result = predict_image(str(save_path))

        if request.is_json:
            return jsonify({"result": result, "image_url": image_url})

        return render_template("image_prediction.html", result=result, image_url=image_url, error=None)

    except ValueError as exc:
        error_msg = f"Invalid image: {exc}"
        if request.is_json:
            return jsonify({"error": error_msg}), 400
        return render_template("image_prediction.html", result=None, image_url=image_url, error=error_msg), 400
    except FileNotFoundError as exc:
        error_msg = "Trained image classification model is not available."
        if request.is_json:
            return jsonify({"error": error_msg}), 500
        return render_template("image_prediction.html", result=None, image_url=image_url, error=error_msg), 500
    except Exception:
        app.logger.exception("Unexpected error in image prediction")
        error_msg = "An unexpected error occurred while processing the flower image."
        if request.is_json:
            return jsonify({"error": error_msg}), 500
        return render_template("image_prediction.html", result=None, image_url=image_url, error=error_msg), 500


@app.errorhandler(413)
def file_too_large(error):
    error_msg = "Image file is too large. Maximum allowed size is 10 MB."
    if request.is_json:
        return jsonify({"error": error_msg}), 413
    return render_template("image_prediction.html", result=None, image_url=None, error=error_msg), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

