import io
import json
from pathlib import Path
import subprocess
import sys
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def ensure_models():
    """Ensure measurement model exists before running tests."""
    model_path = ROOT / "models" / "model.pkl"
    if not model_path.exists():
        subprocess.run([sys.executable, str(ROOT / "train_model.py")], check=True, cwd=ROOT)


@pytest.fixture()
def client():
    sys.path.insert(0, str(ROOT))
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def create_test_image_bytes(fmt="JPEG", color="violet"):
    """Helper to generate an in-memory test image."""
    image = Image.new("RGB", (64, 64), color=color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    buffer.seek(0)
    return buffer


# ==========================================
# 1. Existing Measurement Tests & Core API
# ==========================================

def test_home_page(client):
    """Verify homepage loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"PetalWise" in response.data


def test_health(client):
    """Verify health endpoint status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_prediction_json(client):
    """Verify JSON measurement prediction."""
    response = client.post("/prediction", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    })
    assert response.status_code == 200
    assert response.get_json()["prediction"] == "Iris setosa"


def test_prediction_form(client):
    """Verify HTML form measurement prediction."""
    response = client.post("/prediction", data={
        "sepal_length": "6.2",
        "sepal_width": "3.4",
        "petal_length": "5.4",
        "petal_width": "2.3",
    })
    assert response.status_code == 200
    assert b"Predicted Species" in response.data
    assert b"Iris virginica" in response.data


def test_invalid_prediction(client):
    """Verify validation error handling for tabular measurements."""
    response = client.post("/prediction", json={"sepal_length": "bad"})
    assert response.status_code == 400
    assert "Invalid input" in response.get_json()["error"]


# ==========================================
# 2. Image Prediction Tests
# ==========================================

def test_image_prediction_get(client):
    """Verify GET /image-prediction returns HTTP 200 and renders upload form."""
    response = client.get("/image-prediction")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Upload Flower Image" in html
    assert not html.startswith("```")


def test_image_prediction_valid_upload(client):
    """Verify uploading a valid image processes and displays prediction."""
    img_buf = create_test_image_bytes("JPEG", color="purple")
    response = client.post(
        "/image-prediction",
        data={"flower_image": (img_buf, "sample_flower.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "PREDICTED SPECIES" in html
    assert any(species in html for species in ["Iris setosa", "Iris versicolor", "Iris virginica"])


def test_image_prediction_missing_file(client):
    """Verify missing flower_image field produces a friendly 400 error."""
    response = client.post("/image-prediction", data={})
    assert response.status_code == 400
    assert "Please select an image file" in response.data.decode("utf-8")


def test_image_prediction_empty_filename(client):
    """Verify empty filename submission produces a 400 error."""
    empty_buf = io.BytesIO(b"")
    response = client.post(
        "/image-prediction",
        data={"flower_image": (empty_buf, "")},
    )
    assert response.status_code == 400
    assert "No image selected" in response.data.decode("utf-8")


def test_image_prediction_unsupported_file_type(client):
    """Verify unsupported file extensions are rejected."""
    txt_buf = io.BytesIO(b"Hello Iris")
    response = client.post(
        "/image-prediction",
        data={"flower_image": (txt_buf, "notes.txt")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.data.decode("utf-8")


def test_image_prediction_corrupt_image(client):
    """Verify corrupt images with image extensions are rejected."""
    corrupt_buf = io.BytesIO(b"Fake header content not really an image")
    response = client.post(
        "/image-prediction",
        data={"flower_image": (corrupt_buf, "corrupt.jpg")},
    )
    assert response.status_code == 400
    assert "corrupt or not a valid image" in response.data.decode("utf-8")


def test_image_prediction_oversized_file(client):
    """Verify oversized images triggering 413 are handled gracefully."""
    from werkzeug.exceptions import RequestEntityTooLarge
    with client.application.test_request_context():
        res = client.application.handle_user_exception(RequestEntityTooLarge())
        response = client.application.make_response(res)
        assert response.status_code == 413
        assert "Image file is too large" in response.data.decode("utf-8")



def test_image_prediction_classifier_error(client, monkeypatch):
    """Verify unexpected classifier errors produce safe response without crashing."""
    def broken_predict(path):
        raise ValueError("Inference tensor dimensions mismatched")

    monkeypatch.setattr("app.predict_image", broken_predict)
    img_buf = create_test_image_bytes("PNG", color="green")
    response = client.post(
        "/image-prediction",
        data={"flower_image": (img_buf, "flower.png")},
    )
    assert response.status_code == 400
    assert "Invalid image: Inference tensor dimensions mismatched" in response.data.decode("utf-8")
