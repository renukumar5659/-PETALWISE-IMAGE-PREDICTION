# 🌸 PetalWise — MLOps Iris Prediction System

PetalWise is an educational machine learning and MLOps web application designed for Iris flower species classification. The project provides both classical measurement-based tabular prediction and deep learning-based image classification wrapped in a Flask web application and RESTful API, structured for containerization and cloud deployment practicals.

---

## 🌟 Key Features

- **Flower Measurement Classification:** Classifies an Iris flower into *Setosa*, *Versicolor*, or *Virginica* based on 4 botanical measurements (sepal length, sepal width, petal length, petal width).
- **Computer Vision Image Classification:** Upload and classify Iris flower photos via an end-to-end deep convolutional neural network built with TensorFlow and Keras.
- **Flask Web Interface & REST API:**
  - Modern responsive UI (`/`) with SVG flower artwork and dynamic probability meters.
  - Dedicated image classifier interface (`/image-prediction`) supporting JPG, JPEG, PNG, and WEBP formats with server-side Pillow verification.
  - RESTful JSON prediction endpoints (`/prediction` and `/image-prediction`) for programmatic integration.
  - Service health check endpoint (`/health`).
- **Model Evaluation & Comparison:** 5-fold stratified cross-validation comparing Logistic Regression, Random Forest, and Support Vector Machine (SVM), saving metrics and confusion matrices to `metrics.json`.
- **Automated Test Suite:** Comprehensive pytest test suite (`tests/test_app.py`) verifying API endpoints, tabular predictions, image uploads, file validations, and edge cases.
- **Containerization & Deployment Ready:** Dockerfile packaging with Gunicorn WSGI server, ready for local containers or cloud virtual machines (AWS EC2 / GCP Compute Engine).

---

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.10 / 3.11 | Core runtime |
| **Web Framework** | Flask | >= 3.0, < 4.0 | WSGI Web application and REST API |
| **WSGI Server** | Gunicorn | >= 22.0, < 24.0 | Production HTTP server |
| **Classical ML** | Scikit-Learn | >= 1.4, < 2.0 | Model training, cross-validation, pipelines |
| **Deep Learning** | TensorFlow / Keras | >= 2.15, < 3.0 | Convolutional neural network image classifier |
| **Image Processing** | Pillow (PIL) | >= 10.0, < 14.0 | Image decoding, EXIF rotation, format normalization |
| **Numerical Math** | NumPy | >= 1.26, < 3.0 | Tensor manipulation and probability math |
| **Model Serialization**| Joblib | >= 1.3, < 2.0 | Serialization for Scikit-Learn pipelines |
| **Automated Testing** | Pytest | >= 8.0, < 9.0 | Unit and integration testing |
| **Frontend** | HTML5, CSS3, SVG | Vanilla | Responsive user dashboard |

---

## 📁 Project Directory Structure

```text
PetalWise_MLOps_Practical_Complete/
│
├── .dockerignore                       # Files excluded from Docker builds
├── .gitignore                          # Git ignore rules for caches, venvs, and uploads
├── Dockerfile                          # Multi-stage production container specification
├── README.md                           # Project documentation and user guide
├── requirements.txt                    # Project dependencies
├── app.py                              # Main Flask web application and API routes
├── image_classifier.py                 # Keras image classification helper module
├── metrics.json                        # Cross-validation and model evaluation metrics
├── train_model.py                      # Tabular model training and model selection script
├── train_image_model.py                # Deep learning image classifier training script
│
├── data/
│   └── iris_images/                    # Dataset of Iris flower photos
│       ├── train/                      # Training split (15 images: 5 per class)
│       │   ├── iris_setosa/
│       │   ├── iris_versicolor/
│       │   └── iris_virginica/
│       └── validation/                 # Validation split (15 images: 5 per class)
│           ├── iris_setosa/
│           ├── iris_versicolor/
│           └── iris_virginica/
│
├── models/
│   ├── model.pkl                       # Serialized scikit-learn Support Vector Machine
│   ├── iris_image_model.keras          # Trained Keras CNN model weights
│   └── image_class_names.json          # Target species class labels
│
├── practicals/
│   └── STEP_BY_STEP_GUIDE.md           # Step-by-step practical lab walkthrough
│
├── static/
│   ├── css/
│   │   └── style.css                   # Custom responsive stylesheet
│   ├── images/                         # Vector artwork (setosa.svg, versicolor.svg, virginica.svg)
│   └── uploads/                        # Temporary directory for uploaded flower images
│
├── templates/
│   ├── index.html                      # Homepage with measurement predictor & metrics
│   └── image_prediction.html           # Dedicated flower image upload and results template
│
└── tests/
    └── test_app.py                     # Automated pytest suite (13 test cases)
```

---

## ⚙️ Prerequisites and Installation (Windows PowerShell)

### 1. Prerequisites
- **Python:** Python 3.10 or 3.11 installed. (Verify with `python --version`).
- **PowerShell:** Standard Windows PowerShell terminal.

### 2. Setup Virtual Environment
Open PowerShell in the project directory:

```powershell
cd C:\Users\DELL\Downloads\PetalWise_MLOps_Practical\PetalWise_MLOps_Practical_Complete

# Create virtual environment (if not already created)
python -m venv .venv

# Upgrade pip
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Install required dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## 🚀 Running the Application

Start the Flask development server:

```powershell
.\.venv\Scripts\python.exe app.py
```

Once started, access the application in your browser:
- **Home & Tabular Predictor:** [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **Image Classifier Page:** [http://127.0.0.1:5000/image-prediction](http://127.0.0.1:5000/image-prediction)
- **API Health Endpoint:** [http://127.0.0.1:5000/health](http://127.0.0.1:5000/health)

Press `CTRL+C` in the terminal to stop the application.

---

## 🧪 Testing and Verification

Run the automated test suite with pytest:

```powershell
.\.venv\Scripts\python.exe -m pytest -v tests\test_app.py
```

The test suite validates:
1. `test_home_page`: Homepage rendering and title verification.
2. `test_health`: Health check endpoint status and model loading.
3. `test_prediction_json`: Measurement prediction via JSON payload.
4. `test_prediction_form`: Measurement prediction via HTML form submission.
5. `test_invalid_prediction`: Input range and numeric validation (0.1 cm - 10.0 cm).
6. `test_image_prediction_get`: Image upload page rendering without syntax defects.
7. `test_image_prediction_valid_upload`: Image classification pipeline and probability calculation.
8. `test_image_prediction_missing_file`: Graceful error response when no file is provided.
9. `test_image_prediction_empty_filename`: Rejection of empty file submissions.
10. `test_image_prediction_unsupported_file_type`: Restriction to `.jpg`, `.jpeg`, `.png`, and `.webp`.
11. `test_image_prediction_corrupt_image`: Pillow byte validation rejecting corrupt files.
12. `test_image_prediction_oversized_file`: HTTP 413 handling for uploads exceeding 10 MB.
13. `test_image_prediction_classifier_error`: Exception handling and error reporting.

---

## 📡 REST API Reference

### 1. Health Check
- **Endpoint:** `GET /health`
- **Response:**
  ```json
  {
    "model_loaded": true,
    "service": "PetalWise Iris API",
    "status": "ok"
  }
  ```

### 2. Tabular Prediction API
- **Endpoint:** `POST /prediction`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
  ```
- **Response:**
  ```json
  {
    "class_id": 0,
    "model": "Support Vector Machine",
    "prediction": "Iris setosa",
    "probabilities": {
      "Iris setosa": 0.978,
      "Iris versicolor": 0.014,
      "Iris virginica": 0.008
    }
  }
  ```

### 3. Image Prediction API
- **Endpoint:** `POST /image-prediction`
- **Headers:** `Content-Type: multipart/form-data`, `Accept: application/json`
- **Form Data:**
  - `flower_image`: Binary image file (JPG, PNG, WEBP)
- **Response:**
  ```json
  {
    "image_url": "/static/uploads/7f8a9b...jpg",
    "result": {
      "confidence": 63.9,
      "raw_class": "iris_setosa",
      "species": "Iris setosa",
      "probabilities": [
        {"species": "Iris setosa", "confidence": 63.9, "raw_class": "iris_setosa"},
        {"species": "Iris versicolor", "confidence": 23.51, "raw_class": "iris_versicolor"},
        {"species": "Iris virginica", "confidence": 12.59, "raw_class": "iris_virginica"}
      ]
    }
  }
  ```

---

## 🧠 Model Training and Retraining

### 1. Tabular Model Training (`train_model.py`)
Trains and compares candidate classifiers on the Scikit-learn Iris dataset using 5-fold stratified cross-validation:
```powershell
.\.venv\Scripts\python.exe train_model.py
```
- **Candidates Evaluated:** Logistic Regression, Random Forest, Support Vector Machine (SVC).
- **Selection:** Automatically selects the model with the highest cross-validation score and saves it to `models/model.pkl`.
- **Output:** Writes evaluation metrics and confusion matrices to `metrics.json`.

### 2. Image Model Training (`train_image_model.py`)
Trains a Convolutional Neural Network on the image dataset:
```powershell
.\.venv\Scripts\python.exe train_image_model.py
```
- **Input Dimensions:** `(224, 224, 3)`
- **Data Augmentation:** Horizontal flips, random rotations (0.08).
- **Architecture:** Sequential Conv2D layers, MaxPooling, GlobalAveragePooling, Dense layer with Softmax.
- **Artifacts Saved:** `models/iris_image_model.keras` and `models/image_class_names.json`.

---

## 🐳 Docker Deployment

To build and run the application in a Docker container:

```bash
# Build Docker image
docker build -t petalwise-iris:1.0 .

# Run container
docker run -d --name petalwise-iris-app -p 5000:5000 petalwise-iris:1.0

# Verify logs
docker logs petalwise-iris-app
```

Navigate to `http://localhost:5000` to verify the running application.

---

## ⚠️ Known Limitations

1. **Educational Dataset Size:**  
   The image dataset in `data/iris_images/` contains 30 sample images (15 training, 15 validation; 5 per species). While the deep learning pipeline, data augmentation, and inference workflows are fully functional, high classification accuracy across all angles and lighting requires a larger production dataset.
2. **Local File Storage:**  
   Uploaded user images are saved to `static/uploads/` with UUIDs. In a high-traffic production system, images should be stored in cloud object storage (e.g. AWS S3) with automated lifecycle cleanup policies.

---

## 🔮 Future Improvements

- [ ] Integrate cloud storage (AWS S3 / Google Cloud Storage) for uploaded images.
- [ ] Implement user authentication and prediction history tracking.
- [ ] Add DVC (Data Version Control) for dataset tracking.
- [ ] Configure CI/CD automated pipeline via GitHub Actions.
- [ ] Add drift detection and inference latency monitoring using Prometheus or Evidently AI.

---

## 📄 License & Acknowledgements

This project was developed as part of an MLOps practical curriculum adapting machine learning workflows into containerized cloud-ready applications.
