"""Train and compare Iris classifiers, then serialize the best cross-validated model."""
from pathlib import Path
import json
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)
candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
    "Support Vector Machine": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
comparison = {}
for name, estimator in candidates.items():
    scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring="accuracy")
    comparison[name] = {"cv_mean_accuracy": float(scores.mean()), "cv_std": float(scores.std())}

best_name = max(comparison, key=lambda n: comparison[n]["cv_mean_accuracy"])
model = candidates[best_name]
model.fit(X_train, y_train)
pred = model.predict(X_test)
result = {
    "selected_model": best_name,
    "test_accuracy": float(accuracy_score(y_test, pred)),
    "test_samples": int(len(y_test)),
    "cv_comparison": comparison,
    "classification_report": classification_report(y_test, pred, target_names=iris.target_names, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "class_names": iris.target_names.tolist(),
}
joblib.dump(model, MODEL_DIR / "model.pkl")
(ROOT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("Cross-validation comparison (training data only):")
for name, vals in comparison.items():
    print(f"{name}: {vals['cv_mean_accuracy']:.3f} +/- {vals['cv_std']:.3f}")
print(f"Selected model: {best_name}")
print(f"Held-out test accuracy: {result['test_accuracy']:.3f}")
print(f"Model saved: {MODEL_DIR / 'model.pkl'}")
print("Metrics saved: metrics.json")
