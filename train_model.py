"""Train and compare three classification algorithms."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_data, FEATURES

MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

df, X, y = load_data()

numeric_features = [
    "age", "studytime", "failures", "absences", "G1", "G2",
    "health", "goout", "freetime"
]
categorical_features = ["schoolsup", "famsup", "paid", "activities", "internet"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced"
    )
}

results = {}
best_name = None
best_score = -1
best_pipeline = None

for name, classifier in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("model", classifier)])
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_test, predictions,
            labels=["Needs Improvement", "Average Performance", "Good Performance"]
        ).tolist(),
        "classification_report": classification_report(y_test, predictions, zero_division=0)
    }
    results[name] = metrics
    print(f"\n{name}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision_weighted']:.3f}")
    print(f"Recall: {metrics['recall_weighted']:.3f}")
    print(f"F1-score: {metrics['f1_weighted']:.3f}")
    print("Confusion matrix:\n", metrics["confusion_matrix"])

    if metrics["f1_weighted"] > best_score:
        best_score = metrics["f1_weighted"]
        best_name = name
        best_pipeline = pipe

# Cross-validation is run on the selected model's pipeline.
cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring="f1_weighted")
results["cross_validation_best_model_f1"] = cv_scores.tolist()
results["best_model"] = best_name

joblib.dump(best_pipeline, MODEL_DIR / "best_student_model.joblib")
(OUT_DIR / "evaluation_results.json").write_text(json.dumps(results, indent=2))
print("\nBest model:", best_name)
print("5-fold CV weighted F1:", [round(float(x), 3) for x in cv_scores])
print("Saved model to models/best_student_model.joblib")
print("Saved metrics to outputs/evaluation_results.json")
