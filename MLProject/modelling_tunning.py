import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
)

# MLflow Setup
mlflow.set_tracking_uri("https://dagshub.com/Aryadibelorerung/membangun-system-ml.mlflow")
mlflow.set_experiment("titanic_opt_experiment")

# Load data
df = pd.read_csv("../preprocessing/titanic_clean.csv")
X = df.drop(columns=["Survived"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Model and GridSearch
model = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}
grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Evaluate
def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred)
    }

test_metrics = get_metrics(y_test, best_model.predict(X_test))
train_metrics = get_metrics(y_train, best_model.predict(X_train))

# MLflow Logging
with mlflow.start_run():
    mlflow.sklearn.log_model(best_model, "model")

    for k, v in test_metrics.items():
        mlflow.log_metric(f"test_{k}", v)
    for k, v in train_metrics.items():
        mlflow.log_metric(f"train_{k}", v)

    for k, v in grid.best_params_.items():
        mlflow.log_param(k, v)

    # Save confusion matrix
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion Matrix")
    plt.savefig("training_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("training_confusion_matrix.png")

    # Save JSON artifacts
    with open("confusion_matrix.json", "w") as f:
        json.dump(cm.tolist(), f)
    mlflow.log_artifact("confusion_matrix.json")

    cv_results = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in grid.cv_results_.items()
    }
    with open("estimator.json", "w") as f:
        json.dump({"best_params": grid.best_params_, "cv_results": cv_results}, f, indent=4)
    mlflow.log_artifact("estimator.json")