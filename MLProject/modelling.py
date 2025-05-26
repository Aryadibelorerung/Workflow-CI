import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Aktifkan autologging dari MLflow
mlflow.sklearn.autolog()

# Atur URI tracking dan eksperimen
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mytitanic_experiment")

# Load dataset
df = pd.read_csv("../preprocessing/titanic_clean.csv")
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Inisialisasi model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Mulai run MLflow
with mlflow.start_run():
    model.fit(X_train, y_train)
    
    # Log model secara eksplisit (opsional karena autolog sudah aktif)
    mlflow.sklearn.log_model(model, "model")
    
    # Gunakan .score() untuk akurasi
    accuracy = model.score(X_test, y_test)
    
    # Log metrik akurasi
    mlflow.log_metric("accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")