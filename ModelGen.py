import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# --- Your data generation and model training logic remains the same ---
np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    "previous_sgpa": np.round(np.random.uniform(5.0, 9.0, n_samples), 2),
    "avg_programming_score": np.round(np.random.uniform(50, 100, n_samples), 2),
    "avg_practical_score": np.round(np.random.uniform(50, 100, n_samples), 2),
    "avg_conceptual_score": np.round(np.random.uniform(50, 100, n_samples), 2),
    "attendance": np.round(np.random.uniform(60, 100, n_samples), 2),
    "job_hours": np.round(np.random.uniform(0, 20, n_samples), 2),
})

data["sgpa"] = np.round((
    0.25 * data["previous_sgpa"] +
    0.15 * (data["avg_programming_score"] / 10) +
    0.15 * (data["avg_practical_score"] / 10) +
    0.15 * (data["avg_conceptual_score"] / 10) +
    0.15 * (data["attendance"] / 10) -
    0.15 * (data["job_hours"] / 10)
), 2).clip(lower=4.0, upper=10.0)

data["risk_level"] = pd.cut(data["sgpa"], bins=[0, 6, 7.5, 10], labels=["High", "Medium", "Low"])

def get_weak_course(row):
    scores = {
        "Programming": row["avg_programming_score"],
        "Practical": row["avg_practical_score"],
        "Conceptual": row["avg_conceptual_score"]
    }
    return min(scores, key=scores.get)

data["weak_course_type"] = data.apply(get_weak_course, axis=1)

label_encoder_risk = LabelEncoder()
label_encoder_weak = LabelEncoder()
data["risk_level_encoded"] = label_encoder_risk.fit_transform(data["risk_level"])
data["weak_course_type_encoded"] = label_encoder_weak.fit_transform(data["weak_course_type"])

features = data[["previous_sgpa", "avg_programming_score", "avg_practical_score", "avg_conceptual_score", "attendance", "job_hours"]]
sgpa_labels = data["sgpa"]
risk_labels = data["risk_level_encoded"]
weak_labels = data["weak_course_type_encoded"]

sgpa_model = RandomForestRegressor(random_state=42)
risk_model = RandomForestClassifier(random_state=42)
weak_model = RandomForestClassifier(random_state=42)

sgpa_model.fit(features, sgpa_labels)
risk_model.fit(features, risk_labels)
weak_model.fit(features, weak_labels)

# --- CORRECTED MODEL SAVING LOGIC ---

# Use Path to get the parent directory of your project
# This assumes the 'saved_models' directory is at the same level as 'server'
project_root = Path(__file__).resolve().parent.parent.parent.parent
models_dir = project_root / "saved_models"

# Ensure the directory exists
models_dir.mkdir(parents=True, exist_ok=True)

# Save the models using the pathlib objects.
joblib.dump(sgpa_model, models_dir / "sgpa_model.pkl")
joblib.dump(risk_model, models_dir / "risk_model.pkl")
joblib.dump(weak_model, models_dir / "weak_model.pkl")
joblib.dump(label_encoder_risk, models_dir / "label_encoder_risk.pkl")
joblib.dump(label_encoder_weak, models_dir / "label_encoder_weak.pkl")

# --- End of corrected logic ---

data.head()
