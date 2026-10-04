"""
Heart Disease Prediction System - Training Script
=================================================
This is the Titanic survival code, rebuilt for heart disease.
It follows the SAME steps (1 to 9) as the Titanic code.

What it does:
  - Loads the heart disease data (4 hospitals, 920 patients)
  - Uses ONLY 4 input attributes:  sex, cp, exang, oldpeak
  - Trains an SVM (Support Vector Classifier)
  - Tests it, saves it, and predicts for a new patient

Run it with:   python train_model.py
"""

# ============================================================
# Step 1: Import Libraries
# ============================================================
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # draw charts into files (works on servers too)
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Step 2: Load Sample Data
# ============================================================
# The dataset comes as 4 files (one per hospital). They have no header row,
# so we give every column a name. Missing values are written as "?".
ALL_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
               "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

FILES = ["processed.cleveland.data", "processed.hungarian.data",
         "processed.switzerland.data", "processed.va.data"]

parts = [pd.read_csv(DATA_DIR / f, header=None, names=ALL_COLUMNS, na_values="?")
         for f in FILES]
sample_data = pd.concat(parts, ignore_index=True)

print("Rows loaded from the 4 hospital files:", len(sample_data))


# ============================================================
# Step 3: Understand Sample Data  (select ONLY 4 attributes)
# ============================================================
#   sex      : 1 = male, 0 = female
#   cp       : chest pain type  (1 typical angina, 2 atypical angina,
#                                3 non-anginal pain, 4 no symptoms)
#   exang    : exercise-induced angina (1 = yes, 0 = no)
#   oldpeak  : ST depression on the ECG during exercise (a number)
FEATURES = ["sex", "cp", "exang", "oldpeak"]

# The original label "num" is 0 (healthy) or 1-4 (disease, getting worse).
# We turn it into a simple 0 / 1 label:  0 = no disease, 1 = disease.
sample_data["HeartDisease"] = (sample_data["num"] > 0).astype(int)
TARGET = "HeartDisease"

sample_data = sample_data[FEATURES + [TARGET]]

# Some patients have a missing value in one of the 4 columns -> remove them.
before = len(sample_data)
sample_data = sample_data.dropna()
print(f"Removed {before - len(sample_data)} rows with missing values, "
      f"{len(sample_data)} rows remain.")

print("\nAttributes used:")
print("================\n")
print(list(sample_data.columns))
print("\nNumber of instances:", sample_data[TARGET].count())
print(sample_data.head())


# ============================================================
# Step 5: Encoding
# ============================================================
# In the Titanic code this step turned words ("male", "S", ...) into numbers.
# Our heart data is ALREADY numbers, so NO label encoding is needed.
# (As your teacher said: skip it.)
sample_data.to_csv(OUT_DIR / "sample-data.csv", index=False)


# ============================================================
# Step 6.1: Split Data into Training and Testing Sets
# ============================================================
# 80% of patients to teach the model, 20% hidden to test it afterwards.
# stratify keeps the same healthy/disease ratio in both parts.
training_data, testing_data = train_test_split(
    sample_data, test_size=0.2, random_state=0, shuffle=True,
    stratify=sample_data[TARGET]
)
training_data.to_csv(OUT_DIR / "training-data.csv", index=False)
testing_data.to_csv(OUT_DIR / "testing-data.csv", index=False)
print("\nTraining Data shape:", training_data.shape)
print("Testing Data shape:", testing_data.shape)


# ============================================================
# Step 6.2: Split Inputs and Labels (Training Data)
# ============================================================
input_vector_train = training_data[FEATURES]
output_label_train = training_data[TARGET]


# ============================================================
# Step 6.3: Train the Support Vector Classifier
# ============================================================
# Same SVC as Titanic. We put a StandardScaler in front of it because
# "oldpeak" is on a bigger scale than the 0/1 columns, and SVMs work best
# when all inputs are on a similar scale. The pipeline does both in one go.
svc_model = make_pipeline(StandardScaler(), svm.SVC(gamma="auto", random_state=0))
svc_model.fit(input_vector_train, np.ravel(output_label_train))
print("\nTrained model:", svc_model)


# ============================================================
# Step 6.4: Save the Trained Model
# ============================================================
with open(MODEL_DIR / "svc_trained_model.pkl", "wb") as f:
    pickle.dump(svc_model, f)
print("Saved: model/svc_trained_model.pkl")


# ============================================================
# Step 7.1: Split Inputs and Labels (Testing Data)
# ============================================================
input_vector_test = testing_data[FEATURES]
output_label_test = testing_data[TARGET]


# ============================================================
# Step 7.2: Load the Saved Model
# ============================================================
with open(MODEL_DIR / "svc_trained_model.pkl", "rb") as f:
    model = pickle.load(f)


# ============================================================
# Step 7.3: Make Predictions on Testing Data
# ============================================================
testing_data = testing_data.copy()
testing_data["Predictions"] = model.predict(input_vector_test)
testing_data.to_csv(OUT_DIR / "model-predictions.csv", index=False)
print("\nFirst 10 test predictions:")
print(testing_data.head(10))


# ============================================================
# Step 7.4: Calculate the Accuracy Score
# ============================================================
accuracy = accuracy_score(testing_data[TARGET], testing_data["Predictions"])
print("\nAccuracy Score:", round(accuracy, 2))

# Save the numbers so the website can show them
import json
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump({"accuracy": round(float(accuracy), 4),
               "test_patients": int(len(testing_data)),
               "train_patients": int(len(training_data)),
               "total_patients": int(len(sample_data))}, f)


# ============================================================
# Step 7.5: Detailed Evaluation Metrics
# ============================================================
print("\nClassification Report:\n")
print(classification_report(
    testing_data[TARGET], testing_data["Predictions"],
    target_names=["No Heart Disease (0)", "Heart Disease (1)"]
))

cm = confusion_matrix(testing_data[TARGET], testing_data["Predictions"])
print("Confusion Matrix:\n", cm)

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
labels = ["No Disease", "Disease"]
ax.set_xticks([0, 1], labels)
ax.set_yticks([0, 1], labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - SVC on Heart Disease Testing Data")
for i in range(2):
    for j in range(2):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
print("Saved: outputs/confusion_matrix.png")


# ============================================================
# Step 8: Predict for a new patient
# ============================================================
# 8.1 Take input (edit these values to try other patients)
sex_input = 1          # 1 = male, 0 = female
chest_pain_input = 4   # 1, 2, 3 or 4
angina_input = 1       # 1 = yes, 0 = no
oldpeak_input = 2.0    # ST depression number

# 8.2 Build the feature vector
user_input = pd.DataFrame({
    "sex": [sex_input],
    "cp": [chest_pain_input],
    "exang": [angina_input],
    "oldpeak": [oldpeak_input],
})
print("\nUser input:\n", user_input)

# 8.3 (No encoding needed - values are already numbers)

# 8.4 + 8.5 Load model and predict
with open(MODEL_DIR / "svc_trained_model.pkl", "rb") as f:
    model = pickle.load(f)
result = model.predict(user_input[FEATURES])[0]
print("\n** Prediction:", "HEART DISEASE LIKELY" if result == 1 else "NO HEART DISEASE LIKELY", "**")


# ============================================================
# Step 9: Feedback and Improvements
# ============================================================
improvements = [
    "Add more attributes such as age, thalach (max heart rate) and chol",
    "Use cross-validation instead of a single train/test split",
    "Compare other models (Random Forest, Logistic Regression)",
    "Try tuning SVC settings (C and gamma) with GridSearchCV",
]
print("\nIdeas to improve:")
for i, item in enumerate(improvements, 1):
    print(f"{i}. {item}")
