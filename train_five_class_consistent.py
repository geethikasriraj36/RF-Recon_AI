
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from rfrecon_consistent_features import FEATURE_NAMES

DATA_FILE = Path("robust_training_dataset_consistent_5class.csv")
MODEL_FILE = Path("rfrecon_five_class_synthetic_rf.pkl")
FEATURE_FILE = Path("rfrecon_five_class_synthetic_features.pkl")
REPORT_FILE = Path("rfrecon_five_class_synthetic_report.csv")
MATRIX_FILE = Path("rfrecon_five_class_synthetic_confusion_matrix.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

required = FEATURE_NAMES + ["class_name"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[required].isnull().any().any():
    raise ValueError("Missing values detected. Check the dataset.")

X = df[FEATURE_NAMES].astype(float)
y = df["class_name"].astype(str)

if not X.map(lambda col: col).apply(
    lambda col: col.map(lambda value: abs(value) != float("inf"))
).all().all():
    raise ValueError("Infinite feature values detected.")

if y.nunique() != 5:
    raise ValueError(f"Expected five classes, found {y.nunique()}.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(
    y_test,
    predictions,
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_,
)

joblib.dump(model, MODEL_FILE)
joblib.dump(list(FEATURE_NAMES), FEATURE_FILE)

pd.DataFrame(report).transpose().to_csv(REPORT_FILE)
pd.DataFrame(
    matrix,
    index=model.classes_,
    columns=model.classes_,
).to_csv(MATRIX_FILE)

print("\nFIVE-CLASS SYNTHETIC EVALUATION")
print("--------------------------------")
print("Dataset rows:", len(df))
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print("Classes:", list(model.classes_))
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("Confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(
    matrix,
    index=model.classes_,
    columns=model.classes_,
))

print("\nSaved new artifacts:")
for path in [MODEL_FILE, FEATURE_FILE, REPORT_FILE, MATRIX_FILE]:
    print(path.resolve())

print("\nLIMITATION: Evaluation uses synthetic samples,")
print("not an independent real-world RF capture test.")
