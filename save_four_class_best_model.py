

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from rfrecon_consistent_features import FEATURE_NAMES

DATA_FILE = "real_dataset/four_class_baseline_features.csv"
MODEL_FILE = "rfrecon_four_class_best_rf.pkl"
FEATURE_FILE = "rfrecon_four_class_best_features.pkl"

df = pd.read_csv(DATA_FILE)

required = FEATURE_NAMES + ["class_name"]
missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if df["class_name"].nunique() != 4:
    raise ValueError("Expected exactly four classes.")

if df[FEATURE_NAMES].isna().any().any():
    raise ValueError("Features contain missing values.")

# Split each class independently, preserving all four classes.
train_parts = []
test_parts = []

for class_name, group in df.groupby("class_name", sort=True):
    train_part, test_part = train_test_split(
        group,
        test_size=0.30,
        random_state=42,
        shuffle=True,
    )
    train_parts.append(train_part)
    test_parts.append(test_part)

train_df = pd.concat(train_parts).sample(frac=1, random_state=42)
test_df = pd.concat(test_parts).sample(frac=1, random_state=42)

X_train = train_df[FEATURE_NAMES]
y_train = train_df["class_name"]
X_test = test_df[FEATURE_NAMES]
y_test = test_df["class_name"]

print("Training class counts:\n", y_train.value_counts())
print("\nTesting class counts:\n", y_test.value_counts())

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nStratified holdout results")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Macro F1:", round(f1_score(y_test, predictions, average="macro"), 4))
print("\nClassification report:\n")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, MODEL_FILE)
joblib.dump(FEATURE_NAMES, FEATURE_FILE)

print("\nSaved:", MODEL_FILE)
print("Saved:", FEATURE_FILE)
print("Model classes:", list(model.classes_))
print("\nDone.")
