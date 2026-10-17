import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("rf_dataset.csv")

print("Dataset loaded!")
print("Total examples:", len(df))


# --------------------------------------------------
# 2. SELECT FEATURES
# --------------------------------------------------

features = [
    "mean_power",
    "rms",
    "peak_amplitude",
    "peak_frequency_normalized",
    "spectral_power"
]

X = df[features]
y = df["label"]


# --------------------------------------------------
# 3. SPLIT DATA
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. TRAIN RANDOM FOREST
# --------------------------------------------------

print("\nTraining RFRecon AI model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 6. ACCURACY
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL RESULTS")
print("===================================")

print(f"Accuracy: {accuracy * 100:.2f}%")


# --------------------------------------------------
# 7. CLASSIFICATION REPORT
# --------------------------------------------------

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 8. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2]
)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------------------------
# 9. DISPLAY CONFUSION MATRIX
# --------------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Noise", "Tone", "Tone + Noise"]
)

display.plot()

plt.title("RFRecon AI - Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()