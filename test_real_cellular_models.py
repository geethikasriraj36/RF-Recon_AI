import pandas as pd
import numpy as np
import joblib

FEATURE_FILE = r"real_dataset\real_cellular_features.csv"

RF_MODEL = "rfrecon_hybrid_model.pkl"
RF_FEATURES = "rfrecon_hybrid_features.pkl"

MLP_MODEL = "rfrecon_mlp_model.pkl"
MLP_FEATURES = "rfrecon_mlp_features.pkl"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

df = pd.read_csv(FEATURE_FILE)

print("=" * 70)
print("RFRecon - REAL CELLULAR CANDIDATE MODEL TEST")
print("=" * 70)

print(f"Windows: {len(df)}")


# ---------------------------------------------------------
# Load Random Forest
# ---------------------------------------------------------

rf = joblib.load(RF_MODEL)
rf_features = joblib.load(RF_FEATURES)

X_rf = df[rf_features]

rf_predictions = rf.predict(X_rf)
rf_probabilities = rf.predict_proba(X_rf)

rf_confidence = np.max(rf_probabilities, axis=1)


# ---------------------------------------------------------
# Random Forest results
# ---------------------------------------------------------

print()
print("=" * 70)
print("RANDOM FOREST")
print("=" * 70)

print("\nPrediction distribution:")
print(pd.Series(rf_predictions).value_counts())

print(f"\nAverage confidence : {rf_confidence.mean()*100:.2f}%")
print(f"Maximum confidence : {rf_confidence.max()*100:.2f}%")
print(f"Minimum confidence : {rf_confidence.min()*100:.2f}%")


# ---------------------------------------------------------
# Load MLP
# ---------------------------------------------------------

mlp = joblib.load(MLP_MODEL)
mlp_features = joblib.load(MLP_FEATURES)

X_mlp = df[mlp_features]

mlp_predictions = mlp.predict(X_mlp)
mlp_probabilities = mlp.predict_proba(X_mlp)

mlp_confidence = np.max(mlp_probabilities, axis=1)


# ---------------------------------------------------------
# MLP results
# ---------------------------------------------------------

print()
print("=" * 70)
print("MLP NEURAL NETWORK")
print("=" * 70)

print("\nPrediction distribution:")
print(pd.Series(mlp_predictions).value_counts())

print(f"\nAverage confidence : {mlp_confidence.mean()*100:.2f}%")
print(f"Maximum confidence : {mlp_confidence.max()*100:.2f}%")
print(f"Minimum confidence : {mlp_confidence.min()*100:.2f}%")


# ---------------------------------------------------------
# Agreement
# ---------------------------------------------------------

agreement = np.mean(
    rf_predictions == mlp_predictions
) * 100

print()
print("=" * 70)
print("RF ↔ MLP AGREEMENT")
print("=" * 70)

print(f"Agreement: {agreement:.2f}%")


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results = df.copy()

results["RF_prediction"] = rf_predictions
results["RF_confidence"] = rf_confidence

results["MLP_prediction"] = mlp_predictions
results["MLP_confidence"] = mlp_confidence

results.to_csv(
    "real_dataset\\real_cellular_model_predictions.csv",
    index=False
)

print()
print("Saved:")
print("real_dataset\\real_cellular_model_predictions.csv")

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)