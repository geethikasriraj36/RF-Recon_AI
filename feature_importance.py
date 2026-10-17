import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD ROBUST DATASET
# ============================================================

df = pd.read_csv("robust_training_dataset.csv")

print("=" * 65)
print("RFRecon AI - Feature Importance Analysis")
print("=" * 65)

print("\nTotal examples:", len(df))


# ============================================================
# 2. FEATURES
# ============================================================

features = [
    "mean_power",
    "rms",
    "peak_amplitude",
    "variance",
    "peak_frequency_normalized",
    "spectral_centroid_normalized",
    "spectral_bandwidth_normalized",
    "occupied_bandwidth_normalized",
    "spectral_flatness",
    "peak_to_average_ratio"
]

X = df[features]
y = df["label"]


# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)


# ============================================================
# 4. CALCULATE FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ============================================================
# 5. PRINT RESULTS
# ============================================================

print("\n" + "=" * 65)
print("FEATURE IMPORTANCE")
print("=" * 65)

for _, row in importance_df.iterrows():

    print(
        f"{row['Feature']:<35}"
        f"{row['Importance']:.4f}"
    )


# ============================================================
# 6. SAVE RESULTS
# ============================================================

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)


# ============================================================
# 7. PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("RF Feature")

plt.title(
    "RFRecon AI - Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.show()


print("\nSaved:")
print("feature_importance.csv")
print("feature_importance.png")