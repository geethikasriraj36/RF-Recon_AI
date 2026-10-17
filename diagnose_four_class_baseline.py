
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

DATASET = Path("real_dataset/four_class_baseline_features.csv")
MODEL_PATH = Path("rfrecon_four_class_baseline_rf.pkl")
FEATURES_PATH = Path("rfrecon_four_class_baseline_features.pkl")

df = pd.read_csv(DATASET)

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

with FEATURES_PATH.open("rb") as f:
    feature_names = pickle.load(f)

print("========== FEATURE IMPORTANCE ==========")

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print(importance.to_string(
    index=False,
    formatters={"importance": "{:.4f}".format},
))

print("\n========== NOISE VS WIFI ==========")

subset = df[df["class_name"].isin(["noise", "wifi_like"])]

summary = subset.groupby("class_name")[feature_names].agg(
    ["mean", "std", "median"]
)

for feature in feature_names:
    print(f"\n{feature}")
    for label in ["noise", "wifi_like"]:
        values = subset.loc[
            subset["class_name"] == label, feature
        ]
        print(
            f"  {label:10s} "
            f"mean={values.mean():.6g} "
            f"std={values.std():.6g} "
            f"median={values.median():.6g}"
        )

print("\n========== CLASS FEATURE MEANS ==========")
print(
    df.groupby("class_name")[feature_names]
    .mean()
    .round(5)
    .T.to_string()
)

print("\n========== DUPLICATE FEATURE VECTORS ==========")
print(
    "Exact duplicate feature rows:",
    int(df[feature_names].duplicated().sum())
)

print("\nDiagnosis complete. No files were modified.")
