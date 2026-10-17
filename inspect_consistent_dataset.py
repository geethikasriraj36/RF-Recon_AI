
import pandas as pd

FILE = "robust_training_dataset_consistent.csv"

df = pd.read_csv(FILE)

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
    "peak_to_average_ratio",
]

print("Dataset shape:", df.shape)
print("\nClass counts:")
print(df["class_name"].value_counts().to_string())

print("\nMean feature values by class:")
print(
    df.groupby("class_name")[features]
    .mean()
    .round(4)
    .T
    .to_string()
)

print("\nStandard deviation by class:")
print(
    df.groupby("class_name")[features]
    .std()
    .round(4)
    .T
    .to_string()
)

print("\nFeatures with near-zero global variation:")
for feature in features:
    std = df[feature].std()
    print(f"{feature}: std={std:.8f}")
