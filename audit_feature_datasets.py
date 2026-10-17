
import pandas as pd
import numpy as np

files = [
    "real_dataset/four_class_baseline_features.csv",
    "real_dataset/ieee802154_6lowpan_features.csv",
]

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

for file in files:
    print("\n---", file, "---")
    df = pd.read_csv(file)

    print("Shape:", df.shape)
    print("Total missing values:", int(df.isna().sum().sum()))
    print("All expected features present:",
          all(col in df.columns for col in features))
    print("Feature order correct:",
          list(df.columns[:10]) == features)

    if "class_name" in df.columns:
        print("Class counts:", df["class_name"].value_counts().to_dict())
    else:
        print("Class counts: class_name column missing")

    available = [col for col in features if col in df.columns]
    values = df[available].to_numpy(dtype=float)

    print("All feature values finite:", bool(np.isfinite(values).all()))
    print("Duplicate feature rows:",
          int(df[available].duplicated().sum()))
    print("Metadata columns:",
          [col for col in df.columns if col not in features])
    print("Missing values by column:")
    print(df.isna().sum()[df.isna().sum() > 0].to_dict())