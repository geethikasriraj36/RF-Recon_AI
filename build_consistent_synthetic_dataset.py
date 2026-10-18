
from pathlib import Path
import numpy as np
import pandas as pd

from rfrecon_consistent_features import (
    extract_features,
    FEATURE_NAMES,
)

SOURCE_DIR = Path("robust_training_data")
OUTPUT_FILE = Path("robust_training_dataset_consistent_5class.csv")
SAMPLE_RATE = 2_000_000

if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Source directory not found: {SOURCE_DIR.resolve()}"
    )

rows = []
class_counts = {}

for class_dir in sorted(SOURCE_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    label = class_dir.name
    files = sorted(class_dir.glob("*.npy"))

    if not files:
        continue

    class_counts[label] = len(files)
    print(f"Processing {label}: {len(files)} samples")

    for file_path in files:
        iq = np.load(file_path, allow_pickle=False)

        if iq.ndim != 1 or not np.iscomplexobj(iq):
            raise ValueError(
                f"Expected 1-D complex IQ data: {file_path}"
            )

        features = extract_features(iq, SAMPLE_RATE)
        features["class_name"] = label
        features["source_file"] = file_path.name
        rows.append(features)

if not rows:
    raise RuntimeError("No .npy training samples were found.")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset creation complete.")
print("Output:", OUTPUT_FILE.resolve())
print("Shape:", df.shape)
print("\nClass counts:")
print(df["class_name"].value_counts().to_string())
print("\nFeature summary:")
print(df[FEATURE_NAMES].describe().round(4).to_string())
print("\nSaved successfully.")
