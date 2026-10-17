
import numpy as np
import pandas as pd

from rfrecon_consistent_features import (
    extract_features,
    FEATURE_NAMES,
)

files = [
    "real_dataset/pluto_rx_smoke_test.dat",
    "real_dataset/pluto_fresh_20261014_202446.dat",
]

rows = []
window_size = 4096
sample_rate = 30_720_000

for filename in files:
    iq = np.fromfile(filename, dtype=np.complex64)
    count = len(iq) // window_size

    print(f"{filename}: {count} complete windows")

    for i in range(count):
        start = i * window_size
        window = iq[start:start + window_size]

        features = extract_features(window, sample_rate)
        features["capture"] = filename.split("/")[-1]
        rows.append(features)

df = pd.DataFrame(rows)

print("\nMedian feature comparison:")
print(
    df.groupby("capture")[FEATURE_NAMES]
    .median()
    .T.round(6)
    .to_string()
)

print("\nComparison complete. No models or captures modified.")
