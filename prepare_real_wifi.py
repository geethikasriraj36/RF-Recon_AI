import numpy as np
import os
import pandas as pd

INPUT_FILE = "real_dataset/real_wifi_01.dat"
OUTPUT_DIR = "real_dataset/wifi_like"
METADATA_FILE = "real_dataset/real_wifi_dataset.csv"

WINDOW_SIZE = 4096
NUM_WINDOWS = 500
SAMPLE_RATE = 20e6

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("RFRecon - REAL WIFI DATASET PREPARATION")
print("=" * 60)

print("\nLoading capture...")
data = np.fromfile(INPUT_FILE, dtype=np.complex64)

total_samples = len(data)
required_samples = WINDOW_SIZE * NUM_WINDOWS

print(f"Total samples: {total_samples:,}")
print(f"Required samples: {required_samples:,}")

# Spread windows across the entire recording
start_positions = np.linspace(
    0,
    total_samples - WINDOW_SIZE,
    NUM_WINDOWS,
    dtype=np.int64
)

metadata = []

print("\nExtracting windows...")

for i, start in enumerate(start_positions, start=1):

    end = start + WINDOW_SIZE

    window = data[start:end]

    filename = f"wifi_real_{i:03d}.dat"
    filepath = os.path.join(OUTPUT_DIR, filename)

    window.astype(np.complex64).tofile(filepath)

    metadata.append({
        "sample_id": i,
        "label": "wifi_like",
        "source_file": INPUT_FILE,
        "window_file": filepath,
        "start_sample": start,
        "end_sample": end,
        "window_size": WINDOW_SIZE,
        "sample_rate": SAMPLE_RATE
    })

    if i % 100 == 0:
        print(f"Extracted {i}/{NUM_WINDOWS}")

df = pd.DataFrame(metadata)
df.to_csv(METADATA_FILE, index=False)

print("\n" + "=" * 60)
print("REAL WIFI DATASET CREATED")
print("=" * 60)

print("\nClass: wifi_like")
print(f"Windows: {NUM_WINDOWS}")
print(f"Samples per window: {WINDOW_SIZE}")
print(f"Total complex samples: {required_samples:,}")

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Metadata file: {METADATA_FILE}")

print("\nDataset preparation completed successfully.")