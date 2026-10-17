import numpy as np
import os
import pandas as pd

INPUT_FILE = r"real_dataset\real_cellular_01.dat"
OUTPUT_DIR = r"real_dataset\cellular_like"
METADATA_FILE = r"real_dataset\real_cellular_dataset.csv"

WINDOW_SIZE = 4096
NUM_WINDOWS = 500

CENTER_FREQ = 1.872e9
SAMPLE_RATE = 20e6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Memory-map the capture
data = np.memmap(
    INPUT_FILE,
    dtype=np.complex64,
    mode="r"
)

total_samples = len(data)

print("=" * 60)
print("RFRecon - REAL CELLULAR DATASET PREPARATION")
print("=" * 60)

print(f"Total samples : {total_samples:,}")
print(f"Sample rate   : {SAMPLE_RATE/1e6:.2f} MS/s")
print(f"Center freq   : {CENTER_FREQ/1e9:.6f} GHz")
print(f"Window size   : {WINDOW_SIZE}")
print(f"Windows       : {NUM_WINDOWS}")

# Evenly distribute windows across the entire capture
max_start = total_samples - WINDOW_SIZE

starts = np.linspace(
    0,
    max_start,
    NUM_WINDOWS,
    dtype=np.int64
)

metadata = []

for i, start in enumerate(starts):

    window = np.array(
        data[start:start + WINDOW_SIZE],
        dtype=np.complex64
    )

    filename = f"cellular_{i+1:04d}.npy"
    filepath = os.path.join(OUTPUT_DIR, filename)

    np.save(filepath, window)

    metadata.append({
        "filename": filename,
        "label": "cellular_like",
        "start_sample": int(start),
        "window_size": WINDOW_SIZE,
        "sample_rate": SAMPLE_RATE,
        "center_frequency": CENTER_FREQ
    })

    if (i + 1) % 100 == 0:
        print(f"Saved {i+1}/500 windows")

# Save metadata
df = pd.DataFrame(metadata)
df.to_csv(METADATA_FILE, index=False)

print()
print("=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print(f"Windows saved : {len(df)}")
print(f"Output folder : {OUTPUT_DIR}")
print(f"Metadata file : {METADATA_FILE}")