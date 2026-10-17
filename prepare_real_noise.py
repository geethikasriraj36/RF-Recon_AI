import numpy as np
import pandas as pd
import os


# ============================================================
# RFRecon - PREPARE REAL NOISE DATASET
# ============================================================

INPUT_FILE = "real_dataset/real_noise_01.dat"

OUTPUT_DIR = "real_dataset/noise"

OUTPUT_CSV = "real_dataset/real_noise_dataset.csv"

WINDOW_SIZE = 4096

NUM_WINDOWS = 500

SAMPLE_RATE = 20_000_000


print("=" * 60)
print("RFRecon - REAL NOISE DATASET PREPARATION")
print("=" * 60)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD I/Q DATA
# ============================================================

print("\nLoading capture...")

iq = np.fromfile(
    INPUT_FILE,
    dtype=np.complex64
)

print(
    f"Total samples: {len(iq):,}"
)


# ============================================================
# CHECK SIZE
# ============================================================

required_samples = (
    WINDOW_SIZE * NUM_WINDOWS
)

print(
    f"Required samples: "
    f"{required_samples:,}"
)

if len(iq) < required_samples:

    raise ValueError(
        "Capture does not contain enough samples."
    )


# ============================================================
# SELECT 500 WINDOWS
# ============================================================

print("\nExtracting windows...")

# Spread windows throughout the capture
# rather than taking only the first 500 windows.

max_start = (
    len(iq)
    - WINDOW_SIZE
)

starts = np.linspace(
    0,
    max_start,
    NUM_WINDOWS,
    dtype=np.int64
)


metadata = []


for i, start in enumerate(starts):

    end = (
        start
        + WINDOW_SIZE
    )

    window = iq[
        start:end
    ]

    output_file = os.path.join(
        OUTPUT_DIR,
        f"noise_real_{i+1:03d}.dat"
    )

    window.astype(
        np.complex64
    ).tofile(
        output_file
    )

    metadata.append({

        "sample_id": i + 1,

        "label": "noise",

        "source_file": INPUT_FILE,

        "window_file": output_file,

        "start_sample": int(start),

        "end_sample": int(end),

        "window_size": WINDOW_SIZE,

        "sample_rate": SAMPLE_RATE

    })


    if (i + 1) % 100 == 0:

        print(
            f"Extracted "
            f"{i + 1}/{NUM_WINDOWS}"
        )


# ============================================================
# SAVE METADATA
# ============================================================

df = pd.DataFrame(
    metadata
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("REAL NOISE DATASET CREATED")
print("=" * 60)

print(
    f"\nClass:"
    f" noise"
)

print(
    f"Windows:"
    f" {len(df)}"
)

print(
    f"Samples per window:"
    f" {WINDOW_SIZE}"
)

print(
    f"Total complex samples:"
    f" {len(df) * WINDOW_SIZE:,}"
)

print(
    f"\nOutput directory:"
    f" {OUTPUT_DIR}"
)

print(
    f"\nMetadata file:"
    f" {OUTPUT_CSV}"
)

print("\nDataset preparation completed successfully.")