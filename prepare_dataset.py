import numpy as np
from pathlib import Path
import pandas as pd

DATA_DIR = Path("generated_data")

records = []

for file in DATA_DIR.glob("*.npy"):

    signal = np.load(file)

    # Determine class from filename
    if file.name.startswith("noise_"):
        label = 0
        class_name = "noise"

    elif file.name.startswith("tone_noise_"):
        label = 2
        class_name = "tone_noise"

    elif file.name.startswith("tone_"):
        label = 1
        class_name = "tone"

    else:
        continue

    # -----------------------------
    # Basic IQ features
    # -----------------------------

    magnitude = np.abs(signal)

    mean_power = np.mean(magnitude ** 2)

    rms = np.sqrt(mean_power)

    peak = np.max(magnitude)

    # -----------------------------
    # Frequency-domain features
    # -----------------------------

    fft = np.fft.fft(signal)

    fft_magnitude = np.abs(fft)

    peak_bin = np.argmax(fft_magnitude)

    peak_frequency = peak_bin / len(signal)

    spectral_power = np.mean(fft_magnitude ** 2)

    # -----------------------------
    # Store information
    # -----------------------------

    records.append({
        "filename": file.name,
        "label": label,
        "class": class_name,
        "mean_power": mean_power,
        "rms": rms,
        "peak_amplitude": peak,
        "peak_frequency_normalized": peak_frequency,
        "spectral_power": spectral_power
    })


# Convert to DataFrame
df = pd.DataFrame(records)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save metadata
df.to_csv("rf_dataset.csv", index=False)

print("Dataset preparation complete!")
print()
print("Total examples:", len(df))
print()
print("Class distribution:")
print(df["class"].value_counts())
print()
print("Dataset saved as: rf_dataset.csv")