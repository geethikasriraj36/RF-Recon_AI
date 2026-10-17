import numpy as np
import os

INPUT_FILE = "real_wifi_capture_02.dat"
OUTPUT_DIR = "real_wifi_dataset"

SAMPLES_PER_FILE = 4096
NUM_SAMPLES = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)

x = np.fromfile(INPUT_FILE, dtype=np.complex64)

required = SAMPLES_PER_FILE * NUM_SAMPLES

if len(x) < required:
    print("Not enough samples in capture.")
    print("Available:", len(x))
    print("Required :", required)
    exit()

# Use evenly spaced windows across the capture
indices = np.linspace(
    0,
    len(x) - SAMPLES_PER_FILE,
    NUM_SAMPLES,
    dtype=int
)

for i, start in enumerate(indices):

    sample = x[start:start + SAMPLES_PER_FILE]

    filename = os.path.join(
        OUTPUT_DIR,
        f"real_wifi_{i:04d}.npy"
    )

    np.save(filename, sample)

    if (i + 1) % 50 == 0:
        print(f"Created {i + 1}/500 samples")

print()
print("====================================")
print("REAL Wi-Fi DATASET CREATED")
print("====================================")
print("Class        : WiFi-like")
print("Samples      :", NUM_SAMPLES)
print("Samples/file :", SAMPLES_PER_FILE)
print("Sample rate  : 20 MHz")
print("Output       :", OUTPUT_DIR)