import numpy as np
import os

# Input real PlutoSDR capture
input_file = "real_pluto_capture_01.dat"

# Output folder
output_folder = "real_pluto_data"
os.makedirs(output_folder, exist_ok=True)

# RFRecon AI expects exactly 4096 complex samples
SAMPLES_PER_FILE = 4096

# Number of real samples to create for testing
MAX_FILES = 500

print("Reading PlutoSDR IQ capture...")

with open(input_file, "rb") as f:

    for i in range(MAX_FILES):

        # Read exactly 4096 complex64 samples
        data = np.fromfile(
            f,
            dtype=np.complex64,
            count=SAMPLES_PER_FILE
        )

        # Stop if file has ended
        if len(data) < SAMPLES_PER_FILE:
            break

        output_file = os.path.join(
            output_folder,
            f"real_pluto_{i:04d}.npy"
        )

        np.save(output_file, data)

        if i % 50 == 0:
            print(f"Created {i + 1} samples")

print("\nConversion complete!")
print(f"Output folder: {output_folder}")
print(f"Files created: {i + 1}")