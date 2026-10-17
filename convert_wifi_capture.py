import numpy as np
import os

input_file = "real_wifi_capture_01.dat"

output_folder = "real_wifi_data"
os.makedirs(output_folder, exist_ok=True)

SAMPLES_PER_FILE = 4096
MAX_FILES = 500

print("Reading real Wi-Fi PlutoSDR capture...")

with open(input_file, "rb") as f:

    for i in range(MAX_FILES):

        data = np.fromfile(
            f,
            dtype=np.complex64,
            count=SAMPLES_PER_FILE
        )

        if len(data) < SAMPLES_PER_FILE:
            break

        # Remove DC offset from each window
        data = data - np.mean(data)

        output_file = os.path.join(
            output_folder,
            f"real_wifi_{i:04d}.npy"
        )

        np.save(output_file, data)

        if i % 50 == 0:
            print(f"Created {i + 1} Wi-Fi samples")

print("\nConversion complete!")
print(f"Output folder: {output_folder}")
print(f"Files created: {i + 1}")