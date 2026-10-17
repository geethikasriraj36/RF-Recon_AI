
import numpy as np

file_path = r"real_dataset\real_bluetooth_01.dat"
data = np.fromfile(file_path, dtype=np.complex64)

print("Bluetooth IQ trend analysis")
print("-" * 65)

for i, section in enumerate(np.array_split(data, 10), start=1):
    mean_i = np.mean(section.real)
    mean_q = np.mean(section.imag)
    rms = np.sqrt(np.mean(np.abs(section) ** 2))

    print(
        f"Section {i}: "
        f"mean_I={mean_i:.6f}, "
        f"mean_Q={mean_q:.6f}, "
        f"RMS={rms:.6f}"
    )
