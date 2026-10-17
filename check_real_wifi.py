import numpy as np
import os

FILE = "real_dataset/real_wifi_01.dat"
SAMPLE_RATE = 20e6

print("=" * 60)
print("RFRecon - REAL WIFI CAPTURE CHECK")
print("=" * 60)

print("\nFile:")
print(FILE)

if not os.path.exists(FILE):
    print("\nERROR: Wi-Fi capture file not found.")
    exit()

data = np.fromfile(FILE, dtype=np.complex64)

num_samples = len(data)
duration = num_samples / SAMPLE_RATE

amplitude = np.abs(data)

mean_amplitude = np.mean(amplitude)
rms_amplitude = np.sqrt(np.mean(np.abs(data) ** 2))
peak_amplitude = np.max(amplitude)
mean_power = np.mean(np.abs(data) ** 2)
power_variance = np.var(np.abs(data) ** 2)

nan_count = np.isnan(data).sum()
inf_count = np.isinf(data).sum()

print(f"\nNumber of complex samples:")
print(f"{num_samples:,}")

print("\nSample rate:")
print(f"{SAMPLE_RATE / 1e6:.2f} MS/s")

print("\nRecording duration:")
print(f"{duration:.3f} seconds")

print("\n" + "-" * 60)
print("SIGNAL STATISTICS")
print("-" * 60)

print(f"\nMean amplitude : {mean_amplitude:.6f}")
print(f"RMS amplitude  : {rms_amplitude:.6f}")
print(f"Peak amplitude : {peak_amplitude:.6f}")
print(f"Mean power     : {mean_power:.8f}")
print(f"Power variance : {power_variance:.8f}")

print("\n" + "-" * 60)
print("DATA VALIDITY")
print("-" * 60)

print(f"\nNaN samples : {nan_count:,}")
print(f"Inf samples : {inf_count:,}")

print("\n" + "=" * 60)

if nan_count == 0 and inf_count == 0 and num_samples >= 500 * 4096:
    print("REAL WIFI CAPTURE CHECK: PASSED")
    print("\nThe capture is readable and contains valid complex64 I/Q data.")
else:
    print("REAL WIFI CAPTURE CHECK: FAILED")

print("=" * 60)