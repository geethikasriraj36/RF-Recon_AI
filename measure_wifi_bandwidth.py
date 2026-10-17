import numpy as np
from scipy.signal import welch

FILE = "real_wifi_capture_02.dat"
FS = 20_000_000
CENTER = 2_437_000_000

x = np.fromfile(FILE, dtype=np.complex64)
x = x[:1_000_000]

f, psd = welch(
    x,
    fs=FS,
    nperseg=8192,
    return_onesided=False,
    scaling="density"
)

f = np.fft.fftshift(f)
psd = np.fft.fftshift(psd)

psd_db = 10 * np.log10(psd + 1e-15)

# Estimate noise floor using the lower 20% of PSD values
noise_floor = np.percentile(psd_db, 20)

# Threshold 6 dB above estimated noise floor
threshold = noise_floor + 6

active = psd_db > threshold

if np.any(active):
    active_freqs = f[active]

    low = active_freqs.min()
    high = active_freqs.max()

    bandwidth = high - low
    center_offset = (low + high) / 2

    print("======================================")
    print("RFRecon - REAL RF BANDWIDTH ANALYSIS")
    print("======================================")
    print(f"Estimated noise floor : {noise_floor:.2f} dB")
    print(f"Detection threshold   : {threshold:.2f} dB")
    print(f"Lower frequency       : {low/1e6:.3f} MHz")
    print(f"Upper frequency       : {high/1e6:.3f} MHz")
    print(f"Occupied bandwidth    : {bandwidth/1e6:.3f} MHz")
    print(f"Center offset         : {center_offset/1e6:.3f} MHz")
    print(f"Absolute center       : {(CENTER+center_offset)/1e6:.3f} MHz")
else:
    print("No significant RF activity detected.")