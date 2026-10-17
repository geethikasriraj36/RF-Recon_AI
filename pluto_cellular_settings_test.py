
import iio
import numpy as np
from pathlib import Path
from datetime import datetime

URI = "ip:192.168.2.1"
SAMPLE_RATE = 20_000_000
RF_BANDWIDTH = 18_000_000
CENTER_FREQUENCY = 1_872_000_000
SAMPLES = 262_144

ctx = iio.Context(URI)
phy = ctx.find_device("ad9361-phy")
rx = ctx.find_device("cf-ad9361-lpc")

if phy is None or rx is None:
    raise RuntimeError("Could not find Pluto RX devices.")

lo = phy.find_channel("altvoltage0", True)
rx0 = rx.find_channel("voltage0", False)
rx1 = rx.find_channel("voltage1", False)

print("Current settings will be changed for this test.")
print("LO: 1.872 GHz | Sample rate: 20 MS/s")
print("RF bandwidth: 18 MHz | Gain mode: slow_attack")

try:
    lo.attrs["frequency"].value = str(CENTER_FREQUENCY)
    rx0.attrs["sampling_frequency"].value = str(SAMPLE_RATE)
    rx0.attrs["rf_bandwidth"].value = str(RF_BANDWIDTH)
    rx0.attrs["gain_control_mode"].value = "slow_attack"

    rx0.enabled = True
    rx1.enabled = True

    buf = iio.Buffer(rx, SAMPLES, False)
    try:
        buf.refill()
        raw = np.frombuffer(buf.read(), dtype=np.int16).copy()
    finally:
        del buf

    if raw.size % 2:
        raw = raw[:-1]

    iq = (
        raw[0::2].astype(np.float32)
        + 1j * raw[1::2].astype(np.float32)
    ) / 32768.0

    out = Path("real_dataset")
    out.mkdir(exist_ok=True)
    filename = out / (
        "pluto_cellular_test_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".dat"
    )
    iq.astype(np.complex64).tofile(filename)

    print("\nCapture saved:", filename)
    print("Samples:", len(iq))
    print("Mean power:", float(np.mean(np.abs(iq) ** 2)))
    print("Peak magnitude:", float(np.max(np.abs(iq))))

finally:
    rx0.enabled = False
    rx1.enabled = False
    print("\nRX channels disabled.")
    print("Note: LO and sample rate remain changed until restored.")
