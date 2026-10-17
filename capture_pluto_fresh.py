
from pathlib import Path
from datetime import datetime
import numpy as np
import iio

OUTPUT_DIR = Path("real_dataset")
SAMPLES = 262144

# Unique filename for every capture; existing captures are preserved.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT = OUTPUT_DIR / f"pluto_fresh_{timestamp}.dat"

ctx = iio.Context("ip:192.168.2.1")
rx = ctx.find_device("cf-ad9361-lpc")

if rx is None:
    raise RuntimeError("Pluto RX data device not found.")

channels = {ch.id: ch for ch in rx.channels}

if "voltage0" not in channels or "voltage1" not in channels:
    raise RuntimeError(f"Expected I/Q channels; found {list(channels)}")

i_ch = channels["voltage0"]
q_ch = channels["voltage1"]
buf = None

try:
    i_ch.enabled = True
    q_ch.enabled = True

    buf = iio.Buffer(rx, SAMPLES, False)
    buf.refill()
    raw = buf.read()

    expected_bytes = SAMPLES * 4
    print("Bytes received:", len(raw))
    print("Expected bytes:", expected_bytes)

    if len(raw) != expected_bytes:
        raise RuntimeError("Unexpected I/Q buffer size.")

    iq16 = np.frombuffer(raw, dtype="<i2")

    iq = (
        iq16[0::2].astype(np.float32)
        + 1j * iq16[1::2].astype(np.float32)
    ) / 32768.0

    if not np.isfinite(iq.real).all() or not np.isfinite(iq.imag).all():
        raise RuntimeError("Non-finite I/Q samples detected.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    iq.astype(np.complex64).tofile(OUTPUT)

    print("\nFresh Pluto capture saved:", OUTPUT)
    print("Complex samples:", len(iq))
    print("Mean power:", float(np.mean(np.abs(iq) ** 2)))
    print("Peak magnitude:", float(np.max(np.abs(iq))))
    print("I range:", float(iq.real.min()), float(iq.real.max()))
    print("Q range:", float(iq.imag.min()), float(iq.imag.max()))

finally:
    if buf is not None:
        del buf
    i_ch.enabled = False
    q_ch.enabled = False

print("\nCapture complete.")
print("Note: device sampling rate and RF settings were not changed.")
