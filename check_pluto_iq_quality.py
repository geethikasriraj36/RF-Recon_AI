
from pathlib import Path
import numpy as np

path = Path("real_dataset/pluto_rx_smoke_test.dat")
iq = np.fromfile(path, dtype=np.complex64)

if iq.size == 0:
    raise ValueError("Capture file is empty")

i = iq.real
q = iq.imag
magnitude = np.abs(iq)

print("Samples:", iq.size)
print("Finite samples:", np.isfinite(iq).all())
print("Mean power:", float(np.mean(magnitude ** 2)))
print("Peak magnitude:", float(np.max(magnitude)))
print("I range:", float(i.min()), "to", float(i.max()))
print("Q range:", float(q.min()), "to", float(q.max()))
print("I near +/-1:", float(np.mean(np.abs(i) >= 0.999)))
print("Q near +/-1:", float(np.mean(np.abs(q) >= 0.999)))
print("Magnitude above 0.95:", float(np.mean(magnitude > 0.95)))
