from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path("real_dataset/real_cellular_01.dat")
OUT = Path("real_dataset/cellular_01_windows")
META = Path("real_dataset/cellular_01_windows_metadata.csv")
WINDOW = 4096
COUNT = 500
SAMPLE_RATE_HZ = 20_000_000
CENTER_FREQ_HZ = 1_872_000_000

if META.exists() or any(OUT.glob("*.npy")):
    raise SystemExit("Output already exists; stopping to protect files.")
if SOURCE.stat().st_size % 8:
    raise SystemExit("File size is not divisible by 8; stopping.")

x = np.memmap(SOURCE, dtype=np.complex64, mode="r")
if len(x) < COUNT * WINDOW:
    raise SystemExit("Not enough samples.")

OUT.mkdir(parents=True, exist_ok=True)
starts = np.linspace(0, len(x)-WINDOW, COUNT, dtype=np.int64)
rows = []
for i, start in enumerate(starts, 1):
    chunk = np.asarray(x[start:start+WINDOW])
    if not np.isfinite(chunk).all():
        raise SystemExit(f"Non-finite sample in window {i}; stopping.")
    name = f"cellular01_{i:04d}.npy"
    np.save(OUT/name, chunk)
    rows.append({"filename": name, "label": "cellular_like_candidate", "source_capture": str(SOURCE), "start_sample": int(start), "window_size": WINDOW, "configured_sample_rate_hz": SAMPLE_RATE_HZ, "configured_center_frequency_hz": CENTER_FREQ_HZ, "iq_format_assumption": "complex64; not independently verified", "signal_class_status": "cellular not independently verified"})

pd.DataFrame(rows).to_csv(META, index=False)
print("Windows saved:", len(rows))
print("Output folder:", OUT)
print("Metadata:", META)
print("Original capture unchanged.")
