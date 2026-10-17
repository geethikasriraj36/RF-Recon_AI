from pathlib import Path
import numpy as np
import pandas as pd
from rfrecon_consistent_features import extract_features, FEATURE_NAMES

DATA = Path("real_dataset/cellular_01_windows")
META = Path("real_dataset/cellular_01_windows_metadata.csv")
OUT = Path("real_dataset/cellular_01_candidate_features.csv")
SAMPLE_RATE = 20_000_000

if OUT.exists():
    raise SystemExit(f"Output already exists; preserving it: {OUT}")
meta = pd.read_csv(META)
files = sorted(DATA.glob("*.npy"))
if len(files) != 500 or len(meta) != 500:
    raise SystemExit("Expected exactly 500 windows and 500 metadata rows.")
rows = []
for f in files:
    x = np.load(f, mmap_mode="r")
    if x.shape != (4096,) or x.dtype != np.complex64:
        raise SystemExit(f"Unexpected format: {f}")
    features = extract_features(x, SAMPLE_RATE)
    row = {"filename": f.name, **features}
    row["label"] = "cellular_like_candidate"
    row["source_capture"] = "real_dataset/real_cellular_01.dat"
    rows.append(row)

df = pd.DataFrame(rows, columns=["filename", *FEATURE_NAMES, "label", "source_capture"])
if not np.isfinite(df[FEATURE_NAMES].to_numpy()).all():
    raise SystemExit("Non-finite features detected; output not written.")
df.to_csv(OUT, index=False)
print("Windows processed:", len(df))
print("Feature columns:", len(FEATURE_NAMES))
print("Feature matrix:", df.shape)
print("Saved:", OUT)
print("Label remains cellular_like_candidate; cellular identity is unverified.")
