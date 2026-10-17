
from pathlib import Path
import json

root = Path("real_dataset")

print("=== Cellular metadata ===")
for name in (
    "real_cellular_dataset.csv",
    "real_cellular_features.csv",
    "real_cellular_model_predictions.csv",
):
    path = root / name
    if path.exists():
        print(f"\n--- {name} ---")
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            for _, line in zip(range(5), f):
                print(line.rstrip()[:250])

print("\n=== Large IQ file sidecars ===")
for stem in (
    "real_cellular_scan_01",
    "real_cellular_01",
    "real_wifi_01",
    "real_bluetooth_01",
    "real_noise_01",
):
    print(f"\n--- {stem} ---")
    for suffix in (".sigmf-meta", ".json", ".txt", ".meta", ".csv"):
        p = root / (stem + suffix)
        if p.exists():
            print(f"Found: {p.name} ({p.stat().st_size:,} bytes)")
            if suffix in (".json", ".sigmf-meta", ".meta", ".txt"):
                try:
                    print(p.read_text(
                        encoding="utf-8", errors="replace"
                    )[:1500])
                except OSError as e:
                    print("Read error:", e)

print("\n=== Zigbee metadata sample ===")
zigbee = root / "zigbee"
if zigbee.exists():
    shown = 0
    for p in zigbee.rglob("*.sigmf"):
        print(f"\n--- {p.relative_to(root)} ---")
        try:
            print(p.read_text(encoding="utf-8", errors="replace")[:1000])
        except OSError as e:
            print("Read error:", e)
        shown += 1
        if shown >= 2:
            break
