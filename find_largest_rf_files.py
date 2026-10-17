
from pathlib import Path

root = Path("real_dataset")
files = [
    p for p in root.iterdir()
    if p.is_file()
]

files.sort(key=lambda p: p.stat().st_size, reverse=True)

print("10 largest files directly inside real_dataset:")
print("=" * 75)

for p in files[:10]:
    print(f"{p.stat().st_size / (1024**2):9.2f} MB  {p.name}")

print("\nPotential Zigbee/cellular metadata files:")
for p in root.rglob("*"):
    if p.is_file() and p.suffix.lower() in {
        ".json", ".sigmf", ".meta", ".txt", ".csv"
    }:
        if any(k in str(p).lower()
               for k in ("zigbee", "cellular", "lte", "5g", "metadata")):
            print(p)
