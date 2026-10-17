
from pathlib import Path
import csv
import tarfile

root = Path("real_dataset")

print("=== LARGE .dat FILES ===")
for name in (
    "real_cellular_scan_01.dat",
    "real_cellular_01.dat",
    "real_wifi_01.dat",
    "real_bluetooth_01.dat",
    "real_noise_01.dat",
):
    p = root / name
    if not p.exists():
        print(f"{name}: NOT FOUND")
        continue

    with p.open("rb") as f:
        header = f.read(32)

    size = p.stat().st_size
    print(f"\n{name}")
    print(f"  Size: {size:,} bytes")
    print(f"  Size mod 4: {size % 4}")
    print(f"  Size mod 8: {size % 8}")
    print(f"  First 32 bytes (hex): {header.hex(' ')}")

print("\n=== CELLULAR NUMPY FILES ===")
for name in ("cellular_0001.npy", "cellular_0002.npy"):
    matches = list(root.rglob(name))
    if not matches:
        print(f"{name}: NOT FOUND")
    else:
        for p in matches[:3]:
            print(f"{p} ({p.stat().st_size:,} bytes)")
            with p.open("rb") as f:
                print("  Header:", f.read(16).hex(" "))

print("\n=== ZIGBEE ARCHIVE CHECK ===")
zigbee = root / "zigbee"
shown = 0

if zigbee.exists():
    for p in zigbee.rglob("*.sigmf"):
        print(f"\n{p.relative_to(root)}")
        try:
            with tarfile.open(p, "r:*") as archive:
                for member in archive.getmembers()[:6]:
                    print(
                        f"  {member.name} "
                        f"({member.size:,} bytes)"
                    )
                print("  Archive: readable")
        except (tarfile.TarError, OSError) as e:
            print(f"  Not a readable TAR archive: {e}")

        shown += 1
        if shown >= 2:
            break

print("\n=== CELLULAR CSV SAMPLE COUNT ===")
p = root / "real_cellular_dataset.csv"
if p.exists():
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        count = 0
        labels = {}
        for row in reader:
            count += 1
            label = row.get("label", "unknown")
            labels[label] = labels.get(label, 0) + 1
        print("Rows:", count)
        print("Labels:", labels)
