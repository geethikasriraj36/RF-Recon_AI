import os
import tarfile

ROOT = r"real_dataset\zigbee\scenario6\scene38"

print("=" * 70)
print("RFRecon - SIGMF RECORDING TYPE INSPECTION")
print("=" * 70)

counts = {}
matches = []

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if not file.lower().endswith(".sigmf"):
            continue

        filepath = os.path.join(root, file)

        try:
            with tarfile.open(filepath, "r") as tar:
                meta_files = [
                    m for m in tar.getmembers()
                    if m.name.lower().endswith(".sigmf-meta")
                ]

                if not meta_files:
                    continue

                meta = tar.extractfile(meta_files[0]).read().decode(
                    "utf-8", errors="ignore"
                )

                description = ""

                if '"core:description"' in meta:
                    start = meta.find('"core:description"')
                    description = meta[start:start + 300]

                desc_lower = description.lower()

                if "zigbee" in desc_lower or "802.15.4" in desc_lower:
                    signal_type = "ZIGBEE / 802.15.4"
                    matches.append(filepath)

                elif "ble" in desc_lower or "bluetooth" in desc_lower:
                    signal_type = "BLE"

                else:
                    signal_type = "UNKNOWN"

                counts[signal_type] = counts.get(signal_type, 0) + 1

        except Exception as e:
            print("ERROR:", filepath)
            print(e)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for key, value in counts.items():
    print(f"{key}: {value}")

print("\n" + "=" * 70)
print("ZIGBEE RECORDINGS FOUND")
print("=" * 70)

if matches:
    for path in matches:
        print(path)
else:
    print("NO ZIGBEE / 802.15.4 RECORDINGS FOUND")

print("\n" + "=" * 70)