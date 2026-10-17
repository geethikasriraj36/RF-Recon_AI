
from pathlib import Path
import numpy as np
import tarfile
import json

root = Path("real_dataset")

print("=== RAW .DAT FORMAT CHECK ===")

names = [
    "real_cellular_scan_01.dat",
    "real_cellular_01.dat",
    "real_wifi_01.dat",
    "real_bluetooth_01.dat",
    "real_noise_01.dat",
]

for name in names:
    path = root / name
    if not path.exists():
        continue

    # Read only the first 4096 float32 values
    with path.open("rb") as f:
        raw = f.read(4096 * 4)

    values = np.frombuffer(raw, dtype="<f4")
    finite = np.isfinite(values)

    print(f"\n{name}")
    print("  Float32 values:", len(values))
    print("  All finite:", bool(finite.all()))

    if finite.any():
        v = values[finite]
        print("  Min / max:", float(v.min()), float(v.max()))
        print("  Mean:", float(v.mean()))
        print("  Std dev:", float(v.std()))

        # Tentative interpretation as interleaved I/Q
        if len(v) >= 2 and len(v) % 2 == 0:
            iq = v[:len(v) // 2 * 2].reshape(-1, 2)
            z = iq[:, 0] + 1j * iq[:, 1]
            print("  Tentative I/Q mean power:",
                  float(np.mean(np.abs(z) ** 2)))

print("\n=== CELLULAR NPY FORMAT CHECK ===")

npy_path = root / "cellular_like" / "cellular_0001.npy"

if npy_path.exists():
    arr = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    print("Shape:", arr.shape)
    print("Data type:", arr.dtype)
    print("Complex data:", np.iscomplexobj(arr))
    print("First 8 values:", arr.reshape(-1)[:8])
else:
    print("cellular_0001.npy not found")

print("\n=== ZIGBEE ARCHIVE METADATA CHECK ===")

shown = 0
zigbee = root / "zigbee"

if zigbee.exists():
    for archive_path in zigbee.rglob("*.sigmf"):
        try:
            with tarfile.open(archive_path, "r:*") as archive:
                for member in archive.getmembers():
                    if member.name.endswith(".sigmf-meta"):
                        stream = archive.extractfile(member)
                        if stream is None:
                            continue

                        metadata = json.loads(
                            stream.read().decode(
                                "utf-8", errors="replace"
                            )
                        )
                        print("\nArchive:", archive_path.name)
                        print(json.dumps(metadata, indent=2)[:1200])
                        shown += 1
                        break
        except (tarfile.TarError, OSError, ValueError) as e:
            print("Archive read error:", e)

        if shown >= 2:
            break

    if shown == 0:
        print("No readable metadata found in sampled archives.")
