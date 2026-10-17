
from pathlib import Path

root = Path("real_dataset")

if not root.exists():
    raise FileNotFoundError("real_dataset folder was not found")

print("RFRecon capture-file audit")
print("=" * 55)

files = sorted(p for p in root.rglob("*") if p.is_file())

if not files:
    print("No files found.")
else:
    for path in files:
        size = path.stat().st_size
        print(f"{path} | {size:,} bytes | {'EMPTY' if size == 0 else 'OK'}")

print("\nTotal files:", len(files))
print("Total size (MB):", round(
    sum(p.stat().st_size for p in files) / (1024 ** 2), 2
))
