
from pathlib import Path
from collections import defaultdict

root = Path("real_dataset")
groups = defaultdict(lambda: {"files": 0, "bytes": 0, "empty": 0})

for p in root.rglob("*"):
    if not p.is_file():
        continue

    # Ignore macOS metadata and resource-fork files.
    if "__MACOSX" in p.parts or p.name == ".DS_Store" or p.name.startswith("._"):
        continue

    rel = p.relative_to(root)
    parts = rel.parts
    group = parts[0] if len(parts) > 1 else "(root files)"

    size = p.stat().st_size
    groups[group]["files"] += 1
    groups[group]["bytes"] += size
    groups[group]["empty"] += int(size == 0)

print("RFRecon dataset summary")
print("=" * 65)

for name, data in sorted(groups.items()):
    mb = data["bytes"] / (1024 * 1024)
    print(
        f"{name:25} files={data['files']:5} "
        f"size={mb:9.2f} MB empty={data['empty']}"
    )

print("\nTotal size:",
      round(sum(x["bytes"] for x in groups.values()) / (1024**2), 2),
      "MB")
