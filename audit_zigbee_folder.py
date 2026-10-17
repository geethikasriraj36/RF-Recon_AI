
from pathlib import Path
import tarfile

root = Path(r"real_dataset\zigbee\scenario6\scene38")

caps = list(root.rglob("*.cap"))
sigmf = list(root.rglob("*.sigmf"))
csv_files = list(root.rglob("*.csv"))

print("PCAP files:", len(caps))
print("Total PCAP bytes:", sum(f.stat().st_size for f in caps))
print("SigMF files:", len(sigmf))
print("CSV files:", len(csv_files))

nonempty_iq = 0
metadata_files = 0
errors = 0
examples = []

for f in sigmf:
    try:
        with tarfile.open(f) as archive:
            members = archive.getmembers()

            for member in members:
                if member.name.endswith(".sigmf-meta"):
                    metadata_files += 1

                if member.name.endswith(".sigmf-data"):
                    if member.size > 0:
                        nonempty_iq += 1
                        examples.append((f.name, member.size))
    except (tarfile.TarError, OSError):
        errors += 1

csv_with_rows = 0

for f in csv_files:
    try:
        lines = f.read_text(errors="replace").splitlines()
        if len(lines) > 1 and any(line.strip() for line in lines[1:]):
            csv_with_rows += 1
    except OSError:
        errors += 1

print("SigMF metadata members:", metadata_files)
print("Non-empty IQ data members:", nonempty_iq)
print("Examples of IQ files:", examples[:5])
print("CSV files containing data rows:", csv_with_rows)
print("Files that could not be inspected:", errors)
