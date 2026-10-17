
import zipfile
import tarfile
import io

zip_path = r"real_dataset\zigbee_ble_dataset.zip"

with zipfile.ZipFile(zip_path) as z:
    files = [
        p for p in z.namelist()
        if p.startswith("zigbee/") and p.endswith(".sigmf")
    ]

    count = 0
    print("Checking", len(files), "Zigbee recordings...")

    for p in files:
        try:
            data = z.read(p)

            with tarfile.open(
                fileobj=io.BytesIO(data), mode="r:"
            ) as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".sigmf-data"):
                        if member.size > 0:
                            print(
                                "IQ DATA FOUND:",
                                member.size,
                                "bytes |",
                                p
                            )
                            count += 1
        except (tarfile.TarError, OSError):
            pass

    print("Recordings with non-empty IQ data:", count)
