import struct
from pathlib import Path

path = Path(r"real_dataset\6lowpan_ping.pcap")
data = path.read_bytes()

if data[:4] == bytes.fromhex("d4c3b2a1"):
    endian = "<"
elif data[:4] == bytes.fromhex("a1b2c3d4"):
    endian = ">"
else:
    raise ValueError("Unknown PCAP magic number")

link_type = struct.unpack(endian + "I", data[20:24])[0]
print("Link type:", link_type)

offset = 24
packet_number = 0

while offset + 16 <= len(data):
    ts_sec, ts_frac, captured, original = struct.unpack(
        endian + "IIII", data[offset:offset + 16]
    )
    offset += 16

    if captured > len(data) - offset:
        print("Incomplete packet record; stopping.")
        break

    packet = data[offset:offset + captured]
    offset += captured
    packet_number += 1

    print(
        f"Packet {packet_number}: captured={captured}, "
        f"original={original}, hex={packet.hex(' ')}"
    )

print("Total complete packets:", packet_number)
print("Trailing bytes:", len(data) - offset)