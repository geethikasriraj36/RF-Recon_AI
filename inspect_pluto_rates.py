
import iio

ctx = iio.Context("ip:192.168.2.1")
phy = ctx.find_device("ad9361-phy")

for channel_name, output in [
    ("voltage0", "RX channel"),
    ("altvoltage0", "RX LO"),
]:
    channel = phy.find_channel(
        channel_name,
        channel_name == "altvoltage0"
    )
    print(f"\n--- {output} ---")

    if channel is None:
        print("Channel not found")
        continue

    for attr_name in [
        "sampling_frequency",
        "sampling_frequency_available",
        "sampling_frequency_available",
        "rf_bandwidth",
        "frequency",
        "gain_control_mode",
    ]:
        try:
            print(f"{attr_name}: {channel.attrs[attr_name].value}")
        except (KeyError, OSError):
            pass

print("\n--- RX device ---")
rx = ctx.find_device("cf-ad9361-lpc")
print("RX device found:", rx is not None)
