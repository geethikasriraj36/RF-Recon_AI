
import iio

ctx = iio.Context("ip:192.168.2.1")
phy = ctx.find_device("ad9361-phy")

for channel_id in ("voltage0", "altvoltage0"):
    ch = next(
        (c for c in phy.channels if c.id == channel_id),
        None
    )
    if ch is None:
        print(f"\n{channel_id}: not found")
        continue

    print(f"\n{channel_id}:")
    for attr in (
        "sampling_frequency",
        "rf_bandwidth",
        "frequency",
        "gain_control_mode",
        "hardwaregain",
        "rf_port_select",
    ):
        try:
            print(f"  {attr}: {ch.attrs[attr].value}")
        except Exception:
            pass
