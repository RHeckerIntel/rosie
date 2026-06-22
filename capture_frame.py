# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyrealsense2",
#     "pillow",
#     "numpy",
# ]
# ///
"""
List connected RealSense cameras and capture a 480p PNG frame.

Usage:
    uv run capture_frame.py                        # capture from first camera
    uv run capture_frame.py --serial 123456789     # capture from specific camera
    uv run capture_frame.py --output my_frame.png
    uv run capture_frame.py --list                 # list devices only, no capture
"""

import argparse
import sys
import numpy as np
from PIL import Image

import pyrealsense2 as rs


def list_devices(ctx: rs.context) -> list[rs.device]:
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("No RealSense devices found.")
        return []

    print(f"Found {len(devices)} RealSense device(s):\n")
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        firmware = dev.get_info(rs.camera_info.firmware_version)
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        print(f"  [{i}] {name}")
        print(f"       Serial:   {serial}")
        print(f"       Firmware: {firmware}")
        print(f"       USB:      {usb}")

    return list(devices)


def capture_frame(serial: str | None, output: str) -> None:
    pipeline = rs.pipeline()
    config = rs.config()

    if serial:
        config.enable_device(serial)

    # 640x480 @ 30fps color
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    print("\nStarting camera...")
    pipeline.start(config)

    try:
        # Discard a few frames to let auto-exposure settle
        for _ in range(10):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("Failed to capture color frame.", file=sys.stderr)
            sys.exit(1)

        image = Image.fromarray(np.asarray(color_frame.get_data()))
        image.save(output)
        print(f"Saved {image.width}x{image.height} frame to: {output}")

    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description="Capture a 480p frame from a RealSense camera")
    parser.add_argument("--output", default="frame.png", help="Output PNG path")
    parser.add_argument("--serial", default=None, help="Camera serial number (default: first found)")
    parser.add_argument("--list", action="store_true", help="List devices and exit")
    args = parser.parse_args()

    ctx = rs.context()
    devices = list_devices(ctx)

    if args.list or not devices:
        return

    serial = args.serial
    if serial is None:
        serial = devices[0].get_info(rs.camera_info.serial_number)
        print(f"\nUsing first device (serial: {serial})")

    capture_frame(serial, args.output)


if __name__ == "__main__":
    main()
