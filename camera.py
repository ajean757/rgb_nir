#!/usr/bin/env python3
import gpiod
import threading
from datetime import datetime
import cv2
import numpy as np
from picamera2 import Picamera2
import concurrent.futures

CHIP = 'gpiochip0'
LINE_OFFSET = 26

chip = gpiod.Chip(CHIP)

line = chip.get_line(LINE_OFFSET)

flags = getattr(gpiod, "LINE_REQ_FLAG_BIAS_PULL_UP", 0) 
line.request(consumer="button", type=gpiod.LINE_REQ_EV_FALLING_EDGE, flags=flags)

print("Waiting for button press on GPIO26 (detecting falling edge)...")

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
stop_event = threading.Event()

def gpio_listener(cam0, cam1):
    while not stop_event.is_set():
        if line.event_wait(1):  # block until button press
            event = line.event_read()
            if event.type == gpiod.LineEvent.FALLING_EDGE:
                print("Button pressed! Capturing images...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                rgb_jpeg_filename = f"{timestamp}_rgb.jpg"
                rgb_raw_filename  = f"{timestamp}_rgb.dng"
                ir_jpeg_filename  = f"{timestamp}_ir.jpg"
                ir_raw_filename   = f"{timestamp}_ir.dng"

                executor.submit(capture_camera, cam0, ir_jpeg_filename, ir_raw_filename)
                executor.submit(capture_camera, cam1, rgb_jpeg_filename, rgb_raw_filename)

# Define capture function
def capture_camera(cam, jpeg_filename, raw_filename):
    """Capture JPEG + RAW from a single camera"""
    cam.capture_file(jpeg_filename)
    request = cam.capture_request()
    request.save_dng(raw_filename)
    request.release()
    print(f"Captured {jpeg_filename} and {raw_filename}")


def main():
    picam0 = Picamera2(camera_num=0)
    picam1 = Picamera2(camera_num=1)

    config = picam0.create_preview_configuration(
        main={"size": (2028, 1520), "format": "RGB888"},  # High-quality JPEG preview
        lores={"size": (640, 480), "format": "RGB888"},   # Lower-res preview for faster display
        raw={"format": "SBGGR10_CSI2P", "size": (4056, 3040)}  # RAW output
    )
    picam0.configure(config)
    picam1.configure(config)

    # Start both cameras
    picam0.start()
    picam1.start()

    print("Press 'q' to quit the live feed window...")
    
    t0 = threading.Thread(target=gpio_listener, args=(picam0,picam1,))
    t0.start()

    try:
        while True:
            frame0 = picam0.capture_array("lores")
            frame1 = picam1.capture_array("lores")
            combined = np.hstack((frame0, frame1))
            cv2.imshow("Dual Camera Feed", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        stop_event.set()
        picam0.stop()
        picam1.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete. Exiting...")

if __name__ == "__main__":
    main()
