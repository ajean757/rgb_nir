#!/usr/bin/env python3
import gpiod
import threading
from datetime import datetime
import cv2
import numpy as np
from picamera2 import Picamera2
import concurrent.futures

# Define the chip and the line offset
CHIP = 'gpiochip0'
LINE_OFFSET = 26

# Open the gpiochip device
chip = gpiod.Chip(CHIP)

# Get the line corresponding to GPIO26
line = chip.get_line(LINE_OFFSET)

flags = getattr(gpiod, "LINE_REQ_FLAG_BIAS_PULL_UP", 0)  # Use flag if available
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
                # Start two threads to capture images concurrently.
                # Capture images one after the other

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Define filenames for RGB and IR images
                rgb_jpeg_filename = f"{timestamp}_rgb.jpg"
                rgb_raw_filename  = f"{timestamp}_rgb.dng"
                ir_jpeg_filename  = f"{timestamp}_ir.jpg"
                ir_raw_filename   = f"{timestamp}_ir.dng"

                executor.submit(capture_camera, cam0, ir_jpeg_filename, ir_raw_filename)
                executor.submit(capture_camera, cam1, rgb_jpeg_filename, rgb_raw_filename)
                # t1 = threading.Thread(target=capture_camera, args=(cam0,ir_jpeg_filename,ir_raw_filename,))
                # t2 = threading.Thread(target=capture_camera, args=(cam1,rgb_jpeg_filename,rgb_raw_filename,))
                # t1.start()
                # t2.start()
                # t1.join()
                # t2.join() 

# Define capture function
def capture_camera(cam, jpeg_filename, raw_filename):
    """Capture JPEG + RAW from a single camera"""
    cam.capture_file(jpeg_filename)  # Save JPEG
    request = cam.capture_request()  # Get RAW capture
    request.save_dng(raw_filename)   # Save RAW as DNG
    request.release()                # Release request
    print(f"Captured {jpeg_filename} and {raw_filename}")


def main():
    # Create Picamera2 instances for two cameras
    # If you only have ONE camera, just create picam0
    picam0 = Picamera2(camera_num=0)
    picam1 = Picamera2(camera_num=1)

    # Configure each camera for preview
    # Create configuration:
    config = picam0.create_preview_configuration(
        main={"size": (4056, 3040), "format": "RGB888"},  # High-quality JPEG preview
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
            # Capture frames as NumPy arrays
            frame0 = picam0.capture_array("lores")
            frame1 = picam1.capture_array("lores")
            # Combine frames horizontally: side by side
            combined = np.hstack((frame0, frame1))
            # Show the combined feed in an OpenCV window
            cv2.imshow("Dual Camera Feed", combined)
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        stop_event.set()  # Stop the GPIO listener
        picam0.stop()
        picam1.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete. Exiting...")

if __name__ == "__main__":
    main()
