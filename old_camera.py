#!/usr/bin/env python3
import gpiod
import threading
from datetime import datetime
import cv2
import numpy as np
from picamera2 import Picamera2
import concurrent.futures
import time

CHIP = 'gpiochip0'
LINE_OFFSET = 16

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
                dir = "./images"
                rgb_jpeg_filename = f"{dir}/{timestamp}_rgb.jpg"
                rgb_raw_filename  = f"{dir}/{timestamp}_rgb.dng"
                ir_jpeg_filename  = f"{dir}/{timestamp}_ir.jpg"
                ir_raw_filename   = f"{dir}/{timestamp}_ir.dng"

                ir_task = executor.submit(capture_camera, cam0, ir_jpeg_filename, ir_raw_filename)
                rgb_task = executor.submit(capture_camera, cam1, rgb_jpeg_filename, rgb_raw_filename)

                ir_timestamp = ir_task.result()
                rgb_timestamp = rgb_task.result()

                time_diff = abs(ir_timestamp - rgb_timestamp)
                print(f"Time difference between camera captures {time_diff:.6f} seconds")

# # Define capture function
# def capture_camera(cam, jpeg_filename, raw_filename):
#     """Capture JPEG + RAW from a single camera"""
#     start_time = time.perf_counter()

#     cam.capture_file(jpeg_filename)
#     request = cam.capture_request()
#     if "raw" in request:
#         request.save_dng(raw_filename)
#     else:
#         print(f"No raw found for {raw_filename}")
#     request.release()
#     print(f"Captured {jpeg_filename} and {raw_filename}")
#     end_time = time.perf_counter()
#     print(f"[{start_time:.6f}] Captured {jpeg_filename} on Camera {cam.camera_idx}")
#     print(f"[{end_time:.6f}] Capture duration: {end_time - start_time:.6f} seconds")
#     return start_time

def capture_camera(cam, jpeg_filename, raw_filename):
    """Capture JPEG + RAW from a single camera"""
    start_time = time.perf_counter()

    cam.capture_file(jpeg_filename)

    request = cam.capture_request()
    metadata = request.get_metadata()  # Get debugging info

    if cam.camera_config and "raw" in cam.camera_config:  # Check if raw is configured
        try:
            print(f"Saving RAW image: {raw_filename}")
            request.save_dng(raw_filename)
        except Exception as e:
            print(f"ERROR saving RAW: {e}")
    else:
        print(f"WARNING: RAW stream is not configured for {raw_filename}")

    request.release()

    end_time = time.perf_counter()
    print(f"[{start_time:.6f}] Captured {jpeg_filename} on Camera {cam.camera_idx}")
    print(f"[{end_time:.6f}] Capture duration: {end_time - start_time:.6f} seconds")
    print(f"Metadata: {metadata}")  # Debugging info

    return start_time



def main():
    picam0 = Picamera2(camera_num=0)
    picam1 = Picamera2(camera_num=1)

    config_rgb = picam1.create_preview_configuration(
        main={"size": (2028, 1520), "format": "RGB888"},  # High-quality JPEG preview
        lores={"size": (640, 480), "format": "RGB888"},   # Lower-res preview for faster display
        raw={"format": "SBGGR12_CSI2P", "size": (4056, 3040)}  # RAW output
    )
    config_ir = picam0.create_preview_configuration(
        main={"size": (2028, 1520), "format": "RGB888"},
	lores={"size":(640, 480), "format": "RGB888"},
	raw={"unpacked": "R12", "size": (4056, 3040)}
    )
    picam0.configure(config_ir)
    picam1.configure(config_rgb)

    # picam0.camera_controls["ColourGains"] = (1.0,1.0)
    print(f"Cam 0 (IR) sensor format: {picam0.sensor_format}")
    print(f"Cam1 (RGB) sensor format: {picam1.sensor_format}")
    print(f"RGB Camera Properties: {picam1.camera_configuration()['raw']}")
    print(f"IR Camera Properties: {picam0.camera_configuration()['raw']}")

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
