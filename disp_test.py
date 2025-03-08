#!/usr/bin/env python3
import gpiod
import threading
from datetime import datetime, timedelta
import cv2
import numpy as np
from picamera2 import Picamera2
import concurrent.futures
import time
import spidev
from PIL import Image
from lib import LCD_2inch4


# import debugpy

# # Ensure debugpy only starts once
# if not debugpy.is_client_connected():
#     debugpy.listen(("0.0.0.0", 5678))
#     print("Waiting for debugger to attach...")
#     debugpy.wait_for_client()  # Pause execution until VSCode attaches
# print("Debugger attached. Running script...")


# -----------------------------
# Global definitions and setup
# -----------------------------
# Button setup on GPIO26
BUTTON_OFFSET = 16
CHIP_NAME = "/dev/gpiochip0"

# Create request for button with pull-up and edge detection
button_request = gpiod.request_lines(
    CHIP_NAME,
    consumer="button-app",
    config={
        BUTTON_OFFSET: gpiod.LineSettings(
            direction=gpiod.line.Direction.INPUT,
            edge_detection=gpiod.line.Edge.FALLING,
            bias=gpiod.line.Bias.PULL_UP
        )
    }
)
print("Waiting for button press on GPIO26 (detecting falling edge)...")

# Threading executor and stop event for button listener
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
stop_event = threading.Event()


# ---------------------------
# Camera capture functions
# ---------------------------
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
    print(f"Metadata: {metadata}")
    return start_time

def gpio_listener(cam0, cam1):
    last_press_time = 0
    debounce_time = 0.05 # 1 second debounce period
    
    while not stop_event.is_set():
        # Wait for events with a 1 second timeout
        if button_request.wait_edge_events(timeout=timedelta(seconds=1)):
            events = button_request.read_edge_events(max_events=1)
            print(f"Length of events: {len(events)}")
            current_time = time.time()
            event = events[0]
            # Check if this is a valid button press (debounce)
            if (event.line_offset == BUTTON_OFFSET and 
                event.event_type == gpiod.EdgeEvent.Type.FALLING_EDGE and
                (current_time - last_press_time) > debounce_time):
                
                last_press_time = current_time
                print("Button pressed! Capturing images...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dir = "./imagestest"
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
            else:
                # Clear any additional events that came in during the debounce period
                button_request.read_edge_events()
                
                


# -------------------
# Main function loop
# -------------------
def main():
    # Initialize LCD
    # init_lcd()
    disp = LCD_2inch4.LCD_2inch4()
    # Initialize library.
    disp.Init()
    # Clear display.
    disp.clear()
    #Set the backlight to 100
    disp.bl_DutyCycle(50)

    # Set up cameras
    picam0 = Picamera2(camera_num=0)
    picam1 = Picamera2(camera_num=1)

    config_rgb = picam1.create_preview_configuration(
        main={"size": (2028, 1520), "format": "RGB888"},
        lores={"size": (640, 480), "format": "RGB888"},
        raw={"format": "SBGGR12_CSI2P", "size": (4056, 3040)}
    )
    config_ir = picam0.create_preview_configuration(
        main={"size": (2028, 1520), "format": "RGB888"},
        lores={"size": (640, 480), "format": "RGB888"},
        raw={"unpacked": "R12", "size": (4056, 3040)}
    )
    picam0.configure(config_ir)
    picam1.configure(config_rgb)

    print(f"Cam 0 (IR) sensor format: {picam0.sensor_format}")
    print(f"Cam1 (RGB) sensor format: {picam1.sensor_format}")
    print(f"RGB Camera Properties: {picam1.camera_configuration()['raw']}")
    print(f"IR Camera Properties: {picam0.camera_configuration()['raw']}")

    # Start cameras
    picam0.start()
    picam1.start()

    print("Press 'q' to quit the live feed loop...")
    
    # Start button listener thread
    t0 = threading.Thread(target=gpio_listener, args=(picam0, picam1,))
    t0.daemon = True  # Mark as daemon so it doesn't block shutdown

    try:
        t0.start()
        while True:
            # Capture low-resolution frames from each camera
            frame0 = picam0.capture_array("lores")
            frame1 = picam1.capture_array("lores")

            # Resize each frame individually before combining to maintain aspect ratio
            # For 240x320 display, each image should be 240x160 (half the height)
            frame0_pil = Image.fromarray(frame0, 'RGB')
            frame1_pil = Image.fromarray(frame1, 'RGB')
            
            # Resize maintaining aspect ratio
            frame0_resized = frame0_pil.resize((240, 320), Image.LANCZOS)
            frame1_resized = frame1_pil.resize((240, 320), Image.LANCZOS)

            frame0_resized = frame0_resized.rotate(90)
            frame1_resized = frame1_resized.rotate(90)

            # Calculate crop coordinates to get center of images
            # For 240x320 images, get the center 160 vertical pixels
            crop_top = (320 - 160) // 2  # 80
            crop_height = 160

            # Crop images to center 160 pixels vertically
            frame0_cropped = frame0_resized.crop((0, crop_top, 240, crop_top + crop_height))
            frame1_cropped = frame1_resized.crop((0, crop_top, 240, crop_top + crop_height))
            # Create a blank 240x320 canvas (white background)
            combined_img = Image.new('RGB', (240, 320), color=(0, 0, 0))
            
            # Place the two images vertically on the canvas
            combined_img.paste(frame0_cropped, (0, 0))
            combined_img.paste(frame1_cropped, (0, 160))
            
            # For LCD_2inch4, proper rotation might be 0 or 180, try both
            pil_image = combined_img.rotate(180)
            
            # Send to display
            frame1_resized = frame1_resized.rotate(180)
            disp.ShowImage(pil_image)

            # Allow a short delay and key detection for graceful exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        picam0.stop()
        picam1.stop()
        cv2.destroyAllWindows()
        disp.module_exit()
        button_request.release()  # Release GPIO resources
        print("Cleanup complete. Exiting...")

if __name__ == "__main__":
    main()
