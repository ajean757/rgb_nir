#!/usr/bin/env python3
import gpiod
import threading
from datetime import datetime, timedelta
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform, ColorSpace
import concurrent.futures
import time
import spidev
from PIL import Image
from lib import LCD_2inch, LCD_2inch4
import os


import debugpy

# Ensure debugpy only starts once
if not debugpy.is_client_connected():
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()  # Pause execution until VSCode attaches
print("Debugger attached. Running script...")



# -----------------------------
# Global definitions and setup
# -----------------------------

class SharedState:
    capture_notification = False
    notification_start_time = 0
    notification_duration = 1.5  # Show notification for 1.5 seconds

shared_state = SharedState()


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
    debounce_time = 0.05  # Increase to 500ms (0.5 seconds)
    capture_in_progress = False
    
    while not stop_event.is_set():
        try:
            # Skip event processing if capturing is in progress
            if capture_in_progress:
                time.sleep(0.05)
                continue
                
            # Wait for events with a 1 second timeout
            if button_request.wait_edge_events(timeout=timedelta(seconds=1)):
                events = button_request.read_edge_events(max_events=1)
                
                # Process only if events exist
                if events:
                    current_time = time.time()
                    event = events[0]
                    
                    # Check if this is a valid button press (debounce)
                    if (event.line_offset == BUTTON_OFFSET and 
                        event.event_type == gpiod.EdgeEvent.Type.FALLING_EDGE and
                        (current_time - last_press_time) > debounce_time):
                        
                        # Set flag to prevent further captures while processing
                        capture_in_progress = True
                        last_press_time = current_time
                        
                        # Clear any extra events BEFORE processing
                        button_request.read_edge_events()
                        
                        print("Button pressed! Capturing images...")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dir = "./imagestest"
                        if not os.path.exists(dir):
                            os.makedirs(dir)
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
                        
                        # send notification flag 
                        shared_state.capture_notification = True
                        shared_state.notification_start_time = time.time()
                        # Release capture flag but keep debounce time
                        capture_in_progress = False
                    else:
                        # Clear any additional events that came in during the debounce period
                        button_request.read_edge_events()
        
        except Exception as e:
            print(f"Error in GPIO listener: {e}")
            time.sleep(0.5)
                


# -------------------
# Main function loop
# -------------------
def main():
    # Initialize LCD
    disp = LCD_2inch.LCD_2inch()
    disp.Init()
    disp.clear()
    disp.bl_DutyCycle(50)

    # Set up cameras
    picam0 = Picamera2(camera_num=0)
    picam1 = Picamera2(camera_num=1)

    picam0.set_controls({"AwbEnable":False, "ColourGains": (0,0)})

    config_rgb = picam1.create_preview_configuration(
        main={"size": (2028, 1520), "format": "RGB888"},
        lores={"size": (320, 240), "format": "RGB888"},
        raw={"format": "SBGGR12_CSI2P", "size": (4056, 3040)},
        transform=Transform(vflip=True, hflip=True),
        colour_space=ColorSpace.Raw()
    )
    config_ir = picam0.create_preview_configuration(
        main={"size": (2028, 1520), "format": "RGB888"},
        lores={"size": (320, 240), "format": "RGB888"},
        raw={"format": "R12", "size": (4056, 3040)},
        transform=Transform(vflip=True, hflip=True),
        colour_space=ColorSpace.Raw()
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

            # Convert to BGR to RGB:
            frame0_corrected = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
            frame1_corrected = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

            # Resize each frame individually before combining to maintain aspect ratio
            # For 240x320 display, each image should be 240x160 (half the height)
            frame0_pil = Image.fromarray(frame0_corrected, 'RGB')
            frame1_pil = Image.fromarray(frame1_corrected, 'RGB')

            # Calculate crop coordinates to get center of images
            # For 240x320 images, get the center 160 vertical pixels
            crop_sides = (320 - 160) // 2  # 80
            crop_height = 240

            # Crop images to center 160 pixels vertically
            frame0_cropped = frame0_pil.crop((crop_sides, 0, crop_height, crop_height))
            frame1_cropped = frame1_pil.crop((crop_sides, 0, crop_height, crop_height))
            # Create a blank 240x320 canvas (white background)
            combined_img = Image.new('RGB', (320, 240), color=(0, 0, 0))
            
            # Place the two images vertically on the canvas
            combined_img.paste(frame0_cropped, (0, 0))
            combined_img.paste(frame1_cropped, (160, 0))

            current_time = time.time()
            if shared_state.capture_notification and (current_time - shared_state.notification_start_time) < shared_state.notification_duration:
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(combined_img)

                border_width = 5
                draw.rectangle([(0, 0), (320, 240)], outline=(255, 0, 0), width=border_width)
                # Try to use a larger font
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
                except:
                    # Fallback to default font
                    font = ImageFont.load_default()

                text = "Captured!"

                # Calculate text dimensions for centering
                text_bbox = draw.textbbox((0, 0), text,font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                text_x = (320 - text_width) // 2
                text_y = (240 - text_height) // 2

                # Draw the text
                draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

                # Draw a progress bar
                elapsed = current_time - shared_state.notification_start_time
                progress = elapsed / shared_state.notification_duration
                bar_width = int(240 * (1 - progress))
                draw.rectangle([(0, 235), (bar_width, 240)], fill=(0, 255, 0))
            elif shared_state.capture_notification:
                shared_state.capture_notification = False

            # Send to display
            combined_img = combined_img.rotate(90, expand=True)
            disp.ShowImage(combined_img)
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
