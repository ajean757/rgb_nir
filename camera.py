#!/usr/bin/env python3
import gpiod
import threading
from datetime import datetime, timedelta
import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import Transform, ColorSpace, controls
import concurrent.futures
import time
from PIL import Image, ImageDraw, ImageFont
from lib import LCD_2inch
import os
from enum import Enum


SAVE_DIR = "./two_rgb"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

for key, value in os.environ.items():
    print(f"{key}={value}")

class DisplayState(Enum):
    SPLIT = 0
    RGB_ONLY = 1
    IR_ONLY = 2

class SharedState:
    capture_notification = False
    notification_start_time = 0
    notification_duration = 1 # Seconds
    display_state = DisplayState.SPLIT

shared_state = SharedState()

BUTTON_OFFSET = 16
SWITCH_VIEW_OFFSET = 26
CHIP_NAME = "/dev/gpiochip0"

button_request = gpiod.request_lines(
    CHIP_NAME,
    consumer="button-app",
    config={
        BUTTON_OFFSET: gpiod.LineSettings(
            direction=gpiod.line.Direction.INPUT,
            edge_detection=gpiod.line.Edge.RISING,
            bias=gpiod.line.Bias.PULL_DOWN
        )
    }
)

switch_view_request = gpiod.request_lines(
    CHIP_NAME,
    consumer="button-app",
    config={
        SWITCH_VIEW_OFFSET: gpiod.LineSettings(
            direction=gpiod.line.Direction.INPUT,
            edge_detection=gpiod.line.Edge.RISING,
            bias=gpiod.line.Bias.PULL_DOWN
        )
    }
)

# executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
stop_event = threading.Event()

def capture_camera(cam, jpeg_filename, raw_filename, future_time):
    """Capture JPEG + RAW from a single camera"""
    start_time = time.perf_counter()
    request = cam.capture_request(flush=future_time)
    request.save("main", jpeg_filename)
    request.save_dng(raw_filename)
    metadata = request.get_metadata()
    request.release()
    end_time = time.perf_counter()
    print(f"[{start_time:.6f}] Captured {jpeg_filename} on Camera {cam.camera_idx}")
    print(f"[{end_time:.6f}] Capture duration: {end_time - start_time:.6f} seconds")
    print(f"Metadata: {metadata}")
    return metadata

def gpio_listener(cam0, cam1):
    last_press_time = 0
    debounce_time = 1
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
                        event.event_type == gpiod.EdgeEvent.Type.RISING_EDGE and
                        (current_time - last_press_time) > debounce_time):
                        
                        # Set flag to prevent further captures while processing
                        capture_in_progress = True
                        last_press_time = current_time
                        
                        print("Button pressed! Capturing images...")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dir = SAVE_DIR
                        rgb_jpeg_filename = f"{dir}/{timestamp}_rgb.jpg"
                        rgb_raw_filename  = f"{dir}/{timestamp}_rgb.dng"
                        ir_jpeg_filename  = f"{dir}/{timestamp}_ir.jpg"
                        ir_raw_filename   = f"{dir}/{timestamp}_ir.dng"

                        # barrier = threading.Barrier(2)
                        # rgb_task = executor.submit(capture_camera, cam1, rgb_jpeg_filename, rgb_raw_filename, barrier)
                        # ir_task = executor.submit(capture_camera, cam0, ir_jpeg_filename, ir_raw_filename, barrier)

                        # rgb_timestamp = rgb_task.result()
                        # ir_timestamp = ir_task.result()
                        # time_diff = abs(ir_timestamp - rgb_timestamp)
                        # print(f"Time difference between camera captures {time_diff:.6f} seconds")
                        
                        sync = cam1.capture_sync_request()
                        future_time = time.monotonic_ns() + 33_000_000  # 100 ms = 100,000,000 ns

                        # rgb_task = executor.submit(capture_camera, cam1, rgb_jpeg_filename, rgb_raw_filename, future_time)
                        # ir_task = executor.submit(capture_camera, cam0, ir_jpeg_filename, ir_raw_filename, future_time)

                        request_rgb = cam1.capture_request(flush=future_time)
                        request_ir = cam0.capture_request(flush=future_time)

                        request_rgb.save("main", rgb_jpeg_filename)
                        request_rgb.save_dng(rgb_raw_filename)
                        metadata_rgb = request_rgb.get_metadata()

                        request_ir.save("main", ir_jpeg_filename)
                        request_ir.save_dng(ir_raw_filename)
                        metadata_ir = request_ir.get_metadata()

                        request_rgb.release()
                        request_ir.release()
                        # metadata_rgb = rgb_task.result()
                        # metadata_ir = ir_task.result()
                        
                        sync.release()
                        print(f"Metadata: {metadata_rgb}")
                        print(f"Metadata: {metadata_ir}")
                        print(f"SensorTimestamp difference in ms: {abs(metadata_ir['SensorTimestamp'] - metadata_rgb['SensorTimestamp']) / 1e6:.6f} ms")
                        rgb_exposure_start_time = metadata_rgb['SensorTimestamp'] - 1000 * metadata_rgb['ExposureTime']
                        ir_exposure_start_time = metadata_ir['SensorTimestamp'] - 1000 * metadata_ir['ExposureTime']
                        print(f"difference in exposure start time in ms: {abs(ir_exposure_start_time - rgb_exposure_start_time) / 1e6:.6f} ms")
                       
                        # send notification flag 
                        shared_state.capture_notification = True
                        shared_state.notification_start_time = time.time()
                        # Release capture flag but keep debounce time
                        capture_in_progress = False
                    else:
                        # Clear any additional events that came in during the debounce period
                        button_request.read_edge_events()
            
            elif switch_view_request.wait_edge_events(timeout=timedelta(seconds=1)):
                events = switch_view_request.read_edge_events(max_events=1)
                if events:
                    event = events[0]
                    if (event.line_offset == SWITCH_VIEW_OFFSET and 
                        event.event_type == gpiod.EdgeEvent.Type.RISING_EDGE):
                        print("Switching display state...")
                        cycle_display_state()

        except Exception as e:
            print(f"Error in GPIO listener: {e}")
            time.sleep(0.5)

def cycle_display_state():
    current_state = shared_state.display_state
    next_state = DisplayState((current_state.value + 1) % len(DisplayState))
    shared_state.display_state = next_state
    print(f"Switched to display state: {shared_state.display_state.name}")

def keyboard_listener():
    while not stop_event.is_set():
        user_input = input("Press 'c' to cycle display state or 'q' to quit: ").strip().lower()
        if user_input == 'c':
            cycle_display_state()
        elif user_input == 'q':
            stop_event.set()
            print("Exiting...")

def main():
    # Waveshare 2inch, 240x320 display
    disp = LCD_2inch.LCD_2inch()
    disp.Init()
    disp.clear()
    disp.bl_DutyCycle(50)

    picam_ir = Picamera2(camera_num=0)
    picam_rgb = Picamera2(camera_num=1)

    picam_ir.set_controls({"AwbEnable":False, "ColourGains": (0.0,0.0)})

    config_ir = picam_ir.create_preview_configuration(
        main={"size": (2028, 1520), "format": "RGB888"},
        lores={"size": (320, 240), "format": "RGB888"},
        raw={"format": "SRGGB12_CSI2P", "size": (2028, 1520)},
        transform=Transform(vflip=True, hflip=True),
        colour_space=ColorSpace.Raw(),
        controls={'FrameRate': 15.0, 'SyncMode': controls.rpi.SyncModeEnum.Client}
    )
    config_rgb = picam_rgb.create_preview_configuration(
        main={"size": (2028, 1520), "format": "RGB888"},
        lores={"size": (320, 240), "format": "RGB888"},
        raw={"format": "SRGGB12_CSI2P", "size": (2028, 1520)},
        transform=Transform(vflip=True, hflip=True),
        colour_space=ColorSpace.Raw(),
        controls= {'FrameRate': 15.0, 'SyncMode': controls.rpi.SyncModeEnum.Server}
    )

    picam_ir.configure(config_ir)
    picam_rgb.configure(config_rgb)

    print(f"Cam 0 (IR) sensor format: {picam_ir.sensor_format}")
    print(f"Cam1 (RGB) sensor format: {picam_rgb.sensor_format}")
    print(f"RGB Camera Properties: {picam_rgb.camera_configuration()['raw']}")
    print(f"IR Camera Properties: {picam_ir.camera_configuration()['raw']}")

    picam_ir.start()
    picam_rgb.start()


    print("Press 'q' to quit the live feed loop...")

    # keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    # keyboard_thread.start()
    t0 = threading.Thread(target=gpio_listener, args=(picam_ir, picam_rgb,), daemon=True)

    try:
        t0.start()
        while True and not stop_event.is_set():
            # Capture low-resolution frames from each camera
            frame_ir = picam_ir.capture_array("lores")
            frame_rgb = picam_rgb.capture_array("lores")

            metadata_ir = picam_ir.capture_metadata()
            metadata_rgb = picam_rgb.capture_metadata()

            # Convert to BGR to RGB:
            frame_ir_corrected = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2RGB)
            frame_rgb_corrected = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)

            # Extract the red channel from the IR frame
            r_channel = frame_ir_corrected[:, :, 0]
            frame_ir_corrected = cv2.cvtColor(cv2.merge([r_channel, r_channel, r_channel]), cv2.COLOR_BGR2RGB)

            # Resize each frame individually before combining to maintain aspect ratio
            # For 240x320 display, each image should be 240x160 (half the height)
            frame_ir_pil = Image.fromarray(frame_ir_corrected, 'RGB')
            frame_rgb_pil = Image.fromarray(frame_rgb_corrected, 'RGB')

            # Calculate crop coordinates to get center of images
            # For 240x320 images, get the center 160 vertical pixels
            crop_sides = frame_ir_pil.width // 4
            crop_height = frame_ir_pil.height

            # Crop images to center 160 pixels vertically
            frame_ir_cropped = frame_ir_pil.crop((crop_sides, 0, crop_height, crop_height))
            frame_rgb_cropped = frame_rgb_pil.crop((crop_sides, 0, crop_height, crop_height))
            # Create a blank 240x320 canvas (white background)
            combined_img = Image.new('RGB', (disp.height, disp.width), color=(0, 0, 0))

            # Draw focus value
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
            except:
                font = ImageFont.load_default()
            ir_focus = f"{metadata_ir['FocusFoM']}"
            rgb_focus = f"{metadata_rgb['FocusFoM']}"
            draw = ImageDraw.Draw(combined_img)
            
            if shared_state.display_state == DisplayState.SPLIT:
                combined_img.paste(frame_rgb_cropped, (0, 0))
                combined_img.paste(frame_ir_cropped, (disp.height // 2, 0))
                draw.text((10, 10), f"RGB Focus: {rgb_focus}", fill=(57, 255, 20), font=font)
                draw.text((170, 10), f"IR Focus: {ir_focus}", fill=(57, 255, 20), font=font)
            elif shared_state.display_state == DisplayState.RGB_ONLY:
                combined_img.paste(frame_rgb_pil, (0, 0))
                draw.text((10, 10), f"RGB Focus: {rgb_focus}", fill=(57, 255, 20), font=font)
            elif shared_state.display_state == DisplayState.IR_ONLY:
                combined_img.paste(frame_ir_pil, (0, 0))
                draw.text((170, 10), f"IR Focus: {ir_focus}", fill=(57, 255, 20), font=font)

            current_time = time.time()
            if shared_state.capture_notification and (current_time - shared_state.notification_start_time) < shared_state.notification_duration:
                draw = ImageDraw.Draw(combined_img)

                border_width = 5
                draw.rectangle([(0, 0), (disp.height, disp.width)], outline=(255, 0, 0), width=border_width)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
                except:
                    font = ImageFont.load_default()

                text = "Capturing!"

                # Calculate text dimensions for centering
                text_bbox = draw.textbbox((0, 0), text,font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                text_x = (disp.height - text_width) // 2
                text_y = (disp.width - text_height) // 2

                # Draw the text
                draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

                # Draw a progress bar
                elapsed = current_time - shared_state.notification_start_time
                progress = elapsed / shared_state.notification_duration
                bar_width = int(240 * ( progress))
                draw.rectangle([(0, disp.width - border_width), (bar_width, disp.width)], fill=(0, 255, 0))
            elif shared_state.capture_notification:
                shared_state.capture_notification = False

            # Send to display
            combined_img = combined_img.rotate(90, expand=True)
            disp.ShowImage(combined_img)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        picam_ir.stop()
        picam_rgb.stop()
        cv2.destroyAllWindows()
        disp.module_exit()
        button_request.release()
        print("Cleanup complete. Exiting...")

if __name__ == "__main__":
    main()
