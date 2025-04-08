import os
import cv2
import numpy as np
# Import any GUI framework you prefer (e.g., PySimpleGUI, Tkinter) for viewing/debugging

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def load_camera_info(npz_path):
    """
    Load camera calibration and rectification parameters from an NPZ file.
    
    Args:
        npz_path (str): Path to the NPZ file containing camera information.
        
    Returns:
        dict: Dictionary with keys like:
            - 'cm0': Left camera matrix
            - 'dist0': Left distortion coefficients
            - 'cm1': Right camera matrix
            - 'dist1': Right distortion coefficients
            - 'R1', 'R2', 'P1', 'P2', 'Q': Rectification and projection matrices.
    """
    # TODO: load the NPZ file and extract parameters
    camera_info = np.load(npz_path)
    return camera_info


def get_image_pairs(input_dir):
    """
    Scan the given directory for image pairs.
    Assumes a naming convention that pairs images from two cameras (e.g., left_001.jpg, right_001.jpg).
    
    Args:
        input_dir (str): Directory containing stereo image pairs.
    
    Returns:
        list of tuples: Each tuple is (left_image_path, right_image_path).
    """
    # TODO: Implement a function to parse file names and pair them
    image_pairs = []  # e.g., [('path/to/left_001.jpg', 'path/to/right_001.jpg'), ...]
    return image_pairs


# -----------------------------------------------------------------------------
# Processing Steps
# -----------------------------------------------------------------------------

def rectify_image_pair(left_img, right_img, camera_info, save_intermediate=False, output_dir=None):
    """
    Rectify a stereo pair using the provided camera calibration info.
    
    Args:
        left_img (np.ndarray): Original left image.
        right_img (np.ndarray): Original right image.
        camera_info (dict): Dictionary containing camera matrices, distortion coeffs, R1, R2, P1, P2, etc.
        save_intermediate (bool): Whether to save the rectified images.
        output_dir (str): Directory to save rectified images (if saving is enabled).
    
    Returns:
        tuple: (rectified_left, rectified_right)
    """
    # Use cv2.initUndistortRectifyMap() and cv2.remap() for each image.
    # For example:
    # map1x, map1y = cv2.initUndistortRectifyMap(cm0, dist0, R1, P1, image_size, cv2.CV_32FC1)
    # rect_left = cv2.remap(left_img, map1x, map1y, cv2.INTER_LINEAR)
    # Similarly for right image.
    
    # TODO: Implement rectification
    rect_left = left_img  # Placeholder
    rect_right = right_img  # Placeholder

    if save_intermediate and output_dir:
        # Save rectified images to output_dir
        cv2.imwrite(os.path.join(output_dir, "rectified_left.jpg"), rect_left)
        cv2.imwrite(os.path.join(output_dir, "rectified_right.jpg"), rect_right)
    
    return rect_left, rect_right


def compute_disparity(rect_left, rect_right, save_intermediate=False, output_dir=None):
    """
    Compute the disparity map for a rectified stereo pair using StereoSGBM + WLS filtering.
    
    Args:
        rect_left (np.ndarray): Rectified left image.
        rect_right (np.ndarray): Rectified right image.
        save_intermediate (bool): Whether to save the disparity map.
        output_dir (str): Directory to save the disparity map.
    
    Returns:
        np.ndarray: The filtered disparity map.
    """
    # TODO: Set up StereoSGBM parameters and compute disparity, then filter it.
    # Use your current disparity implementation with WLS filtering.
    disparity = np.zeros(rect_left.shape[:2], dtype=np.float32)  # Placeholder
    
    if save_intermediate and output_dir:
        # Save disparity image (after normalization if needed)
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(output_dir, "disparity.jpg"), disp_vis)
    
    return disparity


def warp_image_pair(rect_right, disparity, save_intermediate=False, output_dir=None):
    """
    Warp the rectified right image into the left image's coordinate system using the disparity map.
    
    Args:
        rect_right (np.ndarray): Rectified right image.
        disparity (np.ndarray): Filtered disparity map.
        save_intermediate (bool): Whether to save the warped image.
        output_dir (str): Directory to save the warped image.
    
    Returns:
        np.ndarray: The warped right image.
    """
    # TODO: Create a coordinate grid, adjust x-coordinates by disparity, and use cv2.remap().
    warped_right = rect_right  # Placeholder
    
    if save_intermediate and output_dir:
        cv2.imwrite(os.path.join(output_dir, "warped_right.jpg"), warped_right)
    
    return warped_right


def overlay_images(rect_left, warped_right, save_intermediate=False, output_dir=None):
    """
    Create an overlay of the rectified left image and the warped right image.
    
    Args:
        rect_left (np.ndarray): Rectified left image.
        warped_right (np.ndarray): Warped right image (aligned to left).
        save_intermediate (bool): Whether to save the overlay image.
        output_dir (str): Directory to save the overlay image.
    
    Returns:
        np.ndarray: The overlay image.
    """
    # TODO: Combine images with cv2.addWeighted (or another method) for visualization.
    overlay = cv2.addWeighted(rect_left, 0.5, warped_right, 0.5, 0)
    
    if save_intermediate and output_dir:
        cv2.imwrite(os.path.join(output_dir, "overlay.jpg"), overlay)
    
    return overlay


# -----------------------------------------------------------------------------
# GUI/Debug Viewer
# -----------------------------------------------------------------------------

def view_pipeline_results(original_left, original_right, rect_left, rect_right, disparity, warped_right, overlay):
    """
    Display a GUI window that shows all intermediate results for debugging.
    
    Args:
        original_left (np.ndarray): Original left image.
        original_right (np.ndarray): Original right image.
        rect_left (np.ndarray): Rectified left image.
        rect_right (np.ndarray): Rectified right image.
        disparity (np.ndarray): Disparity map.
        warped_right (np.ndarray): Warped right image.
        overlay (np.ndarray): Overlay image.
    
    Returns:
        None
    """
    # TODO: Implement a simple GUI window using your preferred framework.
    # This can cycle through the images, or display them side by side.
    # For example, you might use OpenCV's imshow in a loop or a dedicated GUI library.
    cv2.imshow("Original Left", original_left)
    cv2.imshow("Original Right", original_right)
    cv2.imshow("Rectified Left", rect_left)
    cv2.imshow("Rectified Right", rect_right)
    cv2.imshow("Disparity", cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imshow("Warped Right", warped_right)
    cv2.imshow("Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# Main Pipeline Function
# -----------------------------------------------------------------------------

def process_stereo_dataset(input_dir, npz_path, output_dirs, save_options, view_results=False):
    """
    Process a directory of stereo image pairs:
      1. Rectify each pair.
      2. Save rectified images (optionally).
      3. Compute disparity for each pair.
      4. Save disparity images (optionally).
      5. Warp the right image to align with the left.
      6. Save the warped images (optionally).
      7. Create and save overlays (optionally).
      8. Optionally display each processing step in a GUI for debugging.
    
    Args:
        input_dir (str): Directory containing the original stereo images.
        npz_path (str): Path to NPZ file with camera calibration info.
        output_dirs (dict): Dictionary with keys like 'rectified', 'disparity', 'warped', 'overlay'.
        save_options (dict): Options to turn saving of each intermediate result on/off.
            e.g., {'rectified': True, 'disparity': False, 'warped': True, 'overlay': True}
        view_results (bool): Whether to launch a GUI viewer to inspect each image pair.
    
    Returns:
        None
    """
    # Load camera info from NPZ file
    camera_info = load_camera_info(npz_path)
    
    # Get list of image pairs
    pairs = get_image_pairs(input_dir)
    
    for left_path, right_path in pairs:
        # Load original images
        original_left = cv2.imread(left_path)
        original_right = cv2.imread(right_path)
        
        # Step 1: Rectify the stereo pair
        rect_left, rect_right = rectify_image_pair(original_left, original_right,
                                                   camera_info,
                                                   save_intermediate=save_options.get('rectified', False),
                                                   output_dir=output_dirs.get('rectified', None))
        
        # Step 2: Compute disparity map for the rectified pair
        disparity = compute_disparity(rect_left, rect_right,
                                      save_intermediate=save_options.get('disparity', False),
                                      output_dir=output_dirs.get('disparity', None))
        
        # Step 3: Warp the right rectified image to align with the left
        warped_right = warp_image_pair(rect_right, disparity,
                                       save_intermediate=save_options.get('warped', False),
                                       output_dir=output_dirs.get('warped', None))
        
        # Step 4: Create overlay of the warped right image and the left rectified image
        overlay = overlay_images(rect_left, warped_right,
                                 save_intermediate=save_options.get('overlay', False),
                                 output_dir=output_dirs.get('overlay', None))
        
        # Optionally display results in a GUI for debugging/inspection
        if view_results:
            view_pipeline_results(original_left, original_right, rect_left, rect_right, disparity, warped_right, overlay)
        
        # TODO: Optionally log progress, handle errors, etc.
        

# -----------------------------------------------------------------------------
# Example Usage (Main Function)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Define paths and options (adjust as needed)
    input_directory = "path/to/input_images"
    camera_npz_path = "path/to/camera_info.npz"
    output_directories = {
        'rectified': "path/to/save/rectified",
        'disparity': "path/to/save/disparity",
        'warped': "path/to/save/warped",
        'overlay': "path/to/save/overlay"
    }
    save_opts = {
        'rectified': True,
        'disparity': True,
        'warped': True,
        'overlay': True
    }
    
    # Create output directories if they don't exist
    for key, out_dir in output_directories.items():
        os.makedirs(out_dir, exist_ok=True)
    
    process_stereo_dataset(input_directory, camera_npz_path, output_directories, save_opts, view_results=True)
