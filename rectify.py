import os
import cv2
import glob
import numpy as np

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
    camera_info = np.load(npz_path)
    return camera_info


def get_image_pairs(input_dir):
    """
    Scan the given directory for image pairs.
    Assumes a naming convention that pairs images from two cameras (e.g., left_001.jpg, right_001.jpg).
    Where left is IR and right is RGB

    Args:
        input_dir (str): Directory containing stereo image pairs.
    
    Returns:
        list of tuples: Each tuple is (left_image_path, right_image_path).
    """
    ir_images = glob.glob(f"{input_dir}/*_ir.jpg") 
    rgb_images = glob.glob(f"{input_dir}/*_rgb.jpg") 
    ir_images.sort()
    rgb_images.sort()
    image_pairs = []
    for ir, rgb in zip(ir_images, rgb_images):
        image_pairs.append((ir, rgb))
    return image_pairs


# -----------------------------------------------------------------------------
# Processing Steps
# -----------------------------------------------------------------------------

def rectify_image_pair(left_path : str, right_path : str, camera_info, save_intermediate=False, output_dir=None):
    """
    Rectify a stereo pair using the provided camera calibration info.
    
    Args:
        left_path (str): Original left image path.
        right_path (str): Original right image path.
        camera_info (dict): Dictionary containing camera matrices, distortion coeffs, R1, R2, P1, P2, etc.
        save_intermediate (bool): Whether to save the rectified images.
        output_dir (str): Directory to save rectified images (if saving is enabled).
    
    Returns:
        tuple: (rectified_left, rectified_right)
    """
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    rect_left = cv2.remap(left_img, camera_info['map1x'], camera_info['map1y'], cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, camera_info['map2x'], camera_info['map2y'], cv2.INTER_LINEAR)

    left_img_name = os.path.splitext(os.path.basename(left_path))[0]
    right_img_name = os.path.splitext(os.path.basename(right_path))[0]
    if save_intermediate and output_dir:
        # Save rectified images to output_dir
        cv2.imwrite(os.path.join(output_dir, f"{left_img_name}_rect.jpg"), rect_left)
        cv2.imwrite(os.path.join(output_dir, f"{right_img_name}_rect.jpg"), rect_right)
    
    return rect_left, rect_right

def align_images_with_sift(img_src, img_dst, min_match_count=10):
    gray_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY) if img_src.ndim == 3 else img_src
    gray_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY) if img_dst.ndim == 3 else img_dst

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_src, None)
    kp2, des2 = sift.detectAndCompute(gray_dst, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < min_match_count:
        return None, None
        # raise ValueError(f"Not enough matches found - {len(good_matches)} < {min_match_count}")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned = cv2.warpPerspective(img_src, H, (img_dst.shape[1], img_dst.shape[0]))
    return aligned, H




def overlay_images(rect_left, warped_right, img_base_name, save_intermediate=False, output_dir=None):
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
        cv2.imwrite(os.path.join(output_dir, f"{img_base_name}_overlay.jpg"), overlay)
    
    return overlay

def aerochrome_filter(img_ir, img_rgb, img_base_name, save_intermediate=False, output_dir=None):


    ir_channel = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
    green_channel = img_rgb[:,:,1]
    red_channel = img_rgb[:,:,2]


    result = cv2.merge([
        green_channel,   # Blue channel <- Green
        red_channel,     # Green channel <- Red
        ir_channel       # Red channel <- IR
    ])

    if save_intermediate and output_dir:
        cv2.imwrite(os.path.join(output_dir, f"{img_base_name}_aerochrome.jpg"), result)

    


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

    aligned = 0
    num_imgs = 0

    for left_path, right_path in pairs:
        num_imgs += 1
        # Load original images
        original_left = cv2.imread(left_path)
        original_right = cv2.imread(right_path)
        
        # Step 1: Rectify the stereo pair
        rect_left, rect_right = rectify_image_pair(left_path, right_path,
                                                   camera_info,
                                                   save_intermediate=save_options.get('rectified', False),
                                                   output_dir=output_dirs.get('rectified', None))
        
        # Step 2: Compute Homography and align left image to right
        warped_left, H = align_images_with_sift(rect_left, rect_right)
        if warped_left is None:
            print(f"Failed to align {right_path} to {left_path}")
            continue
        
        left_img_name = os.path.splitext(os.path.basename(left_path))[0]

        if save_options.get('warped', False):
            cv2.imwrite(os.path.join(output_dirs.get('warped', None), f'{left_img_name}_warped.jpg'), warped_left)

    
        # Step 3: Create overlay of the warped right image and the left rectified imag
        base_img_name = left_img_name[0:-3]
        overlay = overlay_images(rect_right, warped_left,
                                 base_img_name,
                                 save_intermediate=save_options.get('overlay', True),
                                 output_dir=output_dirs.get('overlay', None))
        
        # Step 4: Create IR false color
        aerochrome = aerochrome_filter(warped_left, rect_right, 
                                       base_img_name,
                                       save_intermediate=save_options.get('aerochrome', True),
                                       output_dir=output_dirs.get('aerochrome',None))
        
        aligned += 1
        print("Aligned images: ", left_path, right_path)
        # if aligned == 5:
        #     break

        # TODO: Optionally log progress, handle errors, etc.
        
    print(f"aligned {aligned} images out of {num_imgs}")
# -----------------------------------------------------------------------------
# Example Usage (Main Function)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Define paths and options (adjust as needed)
    input_directory = "./DATA/data_04_06_2025"
    camera_npz_path = "./calibration_files/calib_data7.npz"
    output_directories = {
        'rectified': "DATA/data_04_06_2025/rectified",
        'disparity': "DATA/data_04_06_2025/disparity",
        'warped': "DATA/data_04_06_2025/warped",
        'overlay': "DATA/data_04_06_2025/overlay",
        'aerochrome': "DATA/data_04_06_2025/aerochrome"
    }
    save_opts = {
        'rectified': True,
        'disparity': True,
        'warped': True,
        'overlay': True,
        'aerochrome':True
    }
    
    # Create output directories if they don't exist
    for key, out_dir in output_directories.items():
        os.makedirs(out_dir, exist_ok=True)
    
    process_stereo_dataset(input_directory, camera_npz_path, output_directories, save_opts, view_results=True)
