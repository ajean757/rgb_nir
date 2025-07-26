import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import rawpy

# -----------------------------------------------------------------------------
# DNG Loading Functions
# -----------------------------------------------------------------------------

debug = False

def load_rgb_from_dng(dng_path):
    """
    Load an RGB DNG file and process it to linear RGB (converted to BGR for OpenCV).
    
    Args:
        dng_path (str): Path to RGB DNG file
        
    Returns:
        np.ndarray: RGB image in [0,1] float32 format, shape (H, W, 3)
    """
    with rawpy.imread(dng_path) as raw:
        # Process with linear gamma and camera white balance
        rgb = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=True
        )
        # Convert to float32 and normalize to [0,1]
        rgb = rgb.astype(np.float32) / 65535.0
        # Convert from RGB to BGR for OpenCV
        rgb = rgb[..., ::-1]
        return rgb

def load_nir_from_dng(dng_path):
    """
    Load a NIR DNG file from monochrome sensor and process it.
    
    Args:
        dng_path (str): Path to NIR DNG file
        
    Returns:
        np.ndarray: NIR image in [0,1] float32 format, shape (H, W, 3) for compatibility
    """
    with rawpy.imread(dng_path) as raw:
        # Extract raw data
        nir_raw = raw.raw_image_visible.astype(np.float32)
        
        # Get black and white levels
        black_level = raw.black_level_per_channel[0] if hasattr(raw, 'black_level_per_channel') else 0
        white_level = raw.white_level if hasattr(raw, 'white_level') else 65535
        
        # Subtract black level and normalize
        nir_raw = nir_raw - black_level
        nir_raw = nir_raw / (white_level - black_level)
        
        # Clip to [0,1]
        nir_raw = np.clip(nir_raw, 0, 1)
        
        # Convert to 3-channel for compatibility with OpenCV operations
        nir_bgr = cv2.merge([nir_raw, nir_raw, nir_raw])
        
        return nir_bgr

def load_image_smart(image_path):
    """
    Load an image file, automatically detecting if it's JPEG or DNG.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        np.ndarray: Image in [0,1] float32 format, shape (H, W, 3)
    """
    file_ext = os.path.splitext(image_path)[1].lower()
    
    if file_ext == '.dng':
        # Determine if it's RGB or NIR based on filename
        if '_rgb' in os.path.basename(image_path).lower():
            return load_rgb_from_dng(image_path)
        elif '_ir' in os.path.basename(image_path).lower() or '_nir' in os.path.basename(image_path).lower():
            return load_nir_from_dng(image_path)
        else:
            # Default to RGB processing
            return load_rgb_from_dng(image_path)
    else:
        # Load JPEG/PNG with OpenCV and normalize
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = img.astype(np.float32) / 255.0
        return img

def save_image_as_png16(image, output_path):
    """
    Save a float32 [0,1] image as 16-bit PNG.
    
    Args:
        image (np.ndarray): Image in [0,1] float32 format
        output_path (str): Output path for PNG file
    """
    # Convert to 16-bit
    img_16bit = (image * 65535).astype(np.uint16)
    cv2.imwrite(output_path, img_16bit)

def save_image_as_npy(image, output_path):
    """
    Save image as numpy array for reproducibility.
    
    Args:
        image (np.ndarray): Image array
        output_path (str): Output path for NPY file
    """
    np.save(output_path, image)

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


def get_image_pairs(input_dir, file_extension=None):
    """
    Scan the given directory for image pairs.
    Assumes a naming convention that pairs images from two cameras (e.g., left_001.jpg, right_001.jpg).
    Where left is IR and right is RGB

    Args:
        input_dir (str): Directory containing stereo image pairs.
        file_extension (str): Optional file extension filter ('jpg', 'dng', etc.)
    
    Returns:
        list of tuples: Each tuple is (left_image_path, right_image_path).
    """
    if file_extension:
        ir_images = glob.glob(f"{input_dir}/*_ir.{file_extension}") 
        rgb_images = glob.glob(f"{input_dir}/*_rgb.{file_extension}") 
    else:
        # Support both jpg and dng
        ir_images = glob.glob(f"{input_dir}/*_ir.jpg") + glob.glob(f"{input_dir}/*_ir.dng")
        rgb_images = glob.glob(f"{input_dir}/*_rgb.jpg") + glob.glob(f"{input_dir}/*_rgb.dng")
    
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
    # Load images using smart loading (supports both JPEG and DNG)
    left_img = load_image_smart(left_path)
    right_img = load_image_smart(right_path)
    
    # Convert to uint8 for cv2.remap if needed
    if left_img.dtype == np.float32:
        left_img_uint8 = (left_img * 255).astype(np.uint8)
        right_img_uint8 = (right_img * 255).astype(np.uint8)
    else:
        left_img_uint8 = left_img
        right_img_uint8 = right_img

    rect_left = cv2.remap(left_img_uint8, camera_info['map1x'], camera_info['map1y'], cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img_uint8, camera_info['map2x'], camera_info['map2y'], cv2.INTER_LINEAR)
    
    # Convert back to float32 [0,1] format
    rect_left = rect_left.astype(np.float32) / 255.0
    rect_right = rect_right.astype(np.float32) / 255.0

    left_img_name = os.path.splitext(os.path.basename(left_path))[0]
    right_img_name = os.path.splitext(os.path.basename(right_path))[0]
    if save_intermediate and output_dir:
        # Save rectified images as 16-bit PNG
        save_image_as_png16(rect_left, os.path.join(output_dir, f"{left_img_name}_rect.png"))
        save_image_as_png16(rect_right, os.path.join(output_dir, f"{right_img_name}_rect.png"))
        # Also save as NPY for reproducibility
        save_image_as_npy(rect_left, os.path.join(output_dir, f"{left_img_name}_rect.npy"))
        save_image_as_npy(rect_right, os.path.join(output_dir, f"{right_img_name}_rect.npy"))
    
    return rect_left, rect_right

class GDISIFT:
    def __init__(self, **sift_kwargs):
        self.sift = cv2.SIFT_create(**sift_kwargs)

    def detect(self, img, mask=None):
        return self.sift.detect(img, mask)

    def compute(self, img, kps):
        _, desc128 = self.sift.compute(img, kps)
        desc64 = self._fold_bins(desc128)
        return kps, desc64

    def detect_and_compute(self, img, mask=None):
        kps = self.detect(img, mask)
        return self.compute(img, kps)

    @staticmethod
    def _fold_bins(desc128):
        d = desc128.reshape(-1, 16, 8)
        folded = d[:, :, :4] + d[:, :, 4:]
        folded = folded.reshape(-1, 64)
        # Renormalise
        norm = np.linalg.norm(folded, axis=1, keepdims=True) + 1e-7
        return folded / norm
    
def anms(h, interest_pts, n, c=0.9): 
    # define a dictionary where r[(x_i, y_i)] = r_i
    # for each interest point i
        # create an array containing radii for interest pt i
        # for each interest point j that's not i
            # if h(x_i) < c * h(x_j)
                # compute distance || x_i - x_j || and add to list
        # set r_i to be minimum radius 
    # sort dictionary by descending value r_i
    # choose first n points and return as array
    interest_r = {}
    new_x = []
    new_y = []
    for i in tqdm(range(interest_pts.shape[1])):
        radii = []
        for j in range(interest_pts.shape[1]):
            if i != j and h[interest_pts[0, i], interest_pts[1, i]] < h[interest_pts[0, j], interest_pts[1, j]]:
                x_i = np.array([interest_pts[0, i], interest_pts[1, i]])
                x_j = np.array([interest_pts[0, j], interest_pts[1, j]])
                radii.append(np.linalg.norm(x_i - x_j))
        x_i = (interest_pts[1, i], interest_pts[0, i])
        if len(radii) == 0:
            interest_r[x_i] = float("inf")
        else:
            interest_r[x_i] = min(radii)
    
    interest_sorted = dict(sorted(interest_r.items(), key=lambda x: x[1], reverse=True))
    for x, y in list(interest_sorted.keys())[:n]:
        new_x.append(x)
        new_y.append(y)
    
    
    new_pts = np.vstack([new_y, new_x])
    return new_pts


def anms_keypoints(keypoints, descriptors, num_to_keep=200, c=0.9):
    """
    Apply Adaptive Non-Maximal Suppression (ANMS) to OpenCV keypoints.
    Args:
        keypoints (list): List of cv2.KeyPoint objects.
        descriptors (np.ndarray): Corresponding descriptors.
        num_to_keep (int): Number of keypoints to keep.
        c (float): Suppression constant.
    Returns:
        selected_kps (list): Filtered keypoints.
        selected_desc (np.ndarray): Filtered descriptors.
    """
    if len(keypoints) <= num_to_keep:
        idxs = np.arange(len(keypoints))
        return keypoints, descriptors
    responses = np.array([kp.response for kp in keypoints])
    coords = np.array([kp.pt for kp in keypoints])
    radii = np.full(len(keypoints), np.inf)
    for i, (r_i, pt_i) in enumerate(zip(responses, coords)):
        mask = responses > c * r_i
        if np.any(mask):
            dists = np.linalg.norm(coords[mask] - pt_i, axis=1)
            radii[i] = dists.min()
    idxs = np.argsort(-radii)[:num_to_keep]
    selected_kps = [keypoints[i] for i in idxs]
    selected_desc = descriptors[idxs]
    return selected_kps, selected_desc

def align_images_with_sift(img_src, img_dst, min_match_count=10):
    # Store original data types for later restoration
    src_is_float = img_src.dtype == np.float32
    dst_is_float = img_dst.dtype == np.float32
    
    # Convert float32 [0,1] images to uint8 for SIFT if needed
    if img_src.dtype == np.float32:
        src_uint8 = (img_src * 255).astype(np.uint8)
    else:
        src_uint8 = img_src
        
    if img_dst.dtype == np.float32:
        dst_uint8 = (img_dst * 255).astype(np.uint8)
    else:
        dst_uint8 = img_dst
    
    gray_src = cv2.cvtColor(src_uint8, cv2.COLOR_BGR2GRAY) if src_uint8.ndim == 3 else src_uint8
    gray_dst = cv2.cvtColor(dst_uint8, cv2.COLOR_BGR2GRAY) if dst_uint8.ndim == 3 else dst_uint8

    sift = GDISIFT()
    kp1, des1 = sift.detect_and_compute(gray_src)
    kp2, des2 = sift.detect_and_compute(gray_dst)

    # sift = cv2.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(gray_src, mask=None)
    # kp2, des2 = sift.detectAndCompute(gray_dst, mask=None)


    # Visual debug: show keypoints before and after ANMS in a single 2x2 grid
    if debug:
        vis_src = cv2.drawKeypoints(src_uint8, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        vis_dst = cv2.drawKeypoints(dst_uint8, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Apply ANMS to get filtered keypoints for display only (don't overwrite real kp1/kp2 yet)
        kp1_anms, _ = anms_keypoints(kp1, des1, num_to_keep=500, c=0.8)
        kp2_anms, _ = anms_keypoints(kp2, des2, num_to_keep=500, c=0.8)
        vis_src_anms = cv2.drawKeypoints(src_uint8, kp1_anms, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        vis_dst_anms = cv2.drawKeypoints(dst_uint8, kp2_anms, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Resize all to same size for tiling
        h, w = vis_src.shape[:2]
        vis_src_anms = cv2.resize(vis_src_anms, (w, h))
        vis_dst = cv2.resize(vis_dst, (w, h))
        vis_dst_anms = cv2.resize(vis_dst_anms, (w, h))
        # Stack in 2x2 grid
        top = np.hstack([vis_src, vis_dst])
        bottom = np.hstack([vis_src_anms, vis_dst_anms])
        grid = np.vstack([top, bottom])
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(grid, 'Src before ANMS', (10, 30), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(grid, 'Dst before ANMS', (w+10, 30), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(grid, 'Src after ANMS', (10, h+30), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(grid, 'Dst after ANMS', (w+10, h+30), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('SIFT Keypoints (Before/After ANMS)', grid)
        print("Debug: Press any key to continue.")
        cv2.waitKey(0)
        cv2.destroyWindow('SIFT Keypoints (Before/After ANMS)')

    # Now actually apply ANMS to keypoints and descriptors for matching

    kp1, des1 = anms_keypoints(kp1, des1, num_to_keep=500, c=0.9)
    kp2, des2 = anms_keypoints(kp2, des2, num_to_keep=500, c=0.9)
    print(f"Debug - after ANMS keypoints: src={len(kp1)}, dst={len(kp2)}")

    # Ensure descriptors are float32 for FLANN
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Debug: descriptor shapes and number of raw matches
    print(f"Debug - descriptor shapes: des1={des1.shape}, des2={des2.shape}")
    matches = flann.knnMatch(des1, des2, k=2)
    print(f"Debug - raw FLANN matches: {len(matches)}")
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    print(f"Debug - good matches after ratio test: {len(good_matches)}")

    if len(good_matches) < min_match_count:
        print("Debug - ANMS matching insufficient, retrying without ANMS filtering...")
        # Retry using all keypoints/descriptors
        kp1, des1 = sift.detect_and_compute(gray_src)  # re-run detection
        kp2, des2 = sift.detect_and_compute(gray_dst)
        # Convert descriptors to float32
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)
        # Rematch
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        print(f"Debug - retry good matches after ratio test: {len(good_matches)}")
        if len(good_matches) < min_match_count:
            print(f"Debug - still insufficient matches after retry: {len(good_matches)} < {min_match_count}")
            return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Apply transformation using the original image (preserve data type)
    aligned = cv2.warpPerspective(img_src, H, (img_dst.shape[1], img_dst.shape[0]))
    
    # Debug: Print transformation info
    print(f"Debug - SIFT matches: {len(good_matches)}, inliers: {int(mask.sum()) if mask is not None else 0}")
    print(f"Debug - Source image dtype: {img_src.dtype}, range: [{img_src.min():.3f}, {img_src.max():.3f}]")
    print(f"Debug - Aligned image dtype: {aligned.dtype}, range: [{aligned.min():.3f}, {aligned.max():.3f}]")
    
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
    # Ensure both images are in the same format (float32 [0,1])
    if rect_left.dtype != np.float32:
        rect_left = rect_left.astype(np.float32) / 255.0
    if warped_right.dtype != np.float32:
        warped_right = warped_right.astype(np.float32) / 255.0
    
    # Debug: Print average values before overlay
    print(f"Debug - rect_left avg per channel: {np.mean(rect_left, axis=(0,1))}")
    print(f"Debug - warped_right avg per channel: {np.mean(warped_right, axis=(0,1))}")
    
    # Create overlay using float32 arithmetic
    overlay = 0.5 * rect_left + 0.5 * warped_right
    
    # Debug: Print overlay average
    print(f"Debug - overlay avg per channel: {np.mean(overlay, axis=(0,1))}")
    
    if save_intermediate and output_dir:
        # Convert to uint8 for saving
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{img_base_name}_overlay.jpg"), overlay_uint8)
        # Also save as 16-bit PNG for better quality
        save_image_as_png16(overlay, os.path.join(output_dir, f"{img_base_name}_overlay.png"))
    
    return overlay

def aerochrome_filter(img_ir, img_rgb, img_base_name, save_intermediate=False, output_dir=None):
    """
    Create an aerochrome (false color infrared) image.
    
    Args:
        img_ir (np.ndarray): IR image
        img_rgb (np.ndarray): RGB image
        img_base_name (str): Base name for output files
        save_intermediate (bool): Whether to save the result
        output_dir (str): Directory to save the result
    
    Returns:
        np.ndarray: Aerochrome image
    """
    # Ensure both images are in float32 [0,1] format
    if img_ir.dtype != np.float32:
        img_ir = img_ir.astype(np.float32) / 255.0
    if img_rgb.dtype != np.float32:
        img_rgb = img_rgb.astype(np.float32) / 255.0

    # Extract IR channel (convert to grayscale if needed)
    if img_ir.ndim == 3:
        ir_channel = cv2.cvtColor((img_ir * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        ir_channel = img_ir
    
    # Extract RGB channels
    green_channel = img_rgb[:,:,1]
    red_channel = img_rgb[:,:,2]

    # Create aerochrome: Blue <- Green, Green <- Red, Red <- IR
    result = np.stack([
        green_channel,   # Blue channel <- Green
        red_channel,     # Green channel <- Red
        ir_channel       # Red channel <- IR
    ], axis=2)
    
    # Debug: Print channel averages
    print(f"Debug - Aerochrome channels avg: {np.mean(result, axis=(0,1))}")

    if save_intermediate and output_dir:
        # Save as both 16-bit PNG and 8-bit JPG
        save_image_as_png16(result, os.path.join(output_dir, f"{img_base_name}_aerochrome.png"))
        result_uint8 = (result * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{img_base_name}_aerochrome.jpg"), result_uint8)
    
    return result

    


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
    print("Debug: Press ESC or 'q' to close.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 27 or key == ord('q'):
            break
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# Main Pipeline Function
# -----------------------------------------------------------------------------

def process_stereo_dataset(input_dir, npz_path, output_dirs, save_options, file_ext=None, view_results=False):
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
    pairs = get_image_pairs(input_dir, file_extension=file_ext)

    aligned = 0
    num_imgs = 0

    for left_path, right_path in pairs:
        num_imgs += 1
        # Load original images
        # original_left = cv2.imread(left_path)
        # original_right = cv2.imread(right_path)
        
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
            # Save warped image properly based on data type
            if warped_left.dtype == np.float32:
                # Save as both 16-bit PNG and 8-bit JPG
                save_image_as_png16(warped_left, os.path.join(output_dirs.get('warped', None), f'{left_img_name}_warped.png'))
                warped_uint8 = (warped_left * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dirs.get('warped', None), f'{left_img_name}_warped.jpg'), warped_uint8)
            else:
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
# Benchmark: GDISIFT vs SIFT
# -----------------------------------------------------------------------------

def benchmark_gdisift_vs_sift(rgb_dir, nir_dir, subset=None, npz_path=None):
    import cv2, glob, os, numpy as np
    from skimage.metrics import structural_similarity as ssim
    from tqdm import tqdm

    sift  = cv2.SIFT_create()
    gdis  = GDISIFT()

    def eval_pair(imgA, imgB, detA, detB):
        kpA, desA = detA.detectAndCompute(imgA, None) if isinstance(detA, cv2.SIFT) \
                    else detA.detect_and_compute(imgA)
        kpB, desB = detB.detectAndCompute(imgB, None) if isinstance(detB, cv2.SIFT) \
                    else detB.detect_and_compute(imgB)
        if desA is None or desB is None:
            return None
        matcher  = cv2.BFMatcher(cv2.NORM_L2)
        knn      = matcher.knnMatch(desA, desB, k=2)
        good     = [m for m,n in knn if m.distance < 0.75*n.distance]
        if len(good) < 4: return None
        ptsA = np.float32([kpA[m.queryIdx].pt for m in good])
        ptsB = np.float32([kpB[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3.0)
        if H is None: return None
        inliers = int(mask.sum())
        reproj  = cv2.perspectiveTransform(ptsA.reshape(-1,1,2), H) - ptsB.reshape(-1,1,2)
        med_rep = np.median(np.linalg.norm(reproj, axis=2)[mask.ravel()==1])
        h,w = imgB.shape[:2]
        warped = cv2.warpPerspective(imgA, H, (w,h))
        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray_B = cv2.cvtColor(imgB , cv2.COLOR_BGR2GRAY)
        ssim_val = ssim(gray_w, gray_B)
        return dict(
            good=len(good),
            inliers=inliers,
            inlier_ratio=inliers/len(good),
            med_reproj=med_rep,
            ssim=ssim_val
        )

    # Load camera info for rectification
    camera_info = load_camera_info(npz_path) if npz_path else None
    all_pairs = sorted(glob.glob(os.path.join(rgb_dir, '*_rgb.jpg'))) + sorted(glob.glob(os.path.join(rgb_dir, '*_rgb.dng')))
    if subset is not None:
        all_pairs = all_pairs[:subset]
    metrics = {'sift': [], 'gdi': []}
    for rgb_path in tqdm(all_pairs, desc="Benchmarking pairs"):
        basename = os.path.basename(rgb_path)
        # Handle both jpg and dng files
        if basename.endswith('_rgb.jpg'):
            nir_filename = basename.replace('_rgb.jpg', '_ir.jpg')
        elif basename.endswith('_rgb.dng'):
            nir_filename = basename.replace('_rgb.dng', '_ir.dng')
        else:
            continue
            
        nir_path = os.path.join(nir_dir, nir_filename)
        if not os.path.exists(nir_path): continue
        
        # Rectify images before benchmarking
        if camera_info:
            rect_ir, rect_rgb = rectify_image_pair(nir_path, rgb_path, camera_info)
            # Convert to uint8 for SIFT processing
            rect_ir = (rect_ir * 255).astype(np.uint8)
            rect_rgb = (rect_rgb * 255).astype(np.uint8)
        else:
            # Load and convert to uint8
            rect_ir_float = load_image_smart(nir_path)
            rect_rgb_float = load_image_smart(rgb_path)
            rect_ir = (rect_ir_float * 255).astype(np.uint8)
            rect_rgb = (rect_rgb_float * 255).astype(np.uint8)
            
        m_sift = eval_pair(rect_rgb, rect_ir, sift,  sift)
        m_gdi  = eval_pair(rect_rgb, rect_ir, gdis, gdis)
        if m_sift:
            metrics['sift'].append(m_sift)
        if m_gdi:
            metrics['gdi'].append(m_gdi)
    def agg(arr, key): v=[x[key] for x in arr]; return np.mean(v), np.std(v)
    for name, arr in metrics.items():
        if not arr: continue
        print(f'\n{name.upper()} on {len(arr)} pairs')
        for k in arr[0]:
            mu,sd = agg(arr,k)
            print(f'  {k:<12}: {mu:8.3f}  ±{sd:.3f}')
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
    
    import argparse
    parser = argparse.ArgumentParser(description="Stereo pipeline and benchmark")
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark instead of pipeline')
    parser.add_argument('--subset', type=int, default=None, help='Number of image pairs to use for benchmark')
    parser.add_argument('--npz', type=str, default="./calibration_files/calib_data7.npz", help='Path to camera calibration npz file')
    parser.add_argument('--ext', type=str, default=None, help='File extension to process (jpg, dng, etc.)')
    parser.add_argument('--anms', action='store_true', help='Use Adaptive Non-Maximal Suppression for keypoint filtering')
    args = parser.parse_args()
    
    # Determine file extension based on arguments
    if args.ext:
        file_ext = args.ext
    else:
        file_ext = None  # Process both jpg and dng
        
    if args.benchmark:
        rgb_dir = "./DATA/data_04_06_2025"
        nir_dir = "./DATA/data_04_06_2025"
        # Update benchmark to also support DNG files
        if file_ext == 'dng':
            all_pairs = sorted(glob.glob(os.path.join(rgb_dir, '*_rgb.dng')))
        else:
            all_pairs = sorted(glob.glob(os.path.join(rgb_dir, '*_rgb.jpg')))
        benchmark_gdisift_vs_sift(rgb_dir, nir_dir, subset=args.subset, npz_path=args.npz)
    else:
        process_stereo_dataset(input_directory, camera_npz_path, output_directories, save_opts, file_ext=file_ext, view_results=True)


### Usage
# # Process JPEG files (original behavior)
# python rectify.py

# # Process DNG files only
# python rectify.py --dng

# # Run benchmark on DNG files
# python rectify.py --benchmark --dng --subset 5

# # Process specific file extension
# python rectify.py --ext png