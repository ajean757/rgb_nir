import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# --- Step 1: Stereo Camera Calibration ---

# Checkerboard parameters
CHECKERBOARD = (10,7)  # adjust if using a different checkerboard
square_size = 0.025  # meter (e.g., 25mm squares)

# Termination criteria for corner subpixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Load calibration images for each camera
ir_images = [] # Right camera
rgb_images = [] # Left camera
directory = "./new_calib_data"
for filename in os.listdir(directory):
    if filename.endswith("_ir.jpg"):
        ir_images.append(os.path.join(directory, filename))
    elif filename.endswith("_rgb.jpg"):
        rgb_images.append(os.path.join(directory, filename))
ir_images.sort()
rgb_images.sort()
print("IR images: ", ir_images)
print("RGB images: ", rgb_images)
images_left = rgb_images
images_right = ir_images


def calibrate_camera(image_paths, pattern_size, square_size):
    """
    Calibrate a camera using chessboard patterns.
    
    Args:
        image_paths: List of image file paths
        pattern_size: Tuple (corners_width, corners_height) for the chessboard pattern
        square_size: Physical size of a square in meters
        
    Returns:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs
    """
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
    
    valid_images = 0
    
    for image_path in image_paths:
        print(f"Processing {image_path}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Warning: File {image_path} does not exist")
            continue
            
        # Read the image
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Failed to load image {image_path}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            valid_images += 1
            print(f"  Found chessboard corners in {image_path}")
            
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_sub)
            
            # Optional: Draw and display the corners
            # cv2.drawChessboardCorners(img, pattern_size, corners_sub, ret)
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # plt.show()
    
    print(f"Found patterns in {valid_images} images out of {len(image_paths)}")
    
    if valid_images == 0:
        raise ValueError("No chessboard patterns were detected in any of the images. Cannot calibrate.")
    
    # Now calibrate
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print('rmse:', ret)
    print('camera matrix:\n', camera_matrix)
    print('distortion coeffs:', dist_coeffs)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def stereo_rectify(cm0, dist0, cm1, dist1, R, T, image_size):
    """
    Stereo rectification. Produces transformation matrices for undistortion and
    rectification for both cameras.
    """
    # alpha=0 -> crop the image to only valid region
    # alpha=1 -> no cropping (may have black borders)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cm0, dist0,
        cm1, dist1,
        image_size,
        R, T,
        alpha=1
    )

    # Create undistortion/rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(cm0, dist0, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cm1, dist1, R2, P2, image_size, cv2.CV_32FC1)

    return (map1x, map1y), (map2x, map2y), Q


def stereo_calibrate(cam0_pairs, cam1_pairs,
                     cam0_matrix, cam0_dist,
                     cam1_matrix, cam1_dist,
                     pattern_size, square_size):
    """
    Stereo calibrate two cameras given matching pairs of calibration images
    (cam0_pairs[i] matches cam1_pairs[i]).
    """

    # Checkerboard 3D point template
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp = objp * square_size

    objpoints = []  # 3D scene points
    imgpoints_left = []   # 2D image points from camera 0
    imgpoints_right = []  # 2D image points from camera 1

    for f0, f1 in zip(cam0_pairs, cam1_pairs):
        # Load the images
        if isinstance(f0, str):
            img0 = cv2.imread(f0, cv2.IMREAD_GRAYSCALE)
        else:
            img0 = f0
            if len(img0.shape) == 3:
                img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        if isinstance(f1, str):
            img1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
        else:
            img1 = f1
            if len(img1.shape) == 3:
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        ret0, corners0 = cv2.findChessboardCorners(img0, pattern_size, None)
        ret1, corners1 = cv2.findChessboardCorners(img1, pattern_size, None)

        if ret0 and ret1:
            # Refine corners
            corners0 = cv2.cornerSubPix(img0, corners0, (5, 5), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            corners1 = cv2.cornerSubPix(img1, corners1, (5, 5), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints_left.append(corners0)
            imgpoints_right.append(corners1)

    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    # You might set flags = cv2.CALIB_FIX_INTRINSIC if you want to fix intrinsics from prior calibrations
    # Example: flags |= cv2.CALIB_FIX_INTRINSIC

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    ret, cm0, dist0, cm1, dist1, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        cam0_matrix, cam0_dist,
        cam1_matrix, cam1_dist,
        img0.shape[::-1],  # image size
        criteria=criteria,
        flags=flags
    )

    print("Stereo calibration RMS error:", ret)
    return cm0, dist0, cm1, dist1, R, T, E, F


def stereo_rectify(cm0, dist0, cm1, dist1, R, T, image_size):
    """
    Stereo rectification. Produces transformation matrices for undistortion and
    rectification for both cameras.
    """
    # alpha=0 -> crop the image to only valid region
    # alpha=1 -> no cropping (may have black borders)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cm0, dist0,
        cm1, dist1,
        image_size,
        R, T,
        alpha=1
    )

    # Create undistortion/rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(cm0, dist0, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cm1, dist1, R2, P2, image_size, cv2.CV_32FC1)

    return (map1x, map1y), (map2x, map2y), Q

def rectify_stereo_pair(img0, img1, map1x, map1y, map2x, map2y):
    """
    Takes a new stereo pair (img0, img1), rectifies them using the provided maps,
    and returns the undistorted, rectified pair.
    """
    rect0 = cv2.remap(img0, map1x, map1y, cv2.INTER_LINEAR)
    rect1 = cv2.remap(img1, map2x, map2y, cv2.INTER_LINEAR)
    return rect0, rect1

    
rgb_err, rgb_cam_matrix, rgb_cam_dist, rvecs_rgb, tvecs_rgb = calibrate_camera(rgb_images, CHECKERBOARD, square_size)
print(f"RGB camera calibration successful, error: {rgb_err}")

ir_err, ir_cam_matrix, ir_cam_dist, rvecs_ir, tvecs_ir = calibrate_camera(ir_images, CHECKERBOARD, square_size)
print(f"IR camera calibration successful, error: {ir_err}")

cm0, dist0, cm1, dist1, R, T, E, F = stereo_calibrate(
    ir_images,  # pairs from camera0
    rgb_images,  # matching pairs from camera1
    ir_cam_matrix, ir_cam_dist,
    rgb_cam_matrix, rgb_cam_dist,
    CHECKERBOARD, square_size
)


# After calibrating:
# Usually the 'image_size' should match the resolution at which you did the calibration.
image_size = (2028,1520)  # e.g. (1280, 720)
(map1x, map1y), (map2x, map2y), Q = stereo_rectify(cm0, dist0, cm1, dist1, R, T, image_size)

np.savez('calib.npz', map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y, Q=Q)

