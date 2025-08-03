#!/usr/bin/env python3
"""
RGB-NIR Image Registration Pipeline

A robust pipeline for registering RGB↔NIR image pairs from JPEG and DNG files,
with support for camera rectification, SIFT-based alignment with ANMS filtering,
automatic cropping to eliminate black borders, and various output formats.

Features:
- DNG and JPEG support with automatic detection
- Camera rectification using calibration parameters
- SIFT feature detection with optional ANMS filtering
- Automatic cropping to valid region (eliminates black borders)
- Visual debugging capabilities
- Multiple output formats (NPY, 16-bit PNG, 8-bit JPEG)
- Average pixel value computation per channel
- Aerochrome false-color generation


# Basic usage (with cropping enabled by default)
python rectify.py

# With custom options
python rectify.py --input ./DATA/data_04_06_2025 --ext dng --debug --anms-keypoints 300

# Disable ANMS
python rectify.py --no-anms

# Disable cropping (keep black borders)
python rectify.py --no-crop
"""

import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import rawpy
from typing import Tuple, Optional, List, Dict, Any
import argparse
from dataclasses import dataclass


@dataclass
class ProcessingOptions:
    """Configuration options for image processing."""
    save_rectified: bool = True
    save_warped: bool = True
    save_overlay: bool = True
    save_aerochrome: bool = True
    use_gdi_sift: bool = False
    use_anms: bool = True
    anms_num_keypoints: int = 500
    anms_suppression: float = 0.9
    min_match_count: int = 10
    debug_visual: bool = False
    verbose: bool = False
    # DNG enhancement options
    dng_enhance_contrast: bool = True
    dng_brightness: float = 1.2
    dng_exposure_shift: float = 0.3
    # Cropping options
    crop_to_valid_region: bool = True  # Enable cropping to remove black borders
    crop_preserve_ratio: float = 0.92  # Preserve at least 92% of pixels when cropping


class ImageLoader:
    """Handles loading and processing of DNG and JPEG images."""

    def __init__(self, options: ProcessingOptions = None):
        self.options = options or ProcessingOptions()

    def load_rgb_from_dng(self, dng_path: str) -> np.ndarray:
        """
        Load an RGB DNG file and process it to linear RGB (converted to BGR for OpenCV).
        
        Args:
            dng_path: Path to RGB DNG file
            
        Returns:
            RGB image in [0,1] float32 format, shape (H, W, 3)
        """
        with rawpy.imread(dng_path) as raw:
            # Process with gamma correction for better feature detection
            # Using gamma=2.2 instead of linear for improved contrast
            rgb = raw.postprocess(
                gamma=(1, 1),  # Apply gamma correction for better contrast
                no_auto_bright=False,  # Allow auto brightness for better feature detection
                output_bps=16,
                use_camera_wb=True,
                bright=self.options.dng_brightness,  # Configurable brightness
                exp_shift=self.options.dng_exposure_shift  # Configurable exposure compensation
            )
            # Convert to float32 and normalize to [0,1]
            rgb = rgb.astype(np.float32) / 65535.0

            # Apply histogram stretching for better contrast if enabled
            if self.options.dng_enhance_contrast:
                rgb = self._enhance_contrast(rgb)

            # Convert from RGB to BGR for OpenCV
            rgb = rgb[..., ::-1]
            return rgb

    def load_nir_from_dng(self, dng_path: str) -> np.ndarray:
        """
        Load a NIR DNG file from monochrome sensor and process it.

        Args:
            dng_path: Path to NIR DNG file

        Returns:
            NIR image in [0,1] float32 format, shape (H, W, 3) for compatibility
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

            # Apply contrast enhancement for better feature detection if enabled
            if self.options.dng_enhance_contrast:
                nir_raw = self._enhance_contrast_single_channel(nir_raw)

            # Convert to 3-channel for compatibility with OpenCV operations
            nir_bgr = cv2.merge([nir_raw, nir_raw, nir_raw])

            return nir_bgr

    @staticmethod
    def load_nir_from_jpeg(image_path: str) -> np.ndarray:
        """
        Load a NIR JPEG image and extract the cleanest monochromatic channel.

        Many NIR JPEG images captured from monochromatic sensors may have color
        artifacts from inappropriate processing during capture. This function
        attempts to recover the cleanest single-channel representation.

        Args:
            image_path: Path to NIR JPEG file

        Returns:
            NIR image in [0,1] float32 format, shape (H, W, 3) for compatibility
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Convert to float32 for processing
        img = img.astype(np.float32) / 255.0

        # Extract individual channels (BGR format)
        b_channel = img[:, :, 0]
        g_channel = img[:, :, 1] 
        r_channel = img[:, :, 2]

        # For NIR cameras captured with RGB888 format, often the red channel
        # contains the cleanest NIR signal. We also check channel variance
        # to automatically select the best channel.
        channel_variances = {
            'red': np.var(r_channel),
            'green': np.var(g_channel),
            'blue': np.var(b_channel)
        }

        # Select channel with highest variance (most information content)
        best_channel_name = max(channel_variances, key=channel_variances.get)

        if best_channel_name == 'red':
            nir_channel = r_channel
        elif best_channel_name == 'green':
            nir_channel = g_channel
        else:
            nir_channel = b_channel

        # Convert to 3-channel for compatibility with OpenCV operations
        nir_bgr = cv2.merge([nir_channel, nir_channel, nir_channel])

        return nir_bgr

    def load_image_smart(self, image_path: str) -> np.ndarray:
        """
        Load an image file, automatically detecting if it's JPEG or DNG.
        
        Args:
            image_path: Path to image file

        Returns:
            Image in [0,1] float32 format, shape (H, W, 3)
        """
        file_ext = os.path.splitext(image_path)[1].lower()

        if file_ext == '.dng':
            # Determine if it's RGB or NIR based on filename
            filename_lower = os.path.basename(image_path).lower()
            if '_rgb' in filename_lower:
                return self.load_rgb_from_dng(image_path)
            elif '_ir' in filename_lower or '_nir' in filename_lower:
                return self.load_nir_from_dng(image_path)
            else:
                # Default to RGB processing
                return self.load_rgb_from_dng(image_path)
        else:
            # Determine if it's RGB or NIR based on filename for JPEG/PNG
            filename_lower = os.path.basename(image_path).lower()
            if '_ir' in filename_lower or '_nir' in filename_lower:
                return self.load_nir_from_jpeg(image_path)
            else:
                # Load RGB JPEG/PNG with OpenCV and normalize
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not load image: {image_path}")
                img = img.astype(np.float32) / 255.0
                return img

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Apply histogram stretching for better contrast in multi-channel images.

        Args:
            img: Input image in [0,1] float32 format

        Returns:
            Contrast-enhanced image in [0,1] float32 format
        """
        enhanced = img.copy()

        # Apply per-channel histogram stretching
        for ch in range(img.shape[2]):
            channel = img[:, :, ch]

            # Calculate percentiles for robust stretching
            p2, p98 = np.percentile(channel, (2, 98))

            # Avoid division by zero
            if p98 > p2:
                # Stretch to full range
                channel_stretched = (channel - p2) / (p98 - p2)
                enhanced[:, :, ch] = np.clip(channel_stretched, 0, 1)

        return enhanced

    def _enhance_contrast_single_channel(self, img: np.ndarray) -> np.ndarray:
        """
        Apply histogram stretching for better contrast in single-channel images.

        Args:
            img: Input single-channel image in [0,1] float32 format

        Returns:
            Contrast-enhanced image in [0,1] float32 format
        """
        # Calculate percentiles for robust stretching
        p2, p98 = np.percentile(img, (2, 98))

        # Avoid division by zero
        if p98 > p2:
            # Stretch to full range
            img_stretched = (img - p2) / (p98 - p2)
            return np.clip(img_stretched, 0, 1)
        else:
            return img

class ImageSaver:
    """Handles saving images in various formats."""

    @staticmethod
    def save_as_png16(image: np.ndarray, output_path: str) -> None:
        """
        Save a float32 [0,1] image as 16-bit PNG.

        Args:
            image: Image in [0,1] float32 format
            output_path: Output path for PNG file
        """
        img_16bit = (image * 65535).astype(np.uint16)
        cv2.imwrite(output_path, img_16bit)

    @staticmethod
    def save_as_npy(image: np.ndarray, output_path: str) -> None:
        """
        Save image as numpy array for reproducibility.

        Args:
            image: Image array
            output_path: Output path for NPY file
        """
        np.save(output_path, image)

    @staticmethod
    def save_as_jpeg(image: np.ndarray, output_path: str) -> None:
        """
        Save a float32 [0,1] image as 8-bit JPEG.
        
        Args:
            image: Image in [0,1] float32 format
            output_path: Output path for JPEG file
        """
        img_uint8 = (image * 255).astype(np.uint8)
        cv2.imwrite(output_path, img_uint8)


class GDISIFT:
    """SIFT with descriptor folding for reduced dimensionality."""

    def __init__(self, **sift_kwargs):
        self.sift = cv2.SIFT_create(**sift_kwargs)

    def detect(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> List[cv2.KeyPoint]:
        """Detect keypoints in image."""
        return self.sift.detect(img, mask)

    def compute(self, img: np.ndarray, kps: List[cv2.KeyPoint]) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Compute descriptors for keypoints."""
        _, desc128 = self.sift.compute(img, kps)
        desc64 = self._fold_bins(desc128)
        return kps, desc64

    def detectAndCompute(self, img: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors."""
        kps = self.detect(img, mask)
        return self.compute(img, kps)

    @staticmethod
    def _fold_bins(desc128: np.ndarray) -> np.ndarray:
        """Fold 128-dim SIFT descriptors to 64-dim."""
        d = desc128.reshape(-1, 16, 8)
        folded = d[:, :, :4] + d[:, :, 4:]
        folded = folded.reshape(-1, 64)
        # Renormalise
        norm = np.linalg.norm(folded, axis=1, keepdims=True) + 1e-7
        return folded / norm


class ANMSFilter:
    """Adaptive Non-Maximal Suppression for keypoint filtering."""

    @staticmethod
    def filter_keypoints(keypoints: List[cv2.KeyPoint], descriptors: np.ndarray, 
                        num_to_keep: int = 500, c: float = 0.9) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Apply Adaptive Non-Maximal Suppression (ANMS) to OpenCV keypoints.

        Args:
            keypoints: List of cv2.KeyPoint objects
            descriptors: Corresponding descriptors
            num_to_keep: Number of keypoints to keep
            c: Suppression constant

        Returns:
            Tuple of filtered keypoints and descriptors
        """
        if len(keypoints) <= num_to_keep:
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


class VisualDebugger:
    """Handles visual debugging and display functions."""

    @staticmethod
    def show_keypoints_grid(src_img: np.ndarray, dst_img: np.ndarray,
                           kp1_before: List[cv2.KeyPoint], kp2_before: List[cv2.KeyPoint],
                           kp1_after: List[cv2.KeyPoint], kp2_after: List[cv2.KeyPoint]) -> None:
        """
        Display keypoints before and after ANMS in a 2x2 grid.

        Args:
            src_img: Source image (uint8)
            dst_img: Destination image (uint8)
            kp1_before: Source keypoints before ANMS
            kp2_before: Destination keypoints before ANMS
            kp1_after: Source keypoints after ANMS
            kp2_after: Destination keypoints after ANMS
        """
        vis_src = cv2.drawKeypoints(src_img, kp1_before, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        vis_dst = cv2.drawKeypoints(dst_img, kp2_before, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        vis_src_anms = cv2.drawKeypoints(src_img, kp1_after, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        vis_dst_anms = cv2.drawKeypoints(dst_img, kp2_after, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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
        cv2.putText(grid, 'Src before ANMS', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(grid, 'Dst before ANMS', (w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(grid, 'Src after ANMS', (10, h + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(grid, 'Dst after ANMS', (w + 10, h + 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('SIFT Keypoints (Before/After ANMS)', grid)
        print("Debug: Press any key to continue.")
        cv2.waitKey(0)
        cv2.destroyWindow('SIFT Keypoints (Before/After ANMS)')


class ImageAligner:
    """Handles SIFT-based image alignment with optional ANMS filtering."""

    def __init__(self, options: ProcessingOptions):
        self.options = options
        self.sift = GDISIFT() if options.use_gdi_sift else cv2.SIFT_create()
        self.anms_filter = ANMSFilter()
        self.debugger = VisualDebugger()

    def align_images(self, img_src: np.ndarray, img_dst: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Align source image to destination image using SIFT features.

        Args:
            img_src: Source image to be aligned
            img_dst: Destination image (reference)

        Returns:
            Tuple of (aligned_image_cropped, reference_image_cropped, homography_matrix) or (None, None, None) if alignment fails
        """
        # Convert to uint8 for SIFT if needed
        src_uint8 = self._to_uint8(img_src)
        dst_uint8 = self._to_uint8(img_dst)

        # Convert to grayscale
        gray_src = cv2.cvtColor(src_uint8, cv2.COLOR_BGR2GRAY) if src_uint8.ndim == 3 else src_uint8
        gray_dst = cv2.cvtColor(dst_uint8, cv2.COLOR_BGR2GRAY) if dst_uint8.ndim == 3 else dst_uint8

        # Detect and compute features
        kp1, des1 = self.sift.detectAndCompute(gray_src, None)
        kp2, des2 = self.sift.detectAndCompute(gray_dst, None)

        if des1 is None or des2 is None:
            if self.options.verbose:
                print("Warning: No descriptors found in one or both images")
            return None, None, None

        if self.options.verbose:
            print(f"Debug - initial SIFT keypoints: src={len(kp1)}, dst={len(kp2)}")

        # Apply ANMS if enabled
        if self.options.use_anms:
            # Store original for debug display
            kp1_orig, kp2_orig = kp1[:], kp2[:]
            
            kp1, des1 = self.anms_filter.filter_keypoints(
                kp1, des1, self.options.anms_num_keypoints, self.options.anms_suppression
            )
            kp2, des2 = self.anms_filter.filter_keypoints(
                kp2, des2, self.options.anms_num_keypoints, self.options.anms_suppression
            )

            if self.options.verbose:
                print(f"Debug - after ANMS keypoints: src={len(kp1)}, dst={len(kp2)}")

            # Visual debug if enabled
            if self.options.debug_visual:
                self.debugger.show_keypoints_grid(src_uint8, dst_uint8, kp1_orig, kp2_orig, kp1, kp2)

        # Match features
        result = self._match_and_align(kp1, des1, kp2, des2, img_src, img_dst, gray_src, gray_dst)
        
        return result

    def _to_uint8(self, img: np.ndarray) -> np.ndarray:
        """Convert float32/float64 [0,1] image to uint8."""
        if img.dtype in [np.float32, np.float64]:
            return (img * 255).astype(np.uint8)
        return img

    def _match_and_align(self, kp1: List[cv2.KeyPoint], des1: np.ndarray,
                        kp2: List[cv2.KeyPoint], des2: np.ndarray,
                        img_src: np.ndarray, img_dst: np.ndarray,
                        gray_src: np.ndarray, gray_dst: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Match features and compute alignment with cropping to remove black borders."""
        # Ensure descriptors are float32 for FLANN
        des1 = des1.astype(np.float32)
        des2 = des2.astype(np.float32)

        # FLANN matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Match and filter
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if self.options.verbose:
            print(f"Debug - raw FLANN matches: {len(matches)}")
            print(f"Debug - good matches after ratio test: {len(good_matches)}")

        # Retry without ANMS if insufficient matches
        if len(good_matches) < self.options.min_match_count and self.options.use_anms:
            if self.options.verbose:
                print("Debug - ANMS matching insufficient, retrying without ANMS filtering...")
            kp1, des1 = self.sift.detectAndCompute(gray_src, None)
            kp2, des2 = self.sift.detectAndCompute(gray_dst, None)
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
            matches = flann.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
            if self.options.verbose:
                print(f"Debug - retry good matches after ratio test: {len(good_matches)}")

        if len(good_matches) < self.options.min_match_count:
            if self.options.verbose:
                print(f"Debug - insufficient matches: {len(good_matches)} < {self.options.min_match_count}")
            return None, None, None

        # Compute homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            if self.options.verbose:
                print("Debug - homography computation failed")
            return None, None, None

        # Apply transformation
        aligned = cv2.warpPerspective(img_src, H, (img_dst.shape[1], img_dst.shape[0]))

        # Optionally crop to valid region to remove black borders
        if self.options.crop_to_valid_region:
            # Use adaptive border detection that preserves specified ratio of pixels
            crop_region = self._compute_adaptive_crop_region(aligned, self.options.crop_preserve_ratio)

            # Crop both aligned and reference images to valid region
            aligned_cropped, dst_cropped = self._crop_images_to_valid_region(aligned, img_dst, crop_region)

            # Debug info
            if self.options.verbose:
                inliers = int(mask.sum()) if mask is not None else 0
                original_pixels = img_dst.shape[0] * img_dst.shape[1]
                cropped_pixels = aligned_cropped.shape[0] * aligned_cropped.shape[1]
                preserve_ratio = cropped_pixels / original_pixels
                print(f"Debug - SIFT matches: {len(good_matches)}, inliers: {inliers}")
                print(f"Debug - Source image dtype: {img_src.dtype}, range: [{img_src.min():.3f}, {img_src.max():.3f}]")
                print(f"Debug - Aligned image dtype: {aligned.dtype}, range: [{aligned.min():.3f}, {aligned.max():.3f}]")
                print(f"Debug - Crop region (x,y,w,h): {crop_region}")
                print(f"Debug - Original size: {img_dst.shape[:2]}, Cropped size: {aligned_cropped.shape[:2]}")
                print(f"Debug - Pixel preservation: {preserve_ratio:.1%} ({cropped_pixels}/{original_pixels})")

            return aligned_cropped, dst_cropped, H
        else:
            # Return uncropped images
            if self.options.verbose:
                inliers = int(mask.sum()) if mask is not None else 0
                print(f"Debug - SIFT matches: {len(good_matches)}, inliers: {inliers}")
                print(f"Debug - Source image dtype: {img_src.dtype}, range: [{img_src.min():.3f}, {img_src.max():.3f}]")
                print(f"Debug - Aligned image dtype: {aligned.dtype}, range: [{aligned.min():.3f}, {aligned.max():.3f}]")
                print("Debug - Cropping disabled, may contain black borders")

            return aligned, img_dst, H

    def _compute_adaptive_crop_region(self, warped_image: np.ndarray, 
                                    preserve_ratio: float = 0.92) -> Tuple[int, int, int, int]:
        """
        Compute crop region that removes black borders while preserving specified ratio of pixels.

        This method works in two stages:
        1. Find the minimal bounding rectangle containing all valid (non-black) pixels
        2. If this rectangle is larger than our target size, intelligently trim it

        The algorithm prioritizes removing areas with more black pixels (common around
        the edges after homography warping) while ensuring we preserve at least the
        specified ratio of the original image pixels.

        Args:
            warped_image: The warped image with potential black borders
            preserve_ratio: Minimum ratio of original image pixels to preserve (0.92 = 92%)

        Returns:
            Tuple of (x, y, width, height) for crop region
        """
        h, w = warped_image.shape[:2]
        total_pixels = h * w
        target_pixels = int(total_pixels * preserve_ratio)

        # Convert to grayscale for analysis
        if warped_image.ndim == 3:
            gray = cv2.cvtColor((warped_image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            gray = (warped_image * 255).astype(np.uint8)

        # Create validity mask (non-black pixels)
        # Use a small threshold to account for compression artifacts
        validity_mask = gray > 5  # Threshold for "black" pixels

        # Find the boundaries of valid regions for each row and column
        row_valid = np.any(validity_mask, axis=1)  # Valid pixels in each row
        col_valid = np.any(validity_mask, axis=0)  # Valid pixels in each column

        # Find the bounds of valid regions
        valid_rows = np.where(row_valid)[0]
        valid_cols = np.where(col_valid)[0]

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            # Fallback: return center region if no valid pixels found
            crop_w = int(w * 0.8)
            crop_h = int(h * 0.8)
            return (w - crop_w) // 2, (h - crop_h) // 2, crop_w, crop_h

        # Get initial bounds from valid pixel analysis
        top_bound = valid_rows[0]
        bottom_bound = valid_rows[-1] + 1
        left_bound = valid_cols[0]
        right_bound = valid_cols[-1] + 1

        # Calculate the number of pixels in the minimal bounding rectangle containing all valid pixels
        initial_pixels = (bottom_bound - top_bound) * (right_bound - left_bound)

        if initial_pixels <= target_pixels:
            # The valid region already has fewer pixels than our preservation target.
            # Keep all valid pixels - we can't afford to lose any more.
            return left_bound, top_bound, right_bound - left_bound, bottom_bound - top_bound

        # The valid region has more pixels than our target, so we can trim some
        # to meet the preservation ratio while removing areas with more black pixels
        pixels_to_remove = initial_pixels - target_pixels

        # Strategy: trim proportionally from all sides, but prioritize removing
        # areas with more black pixels (common in trapezoid warping)

        # Calculate density of valid pixels near each border
        border_width = min(20, min(h, w) // 10)  # Adaptive border width

        # Top border density
        top_region = validity_mask[top_bound:top_bound + border_width, left_bound:right_bound]
        top_density = np.mean(top_region) if top_region.size > 0 else 0

        # Bottom border density
        bottom_region = validity_mask[max(0, bottom_bound - border_width):bottom_bound, left_bound:right_bound]
        bottom_density = np.mean(bottom_region) if bottom_region.size > 0 else 0

        # Left border density
        left_region = validity_mask[top_bound:bottom_bound, left_bound:left_bound + border_width]
        left_density = np.mean(left_region) if left_region.size > 0 else 0

        # Right border density
        right_region = validity_mask[top_bound:bottom_bound, max(0, right_bound - border_width):right_bound]
        right_density = np.mean(right_region) if right_region.size > 0 else 0

        # Calculate trim amounts based on inverse density (trim more from sparser areas)
        densities = np.array([top_density, bottom_density, left_density, right_density])
        # Avoid division by zero and ensure some trimming happens
        weights = 1.0 / (densities + 0.1)
        weights = weights / weights.sum()

        # Distribute pixel removal based on weights and geometric constraints
        current_w = right_bound - left_bound
        current_h = bottom_bound - top_bound

        # Calculate maximum safe trim for each side (don't trim more than 15% from any side)
        max_trim_ratio = 0.15
        max_trim_top = int(current_h * max_trim_ratio)
        max_trim_bottom = int(current_h * max_trim_ratio)
        max_trim_left = int(current_w * max_trim_ratio)
        max_trim_right = int(current_w * max_trim_ratio)

        # Estimate pixels removed per unit trim for each side
        trim_top = min(max_trim_top, int(weights[0] * pixels_to_remove / current_w))
        trim_bottom = min(max_trim_bottom, int(weights[1] * pixels_to_remove / current_w))
        trim_left = min(max_trim_left, int(weights[2] * pixels_to_remove / current_h))
        trim_right = min(max_trim_right, int(weights[3] * pixels_to_remove / current_h))

        # Apply trims
        final_top = top_bound + trim_top
        final_bottom = bottom_bound - trim_bottom
        final_left = left_bound + trim_left
        final_right = right_bound - trim_right

        # Ensure we have a valid region
        final_w = max(1, final_right - final_left)
        final_h = max(1, final_bottom - final_top)

        # Final check: if we're still over the target, do uniform trimming
        final_pixels = final_w * final_h
        if final_pixels > target_pixels:
            # Calculate uniform scale factor
            scale = np.sqrt(target_pixels / final_pixels)

            # Apply uniform scaling
            center_x = (final_left + final_right) / 2
            center_y = (final_top + final_bottom) / 2

            new_w = int(final_w * scale)
            new_h = int(final_h * scale)

            final_left = int(center_x - new_w / 2)
            final_right = final_left + new_w
            final_top = int(center_y - new_h / 2)
            final_bottom = final_top + new_h

        # Ensure bounds are within image
        final_left = max(0, final_left)
        final_top = max(0, final_top)
        final_right = min(w, final_right)
        final_bottom = min(h, final_bottom)

        final_w = final_right - final_left
        final_h = final_bottom - final_top

        return final_left, final_top, final_w, final_h

    def _crop_images_to_valid_region(self, img_warped: np.ndarray, img_ref: np.ndarray,
                                   crop_region: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crop both warped and reference images to the valid region.

        Args:
            img_warped: Warped source image
            img_ref: Reference destination image
            crop_region: (x, y, width, height) of crop region

        Returns:
            Tuple of (cropped_warped, cropped_reference)
        """
        x, y, w, h = crop_region

        # Ensure crop region is within bounds
        img_h, img_w = img_ref.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        # Crop both images
        warped_cropped = img_warped[y:y+h, x:x+w]
        ref_cropped = img_ref[y:y+h, x:x+w]

        return warped_cropped, ref_cropped


class CameraRectifier:
    """Handles camera rectification using calibration parameters."""
    
    def __init__(self, camera_info: Dict[str, Any], options: ProcessingOptions = None):
        self.camera_info = camera_info
        self.loader = ImageLoader(options)

    def rectify_pair(self, left_path: str, right_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify a stereo pair using camera calibration info.

        Args:
            left_path: Path to left image
            right_path: Path to right image

        Returns:
            Tuple of (rectified_left, rectified_right)
        """
        # Load images
        left_img = self.loader.load_image_smart(left_path)
        right_img = self.loader.load_image_smart(right_path)

        # Convert to uint8 for cv2.remap
        left_uint8 = self._to_uint8(left_img)
        right_uint8 = self._to_uint8(right_img)

        # Apply rectification
        rect_left = cv2.remap(left_uint8, self.camera_info['map1x'], self.camera_info['map1y'], cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_uint8, self.camera_info['map2x'], self.camera_info['map2y'], cv2.INTER_LINEAR)
        
        # Convert back to float32 [0,1] format
        rect_left = rect_left.astype(np.float32) / 255.0
        rect_right = rect_right.astype(np.float32) / 255.0

        return rect_left, rect_right

    def _to_uint8(self, img: np.ndarray) -> np.ndarray:
        """Convert float32/float64 [0,1] image to uint8."""
        if img.dtype in [np.float32, np.float64]:
            return (img * 255).astype(np.uint8)
        return img


class ImageProcessor:
    """Handles various image processing operations."""

    @staticmethod
    def calculate_average_per_channel(image: np.ndarray) -> Tuple[float, ...]:
        """
        Calculate average pixel value per channel.

        Args:
            image: Input image with shape (H, W, C)

        Returns:
            Tuple of average values for each channel
        """
        avgs = np.mean(image, axis=(0, 1))
        return tuple(avgs.tolist())

    @staticmethod
    def create_overlay(img1: np.ndarray, img2: np.ndarray, weight: float = 0.5) -> np.ndarray:
        """
        Create an overlay of two images.

        Args:
            img1: First image
            img2: Second image
            weight: Weight for first image (1-weight for second)

        Returns:
            Overlay image
        """
        # Ensure both images are in float32 [0,1] format
        if img1.dtype != np.float32:
            img1 = img1.astype(np.float32) / 255.0
        if img2.dtype != np.float32:
            img2 = img2.astype(np.float32) / 255.0

        return weight * img1 + (1 - weight) * img2

    @staticmethod
    def create_aerochrome(img_ir: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
        """
        Create an aerochrome (false color infrared) image.

        Args:
            img_ir: IR image (may be 3-channel with color artifacts)
            img_rgb: RGB image
            
        Returns:
            Aerochrome image
        """
        # Ensure both images are in float32 [0,1] format
        if img_ir.dtype != np.float32:
            img_ir = img_ir.astype(np.float32) / 255.0
        if img_rgb.dtype != np.float32:
            img_rgb = img_rgb.astype(np.float32) / 255.0

        # Extract IR channel - use best approach for monochromatic data
        if img_ir.ndim == 3:
            # For NIR images with color artifacts, extract the channel with highest variance
            r_channel = img_ir[:, :, 2]  # Red channel (typically best for NIR)
            g_channel = img_ir[:, :, 1]  # Green channel
            b_channel = img_ir[:, :, 0]  # Blue channel

            # Calculate variance to find the channel with most information
            r_var = np.var(r_channel)
            g_var = np.var(g_channel)
            b_var = np.var(b_channel)

            # Select the channel with highest variance as it likely contains the cleanest NIR signal
            if r_var >= g_var and r_var >= b_var:
                ir_channel = r_channel
            elif g_var >= b_var:
                ir_channel = g_channel
            else:
                ir_channel = b_channel
        else:
            ir_channel = img_ir

        # Extract RGB channels (note: OpenCV uses BGR order)
        blue_channel = img_rgb[:, :, 0]   # Blue channel
        green_channel = img_rgb[:, :, 1]  # Green channel  
        red_channel = img_rgb[:, :, 2]    # Red channel

        # Create aerochrome: Blue <- Green, Green <- Red, Red <- IR
        # This follows traditional false-color infrared mapping
        result = np.stack([
            green_channel,   # Blue channel <- Green
            red_channel,     # Green channel <- Red
            ir_channel       # Red channel <- IR
        ], axis=2)

        return result


class RGBNIRPipeline:
    """Main pipeline for RGB-NIR image registration and processing."""

    def __init__(self, camera_npz_path: str, options: ProcessingOptions):
        self.options = options
        self.camera_info = self._load_camera_info(camera_npz_path)
        self.rectifier = CameraRectifier(self.camera_info, options)
        self.aligner = ImageAligner(options)
        self.processor = ImageProcessor()
        self.saver = ImageSaver()

    def _load_camera_info(self, npz_path: str) -> Dict[str, Any]:
        """Load camera calibration and rectification parameters from NPZ file."""
        return dict(np.load(npz_path))

    def get_image_pairs(self, input_dir: str, file_extension: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Scan directory for image pairs.

        Args:
            input_dir: Directory containing stereo image pairs
            file_extension: Optional file extension filter

        Returns:
            List of (left_image_path, right_image_path) tuples
        """
        ir_images = []
        rgb_images = []

        if file_extension:
            # Try organized structure first
            ir_organized = glob.glob(f"{input_dir}/originals/nir_{file_extension}/*")
            rgb_organized = glob.glob(f"{input_dir}/originals/rgb_{file_extension}/*")

            # If organized structure exists, use it
            if ir_organized and rgb_organized:
                ir_images = ir_organized
                rgb_images = rgb_organized
            else:
                # Fall back to flat directory structure
                ir_images = glob.glob(f"{input_dir}/*_ir.{file_extension}")
                rgb_images = glob.glob(f"{input_dir}/*_rgb.{file_extension}")
        else:
            # Support both jpg and dng in organized structure
            ir_organized = glob.glob(f"{input_dir}/originals/nir_jpg/*") + glob.glob(f"{input_dir}/originals/nir_dng/*")
            rgb_organized = glob.glob(f"{input_dir}/originals/rgb_jpg/*") + glob.glob(f"{input_dir}/originals/rgb_dng/*")

            # If organized structure exists, use it
            if ir_organized and rgb_organized:
                ir_images = ir_organized
                rgb_images = rgb_organized
            else:
                # Fall back to flat directory structure for both formats
                ir_images = glob.glob(f"{input_dir}/*_ir.jpg") + glob.glob(f"{input_dir}/*_ir.dng")
                rgb_images = glob.glob(f"{input_dir}/*_rgb.jpg") + glob.glob(f"{input_dir}/*_rgb.dng")

        ir_images.sort()
        rgb_images.sort()

        return list(zip(ir_images, rgb_images))

    def process_image_pair(self, left_path: str, right_path: str, output_dirs: Dict[str, str]) -> bool:
        """
        Process a single image pair through the complete pipeline.

        Args:
            left_path: Path to left (IR) image
            right_path: Path to right (RGB) image
            output_dirs: Dictionary of output directories

        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Step 1: Rectify the stereo pair
            if self.options.verbose:
                print(f"Processing pair: {os.path.basename(left_path)}, {os.path.basename(right_path)}")
            rect_left, rect_right = self.rectifier.rectify_pair(left_path, right_path)

            # Save rectified images if requested
            if self.options.save_rectified and 'rectified' in output_dirs:
                self._save_rectified_images(left_path, right_path, rect_left, rect_right, output_dirs['rectified'])

            # Step 2: Align images using SIFT and crop to remove black borders
            result = self.aligner.align_images(rect_left, rect_right)

            if result[0] is None:
                if self.options.verbose:
                    print(f"Failed to align {os.path.basename(right_path)} to {os.path.basename(left_path)}")
                return False

            warped_left, rect_right_cropped, homography = result

            # Save warped image if requested
            if self.options.save_warped and 'warped' in output_dirs:
                self._save_warped_image(left_path, warped_left, output_dirs['warped'])

            # Step 3: Create overlay (using cropped images)
            if self.options.save_overlay and 'overlay' in output_dirs:
                overlay = self.processor.create_overlay(rect_right_cropped, warped_left)
                self._save_overlay_image(left_path, overlay, output_dirs['overlay'])

                # Print average values for debugging
                if self.options.verbose:
                    avg_left = self.processor.calculate_average_per_channel(rect_right_cropped)
                    avg_warped = self.processor.calculate_average_per_channel(warped_left)
                    avg_overlay = self.processor.calculate_average_per_channel(overlay)
                    print(f"Debug - rect_right_cropped avg per channel: {avg_left}")
                    print(f"Debug - warped_left avg per channel: {avg_warped}")
                    print(f"Debug - overlay avg per channel: {avg_overlay}")

            # Step 4: Create aerochrome (using cropped images)
            if self.options.save_aerochrome and 'aerochrome' in output_dirs:
                aerochrome = self.processor.create_aerochrome(warped_left, rect_right_cropped)
                self._save_aerochrome_image(left_path, aerochrome, output_dirs['aerochrome'])

                if self.options.verbose:
                    avg_aerochrome = self.processor.calculate_average_per_channel(aerochrome)
                    print(f"Debug - aerochrome avg per channel: {avg_aerochrome}")

            # Step 5: Save final RGB and NIR images for training (using cropped images)
            if 'final' in output_dirs:
                # Create subdirectories if they don't exist
                rgb_final_dir = os.path.join(output_dirs['final'], 'rgb_png')
                nir_final_dir = os.path.join(output_dirs['final'], 'nir_png')
                os.makedirs(rgb_final_dir, exist_ok=True)
                os.makedirs(nir_final_dir, exist_ok=True)

                # Get base filename without extension and remove '_ir' suffix if present
                base_name = os.path.splitext(os.path.basename(left_path))[0]
                if base_name.endswith('_ir'):
                    base_name = base_name[:-3]

                # Save RGB (rect_right_cropped) and NIR (warped_left) as 16-bit PNGs
                self.saver.save_as_png16(rect_right_cropped, os.path.join(rgb_final_dir, f"{base_name}.png"))
                self.saver.save_as_png16(warped_left, os.path.join(nir_final_dir, f"{base_name}.png"))

                # Also save as NPY for reproducibility
                os.makedirs(os.path.join(output_dirs['final'], 'rgb_npy'), exist_ok=True)
                os.makedirs(os.path.join(output_dirs['final'], 'nir_npy'), exist_ok=True)
                self.saver.save_as_npy(rect_right_cropped, os.path.join(output_dirs['final'], 'rgb_npy', f"{base_name}.npy"))
                self.saver.save_as_npy(warped_left, os.path.join(output_dirs['final'], 'nir_npy', f"{base_name}.npy"))

                if self.options.verbose:
                    print(f"Debug - saved final RGB and NIR images for {base_name}")
                    print(f"Debug - final image size: {warped_left.shape[:2]} (cropped from {rect_left.shape[:2]})")

            if self.options.verbose:
                print(f"Successfully processed: {os.path.basename(left_path)}")
            return True

        except Exception as e:
            print(f"Error processing {os.path.basename(left_path)}: {str(e)}")
            return False

    def _save_rectified_images(self, left_path: str, right_path: str, 
                              rect_left: np.ndarray, rect_right: np.ndarray, output_dir: str) -> None:
        """Save rectified images in multiple formats."""
        left_name = os.path.splitext(os.path.basename(left_path))[0]
        right_name = os.path.splitext(os.path.basename(right_path))[0]

        # Save as 16-bit PNG
        self.saver.save_as_png16(rect_left, os.path.join(output_dir, f"{left_name}_rect.png"))
        self.saver.save_as_png16(rect_right, os.path.join(output_dir, f"{right_name}_rect.png"))

        # Save as NPY for reproducibility
        self.saver.save_as_npy(rect_left, os.path.join(output_dir, f"{left_name}_rect.npy"))
        self.saver.save_as_npy(rect_right, os.path.join(output_dir, f"{right_name}_rect.npy"))

    def _save_warped_image(self, left_path: str, warped_left: np.ndarray, output_dir: str) -> None:
        """Save warped image in multiple formats."""
        left_name = os.path.splitext(os.path.basename(left_path))[0]

        # Save as 16-bit PNG and 8-bit JPEG
        self.saver.save_as_png16(warped_left, os.path.join(output_dir, f'{left_name}_warped.png'))
        self.saver.save_as_jpeg(warped_left, os.path.join(output_dir, f'{left_name}_warped.jpg'))

    def _save_overlay_image(self, left_path: str, overlay: np.ndarray, output_dir: str) -> None:
        """Save overlay image in multiple formats."""
        base_name = os.path.splitext(os.path.basename(left_path))[0].removesuffix('_ir')  # Remove '_ir' suffix

        self.saver.save_as_png16(overlay, os.path.join(output_dir, f"{base_name}_overlay.png"))
        self.saver.save_as_jpeg(overlay, os.path.join(output_dir, f"{base_name}_overlay.jpg"))

    def _save_aerochrome_image(self, left_path: str, aerochrome: np.ndarray, output_dir: str) -> None:
        """Save aerochrome image in multiple formats."""
        base_name = os.path.splitext(os.path.basename(left_path))[0].removesuffix('_ir') # Remove '_ir' suffix

        self.saver.save_as_png16(aerochrome, os.path.join(output_dir, f"{base_name}_aerochrome.png"))
        self.saver.save_as_jpeg(aerochrome, os.path.join(output_dir, f"{base_name}_aerochrome.jpg"))

    def process_dataset(self, input_dir: str, output_dirs: Dict[str, str], 
                       file_ext: Optional[str] = None, subset: Optional[int] = None) -> None:
        """
        Process a complete dataset of stereo image pairs.

        Args:
            input_dir: Directory containing input images
            output_dirs: Dictionary mapping output types to directories
            file_ext: Optional file extension filter
            subset: Optional limit on number of image pairs to process
        """
        # Create output directories
        for output_dir in output_dirs.values():
            os.makedirs(output_dir, exist_ok=True)

        # Get image pairs
        pairs = self.get_image_pairs(input_dir, file_ext)

        if not pairs:
            print(f"No image pairs found in {input_dir}")
            return

        # Apply subset limit if specified
        if subset is not None and subset > 0:
            pairs = pairs[:subset]
            print(f"Found {len(self.get_image_pairs(input_dir, file_ext))} image pairs, processing subset of {len(pairs)}")
        else:
            print(f"Found {len(pairs)} image pairs to process")

        # Process each pair
        successful = 0
        for left_path, right_path in tqdm(pairs, desc="Processing image pairs"):
            if self.process_image_pair(left_path, right_path, output_dirs):
                successful += 1

        print(f"Successfully processed {successful} out of {len(pairs)} image pairs")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="RGB-NIR Image Registration Pipeline")
    parser.add_argument('--input', type=str, default="./DATA/data_04_06_2025", 
                       help='Input directory containing image pairs')
    parser.add_argument('--npz', type=str, default="./calibration_files/calib_data7.npz",
                       help='Path to camera calibration NPZ file')
    parser.add_argument('--ext', type=str, default=None,
                       help='File extension to process (jpg, dng, etc.)')
    parser.add_argument('-gdi-sift', action='store_true',
                       help='Use GDI SIFT for feature detection')
    parser.add_argument('--anms', action='store_true',
                       help='Enable Adaptive Non-Maximal Suppression')
    parser.add_argument('--anms-keypoints', type=int, default=500,
                       help='Number of keypoints to keep with ANMS')
    parser.add_argument('--anms-suppression', type=float, default=0.9,
                       help='ANMS suppression constant')
    parser.add_argument('--min-matches', type=int, default=10,
                       help='Minimum number of matches required')
    parser.add_argument('--debug', action='store_true',
                       help='Enable visual debugging')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose debug output')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark instead of pipeline')
    parser.add_argument('--subset', type=int, default=None,
                       help='Number of image pairs to use for benchmark')
    # DNG enhancement options
    parser.add_argument('--no-dng-enhance', action='store_true',
                       help='Disable DNG contrast enhancement')
    parser.add_argument('--dng-brightness', type=float, default=1.2,
                       help='DNG brightness multiplier (default: 1.2)')
    parser.add_argument('--dng-exposure', type=float, default=0.3,
                       help='DNG exposure shift (default: 0.3)')
    parser.add_argument('--no-crop', action='store_true',
                       help='Disable cropping to valid region (may result in black borders)')
    parser.add_argument('--crop-preserve-ratio', type=float, default=0.92,
                       help='Minimum ratio of pixels to preserve when cropping (default: 0.92 = 92%%)')


    args = parser.parse_args()

    # Configure processing options
    options = ProcessingOptions(
        use_gdi_sift=args.gdi_sift,
        use_anms= args.anms,
        anms_num_keypoints=args.anms_keypoints,
        anms_suppression=args.anms_suppression,
        min_match_count=args.min_matches,
        debug_visual=args.debug,
        verbose=args.verbose,
        dng_enhance_contrast=not args.no_dng_enhance,
        dng_brightness=args.dng_brightness,
        dng_exposure_shift=args.dng_exposure,
        crop_to_valid_region=not args.no_crop,
        crop_preserve_ratio=args.crop_preserve_ratio
    )

    if args.benchmark:
        print("Benchmark functionality not implemented in refactored version")
        return

    # Define output directories
    output_directories = {
        'rectified': f"{args.input}/processed/rectified",
        'warped': f"{args.input}/processed/warped",
        'overlay': f"{args.input}/processed/overlay",
        'aerochrome': f"{args.input}/processed/aerochrome",
        'final': f"{args.input}/processed/final"
    }

    # Create and run pipeline
    try:
        pipeline = RGBNIRPipeline(args.npz, options)
        pipeline.process_dataset(args.input, output_directories, args.ext, args.subset)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
