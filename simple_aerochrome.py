#!/usr/bin/env python3
"""
Simple Aerochrome Generator for Already-Aligned RGB-NIR Image Pairs

This script creates aerochrome (false color infrared) images from pairs of
already-aligned RGB and NIR images. Unlike the full rectify.py pipeline,
this bypasses camera rectification and SIFT alignment since the images
are already aligned.

Usage:
    python simple_aerochrome.py --rgb-dir path/to/rgb --nir-dir path/to/nir --output-dir path/to/output
    
    # Or if images are in the same directory with different naming patterns:
    python simple_aerochrome.py --input-dir path/to/images --rgb-pattern "*_rgb.png" --nir-pattern "*_nir.png" --output-dir path/to/output
"""

import os
import cv2
import numpy as np
import argparse
import glob
from pathlib import Path
from typing import List, Tuple, Optional
import re


class SimpleImageLoader:
    """Simple image loader for aligned RGB-NIR pairs."""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load image and convert to float32 [0,1] format."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return img.astype(np.float32) / 255.0
    
    @staticmethod
    def extract_nir_channel(nir_image: np.ndarray) -> np.ndarray:
        """
        Extract the best NIR channel from a potentially multi-channel NIR image.
        
        For generated NIR images, they may be saved as 3-channel but contain
        the same data in all channels, or the cleanest signal in one channel.
        """
        if nir_image.ndim == 2:
            return nir_image
        elif nir_image.ndim == 3:
            # Check if all channels are the same (grayscale saved as RGB)
            r_channel = nir_image[:, :, 2]
            g_channel = nir_image[:, :, 1] 
            b_channel = nir_image[:, :, 0]
            
            # If channels are very similar, just use one
            if np.allclose(r_channel, g_channel, atol=0.01) and np.allclose(g_channel, b_channel, atol=0.01):
                return r_channel
            
            # Otherwise, select channel with highest variance (most information)
            channel_variances = {
                'red': np.var(r_channel),
                'green': np.var(g_channel),
                'blue': np.var(b_channel)
            }
            
            best_channel = max(channel_variances, key=channel_variances.get)
            if best_channel == 'red':
                return r_channel
            elif best_channel == 'green':
                return g_channel
            else:
                return b_channel
        else:
            raise ValueError(f"Unsupported image dimensions: {nir_image.shape}")


class AerochromeGenerator:
    """Generates aerochrome (false color infrared) images."""
    
    @staticmethod
    def create_aerochrome(rgb_image: np.ndarray, nir_image: np.ndarray) -> np.ndarray:
        """
        Create aerochrome image from RGB and NIR images.
        
        Standard aerochrome mapping:
        - Blue channel <- Green channel (from RGB)
        - Green channel <- Red channel (from RGB)  
        - Red channel <- NIR channel
        
        Args:
            rgb_image: RGB image in [0,1] float32 format, shape (H, W, 3)
            nir_image: NIR image in [0,1] float32 format, any shape
            
        Returns:
            Aerochrome image in [0,1] float32 format, shape (H, W, 3)
        """
        # Ensure images are same size
        if rgb_image.shape[:2] != nir_image.shape[:2]:
            # Resize NIR to match RGB
            nir_image = cv2.resize(nir_image, (rgb_image.shape[1], rgb_image.shape[0]))
        
        # Extract NIR channel (handles both single and multi-channel NIR images)
        loader = SimpleImageLoader()
        nir_channel = loader.extract_nir_channel(nir_image)
        
        # Extract RGB channels (OpenCV uses BGR order)
        blue_channel = rgb_image[:, :, 0]   # Blue
        green_channel = rgb_image[:, :, 1]  # Green
        red_channel = rgb_image[:, :, 2]    # Red
        
        # Create aerochrome with standard mapping
        aerochrome = np.stack([
            green_channel,   # Blue <- Green
            red_channel,     # Green <- Red  
            nir_channel      # Red <- NIR
        ], axis=2)
        
        return aerochrome


class ImageSaver:
    """Handles saving images in various formats."""
    
    @staticmethod
    def save_as_png(image: np.ndarray, output_path: str, bit_depth: int = 8) -> None:
        """Save image as PNG with specified bit depth."""
        if bit_depth == 16:
            img_out = (np.clip(image, 0, 1) * 65535).astype(np.uint16)
        else:  # 8-bit
            img_out = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(output_path, img_out)
    
    @staticmethod 
    def save_as_jpeg(image: np.ndarray, output_path: str, quality: int = 95) -> None:
        """Save image as JPEG."""
        img_out = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(output_path, img_out, [cv2.IMWRITE_JPEG_QUALITY, quality])


def find_image_pairs_separate_dirs(rgb_dir: str, nir_dir: str) -> List[Tuple[str, str]]:
    """Find matching RGB-NIR pairs from separate directories."""
    rgb_images = glob.glob(os.path.join(rgb_dir, "*.png")) + \
                 glob.glob(os.path.join(rgb_dir, "*.jpg")) + \
                 glob.glob(os.path.join(rgb_dir, "*.jpeg"))
    
    nir_images = glob.glob(os.path.join(nir_dir, "*.png")) + \
                 glob.glob(os.path.join(nir_dir, "*.jpg")) + \
                 glob.glob(os.path.join(nir_dir, "*.jpeg"))
    
    # Extract base names (without extension) for matching
    rgb_bases = {os.path.splitext(os.path.basename(img))[0]: img for img in rgb_images}
    nir_bases = {os.path.splitext(os.path.basename(img))[0]: img for img in nir_images}
    
    # Find matching pairs
    pairs = []
    for base_name in rgb_bases:
        if base_name in nir_bases:
            pairs.append((rgb_bases[base_name], nir_bases[base_name]))
    
    return sorted(pairs)


def find_image_pairs_same_dir(input_dir: str, rgb_pattern: str, nir_pattern: str) -> List[Tuple[str, str]]:
    """Find matching RGB-NIR pairs from same directory using patterns."""
    rgb_images = glob.glob(os.path.join(input_dir, rgb_pattern))
    nir_images = glob.glob(os.path.join(input_dir, nir_pattern))
    
    # Extract identifiers from filenames for matching
    def extract_identifier(filename: str, pattern: str) -> str:
        """Extract the identifying part of a filename given a pattern."""
        # Convert glob pattern to regex and extract the part that matches *
        # First, escape special regex characters except *
        regex_pattern = re.escape(pattern)
        # Then replace escaped \* with capture group
        regex_pattern = regex_pattern.replace(r"\*", r"(.*?)")
        
        match = re.search(regex_pattern, os.path.basename(filename))
        return match.group(1) if match else os.path.splitext(os.path.basename(filename))[0]
    
    rgb_ids = {extract_identifier(img, rgb_pattern): img for img in rgb_images}
    nir_ids = {extract_identifier(img, nir_pattern): img for img in nir_images}
    
    # Find matching pairs
    pairs = []
    for img_id in rgb_ids:
        if img_id in nir_ids:
            pairs.append((rgb_ids[img_id], nir_ids[img_id]))
    
    return sorted(pairs)


def process_image_pair(rgb_path: str, nir_path: str, output_dir: str, 
                      save_png: bool = True, save_jpeg: bool = True, bit_depth: int = 8) -> bool:
    """Process a single RGB-NIR pair to create aerochrome."""
    try:
        # Load images
        loader = SimpleImageLoader()
        rgb_image = loader.load_image(rgb_path)
        nir_image = loader.load_image(nir_path)
        
        # Generate aerochrome
        generator = AerochromeGenerator()
        aerochrome = generator.create_aerochrome(rgb_image, nir_image)
        
        # Prepare output filename
        rgb_basename = os.path.splitext(os.path.basename(rgb_path))[0]
        # Remove common suffixes like "_rgb" if present
        clean_name = rgb_basename.replace("_rgb", "").replace("_RGB", "")
        
        # Save in requested formats
        saver = ImageSaver()
        success = False
        
        if save_png:
            png_path = os.path.join(output_dir, f"{clean_name}_aerochrome.png")
            saver.save_as_png(aerochrome, png_path, bit_depth)
            success = True
            
        if save_jpeg:
            jpeg_path = os.path.join(output_dir, f"{clean_name}_aerochrome.jpg")
            saver.save_as_jpeg(aerochrome, jpeg_path)
            success = True
        
        if success:
            print(f"✓ Created aerochrome: {clean_name}")
            return True
        else:
            print(f"✗ No output format specified for: {clean_name}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {os.path.basename(rgb_path)}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate aerochrome images from aligned RGB-NIR pairs")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--separate-dirs", action="store_true", 
                           help="RGB and NIR images are in separate directories")
    input_group.add_argument("--same-dir", action="store_true",
                           help="RGB and NIR images are in the same directory with different patterns")
    
    # Directory arguments for separate directories
    parser.add_argument("--rgb-dir", type=str, help="Directory containing RGB images")
    parser.add_argument("--nir-dir", type=str, help="Directory containing NIR images") 
    
    # Arguments for same directory
    parser.add_argument("--input-dir", type=str, help="Directory containing both RGB and NIR images")
    parser.add_argument("--rgb-pattern", type=str, default="*_rgb.*", 
                       help="Pattern for RGB images (default: *_rgb.*)")
    parser.add_argument("--nir-pattern", type=str, default="*_nir.*",
                       help="Pattern for NIR images (default: *_nir.*)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for aerochrome images")
    parser.add_argument("--format", choices=["png", "jpeg", "both"], default="both",
                       help="Output format (default: both)")
    parser.add_argument("--bit-depth", type=int, choices=[8, 16], default=8,
                       help="Bit depth for PNG output (default: 8)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.separate_dirs:
        if not args.rgb_dir or not args.nir_dir:
            parser.error("--separate-dirs requires --rgb-dir and --nir-dir")
        if not os.path.exists(args.rgb_dir):
            parser.error(f"RGB directory does not exist: {args.rgb_dir}")
        if not os.path.exists(args.nir_dir):
            parser.error(f"NIR directory does not exist: {args.nir_dir}")
            
    if args.same_dir:
        if not args.input_dir:
            parser.error("--same-dir requires --input-dir")
        if not os.path.exists(args.input_dir):
            parser.error(f"Input directory does not exist: {args.input_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find image pairs
    if args.separate_dirs:
        pairs = find_image_pairs_separate_dirs(args.rgb_dir, args.nir_dir)
        print(f"Found {len(pairs)} RGB-NIR pairs in separate directories")
    else:
        pairs = find_image_pairs_same_dir(args.input_dir, args.rgb_pattern, args.nir_pattern)
        print(f"Found {len(pairs)} RGB-NIR pairs matching patterns")
    
    if not pairs:
        print("No matching image pairs found!")
        return 1
    
    # Process pairs
    save_png = args.format in ["png", "both"]
    save_jpeg = args.format in ["jpeg", "both"]
    
    success_count = 0
    for rgb_path, nir_path in pairs:
        if process_image_pair(rgb_path, nir_path, args.output_dir, save_png, save_jpeg, args.bit_depth):
            success_count += 1
    
    print(f"\n🎉 Successfully processed {success_count}/{len(pairs)} image pairs")
    print(f"Aerochrome images saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())