#!/usr/bin/env python3
"""
Interactive Image Pair Viewer for RGB-NIR Data Quality Assessment
==================================================================

A Streamlit-based tool for reviewing RGB-NIR image pairs with:
- 2x2 grid layout: RGB, NIR, Overlay, Aerochrome
- Navigation and keyboard shortcuts
- Quality marking and batch deletion
- Multiple overlay visualization modes

Usage:
    streamlit run view_pairs.py -- --data-dir DATA/alldata

    # Or with custom subdirectory names:
    streamlit run view_pairs.py -- --data-dir DATA/alldata --rgb-subdir rgb_png --nir-subdir nir_png
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import shutil

import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Import aerochrome generator from existing code
from simple_aerochrome import AerochromeGenerator, SimpleImageLoader


# ============================================================================
# Configuration and State Management
# ============================================================================

class ViewerConfig:
    """Configuration for the image pair viewer."""
    def __init__(self, data_dir: str, rgb_subdir: str = "rgb_png", nir_subdir: str = "nir_png"):
        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / rgb_subdir
        self.nir_dir = self.data_dir / nir_subdir
        self.marked_file = self.data_dir / "marked_for_deletion.json"

        # Validate directories
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        if not self.rgb_dir.exists():
            raise ValueError(f"RGB directory does not exist: {self.rgb_dir}")
        if not self.nir_dir.exists():
            raise ValueError(f"NIR directory does not exist: {self.nir_dir}")


# ============================================================================
# Image Pair Management
# ============================================================================

class ImagePairManager:
    """Manages loading and tracking of RGB-NIR image pairs."""

    def __init__(self, config: ViewerConfig):
        self.config = config
        self.pairs = self._find_pairs()
        self.marked_pairs = self._load_marked()

    def _find_pairs(self) -> List[Tuple[str, str, str]]:
        """Find all matching RGB-NIR pairs. Returns list of (basename, rgb_path, nir_path)."""
        rgb_files = {f.stem: f for f in self.config.rgb_dir.glob("*.png")}
        nir_files = {f.stem: f for f in self.config.nir_dir.glob("*.png")}

        # Find matching pairs
        pairs = []
        for basename in sorted(rgb_files.keys()):
            if basename in nir_files:
                pairs.append((basename, str(rgb_files[basename]), str(nir_files[basename])))

        return pairs

    def _load_marked(self) -> Dict:
        """Load marked pairs from JSON file."""
        if self.config.marked_file.exists():
            with open(self.config.marked_file, 'r') as f:
                return json.load(f)
        return {}

    def save_marked(self):
        """Save marked pairs to JSON file."""
        with open(self.config.marked_file, 'w') as f:
            json.dump(self.marked_pairs, f, indent=2)

    def mark_pair(self, basename: str, reason: str):
        """Mark a pair for deletion."""
        self.marked_pairs[basename] = {
            "reason": reason,
            "timestamp": str(basename)
        }
        self.save_marked()

    def unmark_pair(self, basename: str):
        """Unmark a pair."""
        if basename in self.marked_pairs:
            del self.marked_pairs[basename]
            self.save_marked()

    def is_marked(self, basename: str) -> bool:
        """Check if pair is marked for deletion."""
        return basename in self.marked_pairs

    def get_marked_count(self) -> int:
        """Get count of marked pairs."""
        return len(self.marked_pairs)

    def move_marked_pairs(self, dest_dir: str = "rejected") -> Tuple[int, int]:
        """
        Move all marked pairs to a separate directory.
        Returns (success_count, fail_count).
        """
        # Create destination directory structure
        dest_path = self.config.data_dir / dest_dir
        dest_rgb_dir = dest_path / "rgb_png"
        dest_nir_dir = dest_path / "nir_png"

        dest_rgb_dir.mkdir(parents=True, exist_ok=True)
        dest_nir_dir.mkdir(parents=True, exist_ok=True)

        success = 0
        failed = 0

        for basename in list(self.marked_pairs.keys()):
            rgb_src = self.config.rgb_dir / f"{basename}.png"
            nir_src = self.config.nir_dir / f"{basename}.png"
            rgb_dst = dest_rgb_dir / f"{basename}.png"
            nir_dst = dest_nir_dir / f"{basename}.png"

            try:
                if rgb_src.exists():
                    shutil.move(str(rgb_src), str(rgb_dst))
                if nir_src.exists():
                    shutil.move(str(nir_src), str(nir_dst))
                success += 1
            except Exception as e:
                print(f"Failed to move {basename}: {e}")
                failed += 1

        # Clear marked list after successful move
        if failed == 0:
            self.marked_pairs.clear()
            self.save_marked()

        return success, failed

    def delete_marked_pairs(self) -> Tuple[int, int]:
        """
        Delete all marked pairs from disk.
        Returns (success_count, fail_count).
        """
        success = 0
        failed = 0

        for basename in list(self.marked_pairs.keys()):
            rgb_path = self.config.rgb_dir / f"{basename}.png"
            nir_path = self.config.nir_dir / f"{basename}.png"

            try:
                if rgb_path.exists():
                    rgb_path.unlink()
                if nir_path.exists():
                    nir_path.unlink()
                success += 1
            except Exception as e:
                print(f"Failed to delete {basename}: {e}")
                failed += 1

        # Clear marked list after deletion
        if failed == 0:
            self.marked_pairs.clear()
            self.save_marked()

        return success, failed


# ============================================================================
# Image Processing
# ============================================================================

class ImageProcessor:
    """Handles image loading and processing operations."""

    def __init__(self):
        self.loader = SimpleImageLoader()
        self.aerochrome_gen = AerochromeGenerator()

    def load_pair(self, rgb_path: str, nir_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load RGB and NIR images."""
        rgb = self.loader.load_image(rgb_path)
        nir = self.loader.load_image(nir_path)
        return rgb, nir

    def generate_aerochrome(self, rgb: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """Generate aerochrome false-color image."""
        return self.aerochrome_gen.create_aerochrome(rgb, nir)

    def create_overlay(self, rgb: np.ndarray, nir: np.ndarray, mode: str = "blend") -> np.ndarray:
        """
        Create overlay visualization of RGB and NIR images.

        Modes:
        - blend: Alpha blending
        - difference: Absolute difference
        - checkerboard: Checkerboard pattern
        """
        # Ensure same size
        if rgb.shape[:2] != nir.shape[:2]:
            nir = cv2.resize(nir, (rgb.shape[1], rgb.shape[0]))

        # Convert NIR to 3-channel for visualization
        nir_channel = self.loader.extract_nir_channel(nir)
        nir_3ch = np.stack([nir_channel, nir_channel, nir_channel], axis=2)

        if mode == "blend":
            # 50% alpha blend
            overlay = 0.5 * rgb + 0.5 * nir_3ch

        elif mode == "difference":
            # Absolute difference, normalized
            diff = np.abs(rgb - nir_3ch)
            # Amplify for visibility
            overlay = np.clip(diff * 3, 0, 1)

        elif mode == "checkerboard":
            # Create checkerboard mask (32x32 pixel squares)
            h, w = rgb.shape[:2]
            square_size = 32
            mask = np.zeros((h, w), dtype=bool)
            for i in range(0, h, square_size):
                for j in range(0, w, square_size):
                    if ((i // square_size) + (j // square_size)) % 2 == 0:
                        mask[i:i+square_size, j:j+square_size] = True

            overlay = rgb.copy()
            mask_3ch = np.stack([mask, mask, mask], axis=2)
            overlay[mask_3ch] = nir_3ch[mask_3ch]

        else:
            overlay = rgb

        return overlay

    def numpy_to_pil(self, img: np.ndarray) -> Image.Image:
        """Convert numpy array [0,1] to PIL Image."""
        img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        # Convert BGR to RGB for display
        if img.ndim == 3:
            img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_uint8)


# ============================================================================
# Streamlit UI
# ============================================================================

def init_session_state(pair_manager: ImagePairManager):
    """Initialize Streamlit session state."""
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'overlay_mode' not in st.session_state:
        st.session_state.overlay_mode = "blend"
    if 'filter_mode' not in st.session_state:
        st.session_state.filter_mode = "all"
    if 'pair_manager' not in st.session_state:
        st.session_state.pair_manager = pair_manager


def get_filtered_pairs(pair_manager: ImagePairManager, filter_mode: str) -> List[Tuple[str, str, str]]:
    """Get pairs based on filter mode."""
    if filter_mode == "marked":
        return [p for p in pair_manager.pairs if pair_manager.is_marked(p[0])]
    elif filter_mode == "unmarked":
        return [p for p in pair_manager.pairs if not pair_manager.is_marked(p[0])]
    else:  # "all"
        return pair_manager.pairs


def render_navigation(pair_manager: ImagePairManager, filtered_pairs: List):
    """Render navigation controls."""
    st.sidebar.header("Navigation")

    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        if st.button("⬅ Previous", use_container_width=True):
            st.session_state.current_index = max(0, st.session_state.current_index - 1)
            st.rerun()

    with col2:
        st.write(f"{st.session_state.current_index + 1}/{len(filtered_pairs)}")

    with col3:
        if st.button("Next ➡", use_container_width=True):
            st.session_state.current_index = min(len(filtered_pairs) - 1, st.session_state.current_index + 1)
            st.rerun()

    # Jump to index
    jump_index = st.sidebar.number_input(
        "Jump to index:",
        min_value=1,
        max_value=len(filtered_pairs),
        value=st.session_state.current_index + 1,
        step=1
    )
    if st.sidebar.button("Go", use_container_width=True):
        st.session_state.current_index = jump_index - 1
        st.rerun()


def render_marking_controls(pair_manager: ImagePairManager, basename: str):
    """Render marking/deletion controls."""
    st.sidebar.header("Quality Control")

    is_marked = pair_manager.is_marked(basename)

    if is_marked:
        st.sidebar.warning(f"⚠️ Marked for deletion")
        reason = pair_manager.marked_pairs[basename].get("reason", "No reason")
        st.sidebar.text(f"Reason: {reason}")

        if st.sidebar.button("✓ Unmark", use_container_width=True, type="primary"):
            pair_manager.unmark_pair(basename)
            st.rerun()
    else:
        st.sidebar.success("✓ Not marked")

        reason = st.sidebar.selectbox(
            "Mark reason:",
            ["Blurry", "Misaligned", "Bad exposure", "Artifacts", "Privacy", "Other"],
            key="mark_reason"
        )

        if st.sidebar.button("✗ Mark for deletion", use_container_width=True, type="secondary"):
            pair_manager.mark_pair(basename, reason)
            st.rerun()


def render_statistics(pair_manager: ImagePairManager, filtered_pairs: List):
    """Render statistics panel."""
    st.sidebar.header("Statistics")

    total_pairs = len(pair_manager.pairs)
    marked_count = pair_manager.get_marked_count()
    marked_pct = (marked_count / total_pairs * 100) if total_pairs > 0 else 0

    st.sidebar.metric("Total pairs", total_pairs)
    st.sidebar.metric("Marked for deletion", f"{marked_count} ({marked_pct:.1f}%)")
    st.sidebar.metric("Filtered view", len(filtered_pairs))


def render_filter_controls():
    """Render filter controls."""
    st.sidebar.header("Filters")

    filter_mode = st.sidebar.radio(
        "Show:",
        ["all", "unmarked", "marked"],
        index=["all", "unmarked", "marked"].index(st.session_state.filter_mode),
        key="filter_radio"
    )

    if filter_mode != st.session_state.filter_mode:
        st.session_state.filter_mode = filter_mode
        st.session_state.current_index = 0
        st.rerun()


def render_batch_operations(pair_manager: ImagePairManager):
    """Render batch operation controls."""
    st.sidebar.header("Batch Operations")

    marked_count = pair_manager.get_marked_count()

    if marked_count > 0:
        st.sidebar.warning(f"⚠️ {marked_count} pairs marked")

        # Export to JSON
        if st.sidebar.button("💾 Export marked list", use_container_width=True):
            st.sidebar.success(f"Saved to: {pair_manager.config.marked_file}")

        st.sidebar.divider()

        # Move marked pairs (safer option)
        st.sidebar.subheader("Move to Rejected")
        dest_folder = st.sidebar.text_input("Destination folder:", value="rejected", key="dest_folder")

        if st.sidebar.button(
            f"📁 Move {marked_count} pairs to '{dest_folder}'",
            use_container_width=True,
            type="primary"
        ):
            with st.spinner("Moving files..."):
                success, failed = pair_manager.move_marked_pairs(dest_folder)

            if failed == 0:
                st.sidebar.success(f"✓ Moved {success} pairs to {pair_manager.config.data_dir / dest_folder}")
                # Refresh pairs list
                pair_manager.pairs = pair_manager._find_pairs()
                st.session_state.current_index = 0
                st.rerun()
            else:
                st.sidebar.error(f"Failed to move {failed} pairs")

        # Delete marked pairs (danger zone)
        st.sidebar.divider()
        st.sidebar.error("⚠️ Danger Zone")

        confirm = st.sidebar.checkbox("I understand this will permanently delete files", key="confirm_delete")

        if st.sidebar.button(
            f"🗑️ Delete {marked_count} marked pairs permanently",
            use_container_width=True,
            type="secondary",
            disabled=not confirm
        ):
            with st.spinner("Deleting files..."):
                success, failed = pair_manager.delete_marked_pairs()

            if failed == 0:
                st.sidebar.success(f"✓ Deleted {success} pairs")
                # Refresh pairs list
                pair_manager.pairs = pair_manager._find_pairs()
                st.session_state.current_index = 0
                st.rerun()
            else:
                st.sidebar.error(f"Failed to delete {failed} pairs")
    else:
        st.sidebar.info("No pairs marked for deletion")


def render_overlay_controls():
    """Render overlay mode controls."""
    st.sidebar.header("Overlay Mode")

    overlay_mode = st.sidebar.radio(
        "Visualization:",
        ["blend", "difference", "checkerboard"],
        format_func=lambda x: {
            "blend": "Alpha Blend",
            "difference": "Difference Map",
            "checkerboard": "Checkerboard"
        }[x],
        index=["blend", "difference", "checkerboard"].index(st.session_state.overlay_mode),
        key="overlay_radio"
    )

    if overlay_mode != st.session_state.overlay_mode:
        st.session_state.overlay_mode = overlay_mode
        st.rerun()


def render_image_grid(rgb: np.ndarray, nir: np.ndarray, overlay: np.ndarray,
                     aerochrome: np.ndarray, processor: ImageProcessor, basename: str):
    """Render 2x2 image grid."""
    st.header(f"Image Pair: {basename}")

    # Top row
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RGB Original")
        st.image(processor.numpy_to_pil(rgb), use_container_width=True)

    with col2:
        st.subheader("NIR Original")
        st.image(processor.numpy_to_pil(nir), use_container_width=True)

    # Bottom row
    col3, col4 = st.columns(2)
    with col3:
        st.subheader(f"Overlay ({st.session_state.overlay_mode})")
        st.image(processor.numpy_to_pil(overlay), use_container_width=True)

    with col4:
        st.subheader("Aerochrome (False Color)")
        st.image(processor.numpy_to_pil(aerochrome), use_container_width=True)


# ============================================================================
# Main Application
# ============================================================================

def main():
    st.set_page_config(
        page_title="RGB-NIR Image Pair Viewer",
        page_icon="🖼️",
        layout="wide"
    )

    st.title("🖼️ RGB-NIR Image Pair Viewer")
    st.markdown("Interactive tool for reviewing and marking image pairs for quality assessment")

    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="DATA/alldata", help="Path to data directory")
    parser.add_argument("--rgb-subdir", default="rgb_png", help="RGB subdirectory name")
    parser.add_argument("--nir-subdir", default="nir_png", help="NIR subdirectory name")

    try:
        args = parser.parse_args()
    except SystemExit:
        # Fallback to defaults if running in Streamlit Cloud
        args = argparse.Namespace(data_dir="DATA/alldata", rgb_subdir="rgb_png", nir_subdir="nir_png")

    # Initialize
    try:
        config = ViewerConfig(args.data_dir, args.rgb_subdir, args.nir_subdir)
        pair_manager = ImagePairManager(config)
        processor = ImageProcessor()

        init_session_state(pair_manager)

    except Exception as e:
        st.error(f"Error initializing viewer: {e}")
        st.info("Usage: `streamlit run view_pairs.py -- --data-dir DATA/alldata`")
        return

    # Check if we have any pairs
    if len(pair_manager.pairs) == 0:
        st.error("No image pairs found in the specified directory!")
        st.info(f"Looking for pairs in:\n- RGB: {config.rgb_dir}\n- NIR: {config.nir_dir}")
        return

    # Get filtered pairs
    filtered_pairs = get_filtered_pairs(pair_manager, st.session_state.filter_mode)

    if len(filtered_pairs) == 0:
        st.warning(f"No pairs match the current filter: {st.session_state.filter_mode}")
        st.info("Try changing the filter in the sidebar")
        render_filter_controls()
        return

    # Ensure current index is valid
    if st.session_state.current_index >= len(filtered_pairs):
        st.session_state.current_index = len(filtered_pairs) - 1

    # Sidebar controls
    render_statistics(pair_manager, filtered_pairs)
    render_filter_controls()
    render_navigation(pair_manager, filtered_pairs)
    render_overlay_controls()

    # Get current pair
    basename, rgb_path, nir_path = filtered_pairs[st.session_state.current_index]

    # Marking controls
    render_marking_controls(pair_manager, basename)
    render_batch_operations(pair_manager)

    # Load and process images
    try:
        with st.spinner("Loading images..."):
            rgb, nir = processor.load_pair(rgb_path, nir_path)
            overlay = processor.create_overlay(rgb, nir, st.session_state.overlay_mode)
            aerochrome = processor.generate_aerochrome(rgb, nir)

        # Display grid
        render_image_grid(rgb, nir, overlay, aerochrome, processor, basename)

        # Keyboard shortcuts info
        with st.expander("ℹ️ Keyboard Shortcuts"):
            st.markdown("""
            - **Left Arrow**: Previous image
            - **Right Arrow**: Next image
            - Use the sidebar controls for marking and filtering
            """)

    except Exception as e:
        st.error(f"Error loading pair {basename}: {e}")


if __name__ == "__main__":
    main()