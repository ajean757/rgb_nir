#!/usr/bin/env python3
"""
Multi-directory image comparison viewer.
Displays images with matching filenames from multiple directories side-by-side.

Usage:
    streamlit run compare_dirs.py -- --dirs dir1 dir2 dir3 [--extensions jpg,png,jpeg]
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import argparse
from collections import defaultdict
from typing import List, Dict
import sys


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare images across multiple directories")
    parser.add_argument('--dirs', nargs='+', required=True, help='List of directories to compare')
    parser.add_argument('--extensions', default='jpg,png,jpeg,bmp,tiff',
                       help='Comma-separated list of image extensions (default: jpg,png,jpeg,bmp,tiff)')
    return parser.parse_args()


def find_matching_images(directories: List[Path], extensions: List[str]) -> Dict[str, Dict[str, Path]]:
    """
    Find images with matching filenames (stem) across directories.

    Returns:
        Dict mapping filename stem to dict of {directory_name: file_path}
    """
    images_by_stem = defaultdict(dict)

    for dir_path in directories:
        if not dir_path.exists():
            st.warning(f"Directory not found: {dir_path}")
            continue

        dir_name = dir_path.name

        for ext in extensions:
            for img_file in dir_path.glob(f"*.{ext}"):
                stem = img_file.stem
                images_by_stem[stem][dir_name] = img_file

    # Filter to only include stems present in at least one directory
    matching_images = {stem: dirs for stem, dirs in images_by_stem.items() if dirs}

    return matching_images


def main():
    st.set_page_config(
        page_title="Multi-Directory Image Viewer",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Parse arguments
    try:
        args = parse_args()
    except SystemExit:
        st.error("Please run with: streamlit run compare_dirs.py -- --dirs dir1 dir2 ...")
        st.info("Example: streamlit run compare_dirs.py -- --dirs /path/to/dir1 /path/to/dir2")
        return

    # Convert to Path objects
    directories = [Path(d).resolve() for d in args.dirs]
    extensions = [ext.strip().lower().lstrip('.') for ext in args.extensions.split(',')]

    # Title
    st.title("📸 Multi-Directory Image Viewer")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    st.sidebar.write("**Directories:**")
    for i, dir_path in enumerate(directories, 1):
        exists = "✅" if dir_path.exists() else "❌"
        st.sidebar.text(f"{exists} {i}. {dir_path.name}")

    # Find matching images
    with st.spinner("Scanning directories..."):
        matching_images = find_matching_images(directories, extensions)

    if not matching_images:
        st.error("No images found in the specified directories!")
        return

    # Sort stems for consistent ordering
    sorted_stems = sorted(matching_images.keys())

    # Sidebar stats
    st.sidebar.metric("Total unique images", len(sorted_stems))
    st.sidebar.metric("Directories", len(directories))

    # Navigation
    st.sidebar.header("Navigation")

    # Initialize session state
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # Navigation controls
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])

    if col1.button("⬅️ Prev"):
        st.session_state.current_index = max(0, st.session_state.current_index - 1)

    if col3.button("Next ➡️"):
        st.session_state.current_index = min(len(sorted_stems) - 1, st.session_state.current_index + 1)

    # Slider for quick navigation
    st.session_state.current_index = st.sidebar.slider(
        "Image index",
        0,
        len(sorted_stems) - 1,
        st.session_state.current_index
    )

    # Jump to specific image
    st.sidebar.write("**Jump to image:**")
    selected_stem = st.sidebar.selectbox(
        "Select by filename",
        sorted_stems,
        index=st.session_state.current_index,
        label_visibility="collapsed"
    )
    st.session_state.current_index = sorted_stems.index(selected_stem)

    # Display options
    st.sidebar.header("Display Options")
    image_width = st.sidebar.slider("Image width (pixels)", 200, 1000, 400)
    show_filename = st.sidebar.checkbox("Show filenames", value=True)
    show_missing = st.sidebar.checkbox("Show missing images", value=True)

    # Current image
    current_stem = sorted_stems[st.session_state.current_index]
    current_images = matching_images[current_stem]

    # Main display
    st.header(f"Image: {current_stem}")
    st.caption(f"Showing {st.session_state.current_index + 1} of {len(sorted_stems)}")

    # Create columns for each directory
    cols = st.columns(len(directories))

    for col, dir_path in zip(cols, directories):
        dir_name = dir_path.name

        with col:
            st.subheader(dir_name)

            if dir_name in current_images:
                img_path = current_images[dir_name]
                try:
                    img = Image.open(img_path)
                    st.image(img, width=image_width)

                    if show_filename:
                        st.caption(f"📄 {img_path.name}")
                        st.caption(f"📐 {img.size[0]}×{img.size[1]}")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            else:
                if show_missing:
                    st.warning("⚠️ Not found in this directory")

    # Statistics in sidebar
    st.sidebar.header("Current Image Stats")
    st.sidebar.write(f"**Present in {len(current_images)}/{len(directories)} directories**")
    for dir_name in [d.name for d in directories]:
        icon = "✅" if dir_name in current_images else "❌"
        st.sidebar.text(f"{icon} {dir_name}")

    # Keyboard shortcuts hint
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **💡 Tips:**
    - Use slider for quick navigation
    - Select from dropdown to jump to specific image
    - Adjust image width for better comparison
    - Toggle missing images display
    """)


if __name__ == "__main__":
    main()
