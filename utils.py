import os
import shutil
import glob
import argparse
from pathlib import Path
from tqdm import tqdm


def organize_image_directory(directory_path, originals_name="originals", processed_name="processed", 
                            rgb_jpg_name="rgb_jpg", nir_jpg_name="nir_jpg", 
                            rgb_dng_name="rgb_dng", nir_dng_name="nir_dng",
                            copy_files=False, verbose=True):
    """
    Organizes an image directory by moving images into subdirectories based on type and format.
    
    Args:
        directory_path (str): Path to the directory containing images to organize
        originals_name (str): Name of the originals subdirectory (default: "originals")
        processed_name (str): Name of the processed subdirectory (default: "processed")
        rgb_jpg_name (str): Name of RGB JPEG subdirectory (default: "rgb_jpg")
        nir_jpg_name (str): Name of NIR JPEG subdirectory (default: "nir_jpg")
        rgb_dng_name (str): Name of RGB DNG subdirectory (default: "rgb_dng")
        nir_dng_name (str): Name of NIR DNG subdirectory (default: "nir_dng")
        copy_files (bool): If True, copy files instead of moving them (default: False)
        verbose (bool): If True, print detailed output (default: True)
        
    The function creates the following structure:
    /directory_path/
        /{originals_name}/
            /{rgb_jpg_name}/    - Contains *_rgb.jpg files
            /{nir_jpg_name}/    - Contains *_ir.jpg files  
            /{rgb_dng_name}/    - Contains *_rgb.dng files
            /{nir_dng_name}/    - Contains *_ir.dng files
        /{processed_name}/      - Empty directory for processed images
    """
    
    # Convert to Path object for easier manipulation
    base_path = Path(directory_path)
    
    if not base_path.exists():
        raise ValueError(f"Directory {directory_path} does not exist")
    
    # Create main subdirectories using custom names
    originals_path = base_path / originals_name
    processed_path = base_path / processed_name
    
    # Create subdirectories for different image types using custom names
    subdirs = {
        "rgb_jpg": originals_path / rgb_jpg_name,
        "nir_jpg": originals_path / nir_jpg_name, 
        "rgb_dng": originals_path / rgb_dng_name,
        "nir_dng": originals_path / nir_dng_name
    }
    
    # Create all directories
    for subdir_path in subdirs.values():
        subdir_path.mkdir(parents=True, exist_ok=True)
    
    processed_path.mkdir(parents=True, exist_ok=True)
    
    # Define file patterns and their destination directories
    file_patterns = {
        "*_rgb.jpg": subdirs["rgb_jpg"],
        "*_ir.jpg": subdirs["nir_jpg"],
        "*_rgb.dng": subdirs["rgb_dng"], 
        "*_ir.dng": subdirs["nir_dng"]
    }
    
    moved_files = {key: 0 for key in file_patterns.keys()}
    
    # Move files to appropriate subdirectories
    for pattern, destination in file_patterns.items():
        # Find all files matching the pattern in the base directory
        files = list(base_path.glob(pattern))
        
        for file_path in files:
            # Skip if file is already in a subdirectory
            if file_path.parent != base_path:
                continue
                
            destination_file = destination / file_path.name
            
            try:
                # Copy or move file to destination
                if copy_files:
                    shutil.copy2(str(file_path), str(destination_file))
                    action = "Copied"
                else:
                    shutil.move(str(file_path), str(destination_file))
                    action = "Moved"
                    
                moved_files[pattern] += 1
                if verbose:
                    print(f"{action} {file_path.name} to {destination.name}/")
            except Exception as e:
                if verbose:
                    print(f"Error {'copying' if copy_files else 'moving'} {file_path.name}: {e}")
    
    # Print summary
    if verbose:
        print(f"\nOrganization complete!")
        print(f"Directory structure created in: {directory_path}")
        for pattern, count in moved_files.items():
            action = "copied" if copy_files else "moved"
            print(f"  {pattern}: {count} files {action}")
    
    return True


def create_processed_subdirectories(directory_path, processed_name="processed",
                                  rgb_jpg_name="rgb_jpg", nir_jpg_name="nir_jpg",
                                  rgb_dng_name="rgb_dng", nir_dng_name="nir_dng",
                                  verbose=True):
    """
    Creates processed subdirectories for each image type.
    
    Args:
        directory_path (str): Path to the base directory
        processed_name (str): Name of the processed subdirectory (default: "processed")
        rgb_jpg_name (str): Name of RGB JPEG subdirectory (default: "rgb_jpg")
        nir_jpg_name (str): Name of NIR JPEG subdirectory (default: "nir_jpg")
        rgb_dng_name (str): Name of RGB DNG subdirectory (default: "rgb_dng")
        nir_dng_name (str): Name of NIR DNG subdirectory (default: "nir_dng")
        verbose (bool): If True, print output (default: True)
    """
    base_path = Path(directory_path)
    processed_path = base_path / processed_name
    
    # Create processed subdirectories using custom names
    processed_subdirs = [
        rgb_jpg_name,
        nir_jpg_name, 
        rgb_dng_name,
        nir_dng_name
    ]
    
    for subdir in processed_subdirs:
        (processed_path / subdir).mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Created processed subdirectories in {processed_path}")


def find_unpaired_images(directory_path, originals_name="originals", 
                        rgb_jpg_name="rgb_jpg", nir_jpg_name="nir_jpg",
                        rgb_dng_name="rgb_dng", nir_dng_name="nir_dng",
                        verbose=True):
    """
    Identifies images that do not have corresponding pairs in the organized directory structure.
    
    Args:
        directory_path (str): Path to the base directory
        originals_name (str): Name of the originals subdirectory (default: "originals")
        rgb_jpg_name (str): Name of RGB JPEG subdirectory (default: "rgb_jpg")
        nir_jpg_name (str): Name of NIR JPEG subdirectory (default: "nir_jpg")
        rgb_dng_name (str): Name of RGB DNG subdirectory (default: "rgb_dng")
        nir_dng_name (str): Name of NIR DNG subdirectory (default: "nir_dng")
        verbose (bool): If True, print detailed report (default: True)
    
    Returns:
        dict: Dictionary containing unpaired images by category
    """
    base_path = Path(directory_path)
    originals_path = base_path / originals_name
    
    if not originals_path.exists():
        raise ValueError(f"Originals directory {originals_path} does not exist")
    
    # Define subdirectory paths
    subdirs = {
        "rgb_jpg": originals_path / rgb_jpg_name,
        "nir_jpg": originals_path / nir_jpg_name,
        "rgb_dng": originals_path / rgb_dng_name,
        "nir_dng": originals_path / nir_dng_name
    }
    
    def extract_timestamp(filename, suffix):
        """Extract timestamp from filename by removing the suffix."""
        if filename.endswith(suffix):
            return filename[:-len(suffix)]
        return None
    
    def get_timestamps_from_dir(dir_path, suffix):
        """Get set of timestamps from files in directory."""
        if not dir_path.exists():
            return set()
        
        timestamps = set()
        for file_path in dir_path.glob(f"*{suffix}"):
            timestamp = extract_timestamp(file_path.name, suffix)
            if timestamp:
                timestamps.add(timestamp)
        return timestamps
    
    # Get timestamps for each image type
    rgb_jpg_timestamps = get_timestamps_from_dir(subdirs["rgb_jpg"], "_rgb.jpg")
    nir_jpg_timestamps = get_timestamps_from_dir(subdirs["nir_jpg"], "_ir.jpg")
    rgb_dng_timestamps = get_timestamps_from_dir(subdirs["rgb_dng"], "_rgb.dng")
    nir_dng_timestamps = get_timestamps_from_dir(subdirs["nir_dng"], "_ir.dng")
    
    # Find unpaired images
    unpaired = {
        "jpg": {
            "rgb_without_nir": rgb_jpg_timestamps - nir_jpg_timestamps,
            "nir_without_rgb": nir_jpg_timestamps - rgb_jpg_timestamps
        },
        "dng": {
            "rgb_without_nir": rgb_dng_timestamps - nir_dng_timestamps,
            "nir_without_rgb": nir_dng_timestamps - rgb_dng_timestamps
        }
    }
    
    # Calculate totals
    totals = {
        "jpg": {
            "rgb_total": len(rgb_jpg_timestamps),
            "nir_total": len(nir_jpg_timestamps),
            "paired": len(rgb_jpg_timestamps & nir_jpg_timestamps)
        },
        "dng": {
            "rgb_total": len(rgb_dng_timestamps),
            "nir_total": len(nir_dng_timestamps),
            "paired": len(rgb_dng_timestamps & nir_dng_timestamps)
        }
    }
    
    if verbose:
        print(f"\nUnpaired Images Report for: {directory_path}")
        print("=" * 60)
        
        for format_type in ["jpg", "dng"]:
            format_upper = format_type.upper()
            print(f"\n{format_upper} Files:")
            print(f"  RGB files: {totals[format_type]['rgb_total']}")
            print(f"  NIR files: {totals[format_type]['nir_total']}")
            print(f"  Paired files: {totals[format_type]['paired']}")
            
            rgb_unpaired = unpaired[format_type]["rgb_without_nir"]
            nir_unpaired = unpaired[format_type]["nir_without_rgb"]
            
            print(f"  RGB files without NIR pair: {len(rgb_unpaired)}")
            print(f"  NIR files without RGB pair: {len(nir_unpaired)}")
            
            if rgb_unpaired:
                print(f"    RGB-only timestamps: {sorted(list(rgb_unpaired))[:10]}")
                if len(rgb_unpaired) > 10:
                    print(f"    ... and {len(rgb_unpaired) - 10} more")
            
            if nir_unpaired:
                print(f"    NIR-only timestamps: {sorted(list(nir_unpaired))[:10]}")
                if len(nir_unpaired) > 10:
                    print(f"    ... and {len(nir_unpaired) - 10} more")
    
    return {
        "unpaired": unpaired,
        "totals": totals,
        "summary": {
            "jpg_rgb_unpaired": len(unpaired["jpg"]["rgb_without_nir"]),
            "jpg_nir_unpaired": len(unpaired["jpg"]["nir_without_rgb"]),
            "dng_rgb_unpaired": len(unpaired["dng"]["rgb_without_nir"]),
            "dng_nir_unpaired": len(unpaired["dng"]["nir_without_rgb"])
        }
    }


def merge_processed_directories(source_dirs, output_dir, 
                              rgb_subdir_name="rgb_png", nir_subdir_name="nir_png",
                              conflict_resolution="rename", copy_files=True,
                              validate_pairs=True, dry_run=False, verbose=True):
    """
    Merge multiple processed image directories into a single directory.
    
    Args:
        source_dirs (list): List of source directory paths to merge
        output_dir (str): Target merged directory path
        rgb_subdir_name (str): Name of RGB subdirectory (default: "rgb_png")
        nir_subdir_name (str): Name of NIR subdirectory (default: "nir_png")
        conflict_resolution (str): How to handle filename conflicts - "rename" or "skip" (default: "rename")
        copy_files (bool): If True, copy files instead of moving them (default: True)
        validate_pairs (bool): If True, ensure RGB/NIR pairs are maintained (default: True)
        dry_run (bool): If True, show what would be done without making changes (default: False)
        verbose (bool): If True, print detailed output (default: True)
        
    Returns:
        dict: Dictionary containing merge results and statistics
    """
    
    if not source_dirs:
        raise ValueError("No source directories provided")
    
    # Convert paths to Path objects
    source_paths = [Path(d) for d in source_dirs]
    output_path = Path(output_dir)
    
    # Validate source directories exist
    for i, source_path in enumerate(source_paths):
        if not source_path.exists():
            raise ValueError(f"Source directory {source_path} does not exist")
    
    # Create output directory structure (unless dry run)
    output_rgb_dir = output_path / rgb_subdir_name
    output_nir_dir = output_path / nir_subdir_name
    
    if not dry_run:
        output_rgb_dir.mkdir(parents=True, exist_ok=True)
        output_nir_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"{'DRY RUN: ' if dry_run else ''}Merging {len(source_dirs)} directories into {output_path}")
        print(f"RGB subdirectory: {rgb_subdir_name}")
        print(f"NIR subdirectory: {nir_subdir_name}")
        print(f"Conflict resolution: {conflict_resolution}")
        print(f"File operation: {'copy' if copy_files else 'move'}")
        print()
    
    # Step 1: Validate directory structures and collect file information
    all_source_files = {"rgb": {}, "nir": {}}  # {filename: (source_path, full_path)}
    file_extensions = set()
    
    for source_path in source_paths:
        source_rgb_dir = source_path / rgb_subdir_name
        source_nir_dir = source_path / nir_subdir_name
        
        # Check if required subdirectories exist
        if not source_rgb_dir.exists():
            if verbose:
                print(f"Warning: {source_path} missing {rgb_subdir_name} subdirectory, skipping")
            continue
        if not source_nir_dir.exists():
            if verbose:
                print(f"Warning: {source_path} missing {nir_subdir_name} subdirectory, skipping")
            continue
        
        # Collect RGB files
        for file_path in source_rgb_dir.iterdir():
            if file_path.is_file():
                file_extensions.add(file_path.suffix.lower())
                filename = file_path.name
                if filename in all_source_files["rgb"]:
                    if verbose:
                        print(f"Warning: Duplicate RGB file {filename} found in {source_path}")
                all_source_files["rgb"][filename] = (source_path.name, file_path)
        
        # Collect NIR files
        for file_path in source_nir_dir.iterdir():
            if file_path.is_file():
                file_extensions.add(file_path.suffix.lower())
                filename = file_path.name
                if filename in all_source_files["nir"]:
                    if verbose:
                        print(f"Warning: Duplicate NIR file {filename} found in {source_path}")
                all_source_files["nir"][filename] = (source_path.name, file_path)
    
    # Step 2: File type sanity check
    if len(file_extensions) > 1:
        print(f"Error: Multiple file types detected: {sorted(file_extensions)}")
        print("Please merge directories with the same file type separately")
        return {"success": False, "error": "Mixed file types"}
    
    if not file_extensions:
        print("Warning: No image files found in any source directory")
        return {"success": True, "files_processed": 0, "conflicts": 0, "unpaired": 0}
    
    file_extension = list(file_extensions)[0]
    if verbose:
        print(f"File type detected: {file_extension}")
        print()
    
    # Step 3: Validate RGB/NIR pairing if requested
    unpaired_files = []
    if validate_pairs:
        def extract_timestamp_from_filename(filename, suffix):
            """Extract timestamp from filename by removing the suffix."""
            if filename.endswith(suffix):
                return filename[:-len(suffix)]
            return None
        
        # Extract timestamps
        rgb_timestamps = set()
        nir_timestamps = set()
        
        for filename in all_source_files["rgb"]:
            # Try common RGB suffixes
            for suffix in [f"_rgb{file_extension}", f"_color{file_extension}"]:
                timestamp = extract_timestamp_from_filename(filename, suffix)
                if timestamp:
                    rgb_timestamps.add(timestamp)
                    break
        
        for filename in all_source_files["nir"]:
            # Try common NIR suffixes  
            for suffix in [f"_nir{file_extension}", f"_ir{file_extension}", f"_infrared{file_extension}"]:
                timestamp = extract_timestamp_from_filename(filename, suffix)
                if timestamp:
                    nir_timestamps.add(timestamp)
                    break
        
        # Find unpaired files
        unpaired_rgb = rgb_timestamps - nir_timestamps
        unpaired_nir = nir_timestamps - rgb_timestamps
        
        if unpaired_rgb or unpaired_nir:
            if verbose:
                print(f"Warning: Found {len(unpaired_rgb)} unpaired RGB files and {len(unpaired_nir)} unpaired NIR files")
                if unpaired_rgb:
                    print(f"  Unpaired RGB timestamps: {sorted(list(unpaired_rgb))[:5]}")
                    if len(unpaired_rgb) > 5:
                        print(f"    ... and {len(unpaired_rgb) - 5} more")
                if unpaired_nir:
                    print(f"  Unpaired NIR timestamps: {sorted(list(unpaired_nir))[:5]}")
                    if len(unpaired_nir) > 5:
                        print(f"    ... and {len(unpaired_nir) - 5} more")
                print("  Unpaired files will be skipped during merge")
                print()
            
            # Remove unpaired files from processing
            files_to_remove = []
            for filename in all_source_files["rgb"]:
                for suffix in [f"_rgb{file_extension}", f"_color{file_extension}"]:
                    timestamp = extract_timestamp_from_filename(filename, suffix)
                    if timestamp and timestamp in unpaired_rgb:
                        files_to_remove.append(("rgb", filename))
                        unpaired_files.append(filename)
                        break
            
            for filename in all_source_files["nir"]:
                for suffix in [f"_nir{file_extension}", f"_ir{file_extension}", f"_infrared{file_extension}"]:
                    timestamp = extract_timestamp_from_filename(filename, suffix)
                    if timestamp and timestamp in unpaired_nir:
                        files_to_remove.append(("nir", filename))
                        unpaired_files.append(filename)
                        break
            
            # Remove unpaired files
            for file_type, filename in files_to_remove:
                del all_source_files[file_type][filename]
    
    # Step 4: Check for conflicts and prepare final file operations
    operations = []  # List of (src_path, dst_path, original_name, final_name, file_type)
    conflicts = 0
    
    for file_type in ["rgb", "nir"]:
        output_subdir = output_rgb_dir if file_type == "rgb" else output_nir_dir
        existing_files = set()
        
        if not dry_run and output_subdir.exists():
            existing_files = {f.name for f in output_subdir.iterdir() if f.is_file()}
        
        for filename, (source_name, file_path) in all_source_files[file_type].items():
            final_filename = filename
            
            # Handle conflicts
            if filename in existing_files:
                conflicts += 1
                if conflict_resolution == "rename":
                    # Add source directory prefix
                    name_part, ext_part = filename.rsplit('.', 1)
                    final_filename = f"{source_name}_{filename}"
                    if verbose:
                        print(f"Conflict: {filename} -> {final_filename}")
                elif conflict_resolution == "skip":
                    if verbose:
                        print(f"Skipping duplicate: {filename}")
                    continue
            
            dst_path = output_subdir / final_filename
            operations.append((file_path, dst_path, filename, final_filename, file_type))
            existing_files.add(final_filename)
    
    if verbose:
        print(f"Planned operations: {len(operations)} files to {'copy' if copy_files else 'move'}")
        if conflicts > 0:
            print(f"Filename conflicts handled: {conflicts}")
        print()
    
    # Step 5: Execute file operations (unless dry run)
    if not dry_run:
        if verbose and len(operations) > 10:
            print("Processing files...")
            
        # Use progress bar for large operations
        operation_iter = tqdm(operations, desc="Copying files" if copy_files else "Moving files") if len(operations) > 10 else operations
        
        successful_operations = 0
        failed_operations = []
        
        for src_path, dst_path, original_name, final_name, file_type in operation_iter:
            try:
                if copy_files:
                    shutil.copy2(str(src_path), str(dst_path))
                else:
                    shutil.move(str(src_path), str(dst_path))
                successful_operations += 1
                
                if verbose and len(operations) <= 10:
                    action = "Copied" if copy_files else "Moved"
                    print(f"{action} {file_type.upper()}: {original_name} -> {final_name}")
                    
            except Exception as e:
                failed_operations.append((original_name, str(e)))
                if verbose:
                    print(f"Error processing {original_name}: {e}")
        
        if failed_operations and verbose:
            print(f"\nFailed operations: {len(failed_operations)}")
            for filename, error in failed_operations[:5]:
                print(f"  {filename}: {error}")
            if len(failed_operations) > 5:
                print(f"  ... and {len(failed_operations) - 5} more")
    
    # Step 6: Summary
    results = {
        "success": True,
        "dry_run": dry_run,
        "files_processed": len(operations) if dry_run else successful_operations,
        "conflicts": conflicts,
        "unpaired_skipped": len(unpaired_files),
        "failed_operations": len(failed_operations) if not dry_run else 0,
        "file_extension": file_extension,
        "source_directories": len(source_dirs)
    }
    
    if verbose:
        print(f"\n{'DRY RUN ' if dry_run else ''}MERGE SUMMARY:")
        print(f"  Source directories: {results['source_directories']}")
        print(f"  Files {'planned' if dry_run else 'processed'}: {results['files_processed']}")
        print(f"  Conflicts handled: {results['conflicts']}")
        print(f"  Unpaired files skipped: {results['unpaired_skipped']}")
        if not dry_run and results['failed_operations'] > 0:
            print(f"  Failed operations: {results['failed_operations']}")
        print(f"  File type: {results['file_extension']}")
        if not dry_run:
            print(f"  Output directory: {output_path}")
    
    return results


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Organize image directories by type and format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils.py /path/to/images --reorganize
  python utils.py /path/to/images --reorganize --copy --verbose
  python utils.py /path/to/images --check-pairs
  python utils.py /path/to/images --check-pairs --quiet
  python utils.py /path/to/images --reorganize --originals-name raw_data --processed-name output
  python utils.py /path/to/images --reorganize --rgb-jpg-name color_jpeg --nir-jpg-name infrared_jpeg
  python utils.py --merge --sources dir1 dir2 dir3 --output merged_dir --dry-run
  python utils.py --merge --sources dir1 dir2 --output merged_dir --rgb-subdir rgb_png --nir-subdir nir_png
        """
    )
    
    parser.add_argument(
        "directory", 
        nargs="?",  # Make directory optional for merge operations
        help="Path to the directory containing images to organize"
    )
    
    parser.add_argument(
        "--originals-name", 
        default="originals",
        help="Name of the originals subdirectory (default: originals)"
    )
    
    parser.add_argument(
        "--processed-name", 
        default="processed",
        help="Name of the processed subdirectory (default: processed)"
    )
    
    parser.add_argument(
        "--rgb-jpg-name", 
        default="rgb_jpg",
        help="Name of RGB JPEG subdirectory (default: rgb_jpg)"
    )
    
    parser.add_argument(
        "--nir-jpg-name", 
        default="nir_jpg", 
        help="Name of NIR JPEG subdirectory (default: nir_jpg)"
    )
    
    parser.add_argument(
        "--rgb-dng-name", 
        default="rgb_dng",
        help="Name of RGB DNG subdirectory (default: rgb_dng)"
    )
    
    parser.add_argument(
        "--nir-dng-name", 
        default="nir_dng",
        help="Name of NIR DNG subdirectory (default: nir_dng)"
    )
    
    parser.add_argument(
        "--copy", 
        action="store_true",
        help="Copy files instead of moving them"
    )
    
    parser.add_argument(
        "--no-processed", 
        action="store_true",
        help="Don't create processed subdirectories"
    )
    
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress output messages"
    )
    
    parser.add_argument(
        "--reorganize", 
        action="store_true",
        help="Reorganize images into subdirectories (required to perform reorganization)"
    )
    
    parser.add_argument(
        "--check-pairs", 
        action="store_true",
        help="Check for unpaired images in organized directory structure"
    )
    
    # Merge-specific arguments
    parser.add_argument(
        "--merge", 
        action="store_true",
        help="Merge multiple processed directories into one"
    )
    
    parser.add_argument(
        "--sources", 
        nargs="+",
        help="Source directories to merge (required with --merge)"
    )
    
    parser.add_argument(
        "--output", 
        help="Output directory for merged files (required with --merge)"
    )
    
    parser.add_argument(
        "--rgb-subdir", 
        default="rgb_png",
        help="Name of RGB subdirectory in source/output (default: rgb_png)"
    )
    
    parser.add_argument(
        "--nir-subdir", 
        default="nir_png",
        help="Name of NIR subdirectory in source/output (default: nir_png)"
    )
    
    parser.add_argument(
        "--conflict-resolution", 
        choices=["rename", "skip"],
        default="rename",
        help="How to handle filename conflicts: rename or skip (default: rename)"
    )
    
    parser.add_argument(
        "--no-pair-validation", 
        action="store_true",
        help="Skip RGB/NIR pair validation during merge"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    return parser.parse_args()


def main():
    """Main function for command-line usage."""
    args = parse_arguments()
    
    verbose = not args.quiet
    
    # Check for merge operation
    if args.merge:
        if not args.sources or not args.output:
            print("Error: --merge requires --sources and --output arguments")
            print("Example: python utils.py --merge --sources dir1 dir2 --output merged_dir")
            return 1
        
        try:
            result = merge_processed_directories(
                source_dirs=args.sources,
                output_dir=args.output,
                rgb_subdir_name=args.rgb_subdir,
                nir_subdir_name=args.nir_subdir,
                conflict_resolution=args.conflict_resolution,
                copy_files=True,  # Always copy for merge operations
                validate_pairs=not args.no_pair_validation,
                dry_run=args.dry_run,
                verbose=verbose
            )
            
            if not result["success"]:
                return 1
            
            # Return specific exit codes
            if result["unpaired_skipped"] > 0:
                return 3  # Unpaired files were skipped
            elif result["failed_operations"] > 0:
                return 4  # Some operations failed
            
        except Exception as e:
            print(f"Error during merge: {e}")
            return 1
        
        return 0
    
    # For non-merge operations, directory argument is required
    if not args.directory:
        print("Error: directory argument is required for --reorganize and --check-pairs")
        return 1
    
    # Check if no action flags are provided
    if not args.reorganize and not args.check_pairs:
        print("Error: You must specify an action:")
        print("  --reorganize    : Organize images into subdirectories")
        print("  --check-pairs   : Check for unpaired images")
        print("  --merge         : Merge multiple processed directories")
        print("  Use --help for more information")
        return 1
    
    # Check if both flags are provided
    if args.reorganize and args.check_pairs:
        print("Error: Cannot use both --reorganize and --check-pairs at the same time")
        print("Please choose one action to perform")
        return 1
    
    try:
        if args.reorganize:
            # Organize the main directory
            organize_image_directory(
                directory_path=args.directory,
                originals_name=args.originals_name,
                processed_name=args.processed_name,
                rgb_jpg_name=args.rgb_jpg_name,
                nir_jpg_name=args.nir_jpg_name,
                rgb_dng_name=args.rgb_dng_name,
                nir_dng_name=args.nir_dng_name,
                copy_files=args.copy,
                verbose=verbose
            )
        
        elif args.check_pairs:
            # Check for unpaired images
            result = find_unpaired_images(
                directory_path=args.directory,
                originals_name=args.originals_name,
                rgb_jpg_name=args.rgb_jpg_name,
                nir_jpg_name=args.nir_jpg_name,
                rgb_dng_name=args.rgb_dng_name,
                nir_dng_name=args.nir_dng_name,
                verbose=verbose
            )
            
            # Return non-zero exit code if unpaired images found
            total_unpaired = sum(result["summary"].values())
            if total_unpaired > 0 and verbose:
                print(f"\nTotal unpaired images found: {total_unpaired}")
                return 2  # Different exit code to indicate unpaired images found
        
        # # Create processed subdirectories unless explicitly disabled
        # if not args.no_processed:
        #     create_processed_subdirectories(
        #         directory_path=args.directory,
        #         processed_name=args.processed_name,
        #         rgb_jpg_name=args.rgb_jpg_name,
        #         nir_jpg_name=args.nir_jpg_name,
        #         rgb_dng_name=args.rgb_dng_name,
        #         nir_dng_name=args.nir_dng_name,
        #         verbose=verbose
        #     )
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


# Example usage:
if __name__ == "__main__":
    exit(main())
