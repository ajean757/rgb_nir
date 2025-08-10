import os
import shutil
import glob
import argparse
from pathlib import Path


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
        """
    )
    
    parser.add_argument(
        "directory", 
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
    
    return parser.parse_args()


def main():
    """Main function for command-line usage."""
    args = parse_arguments()
    
    verbose = not args.quiet
    
    # Check if no action flags are provided
    if not args.reorganize and not args.check_pairs:
        print("Error: You must specify an action:")
        print("  --reorganize    : Organize images into subdirectories")
        print("  --check-pairs   : Check for unpaired images")
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
