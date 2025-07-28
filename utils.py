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


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Organize image directories by type and format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python utils.py /path/to/images
  python utils.py /path/to/images --copy --verbose
  python utils.py /path/to/images --originals-name raw_data --processed-name output
  python utils.py /path/to/images --rgb-jpg-name color_jpeg --nir-jpg-name infrared_jpeg
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
    
    return parser.parse_args()


def main():
    """Main function for command-line usage."""
    args = parse_arguments()
    
    verbose = not args.quiet
    
    try:
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
        
        # Create processed subdirectories unless explicitly disabled
        if not args.no_processed:
            create_processed_subdirectories(
                directory_path=args.directory,
                processed_name=args.processed_name,
                rgb_jpg_name=args.rgb_jpg_name,
                nir_jpg_name=args.nir_jpg_name,
                rgb_dng_name=args.rgb_dng_name,
                nir_dng_name=args.nir_dng_name,
                verbose=verbose
            )
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


# Example usage:
if __name__ == "__main__":
    exit(main())
