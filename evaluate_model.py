"""
Model Evaluation Script for RGB to NIR Translation
=================================================

Standalone evaluation script for trained RGB-to-NIR models.
Loads checkpoints and evaluates on test data without training.

Usage:
-----
# Evaluate best model on test set
python evaluate_model.py --data_root /path/to/data --checkpoint training_output/best_model.pt

# Evaluate specific checkpoint
python evaluate_model.py --data_root /path/to/data --checkpoint training_output/checkpoint_epoch_050.pt

# Custom test split
python evaluate_model.py --data_root /path/to/data --checkpoint best_model.pt --test_size 0.2

# Generate sample outputs
python evaluate_model.py --data_root /path/to/data --checkpoint best_model.pt --save_samples --num_samples 10
"""

import argparse
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

# Import the model architectures and utilities
from rgb2nir_minimal_unet import (
    GeneratorUNet, PairedRGBNIR, ssim, gradient_loss,
    validate_data_structure, auto_detect_subdirectories
)


def evaluate_model(model, test_loader, device, lambda_l1=100.0, lambda_ssim=10.0, lambda_grad=10.0):
    """
    Evaluate model performance on the test set.
    
    Args:
        model: Generator model to evaluate
        test_loader: DataLoader for test dataset
        device: Device to run evaluation on
        lambda_l1, lambda_ssim, lambda_grad: Loss weights
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    model.eval()
    metrics = {
        'test_loss': 0.0,
        'test_l1': 0.0,
        'test_ssim': 0.0,
        'test_grad': 0.0,
    }
    
    batch_metrics = []
    
    with torch.no_grad():
        for rgb, nir in tqdm(test_loader, desc="Evaluating"):
            rgb, nir = rgb.to(device), nir.to(device)
            fake = model(rgb)
            
            # Calculate metrics
            l1_loss = F.l1_loss(fake, nir)
            ssim_score = ssim(fake, nir)
            grad_loss_val = gradient_loss(fake, nir)
            
            # Composite loss
            total_loss = (lambda_l1 * l1_loss + 
                        lambda_ssim * (1 - ssim_score) + 
                        lambda_grad * grad_loss_val)
            
            # Store batch metrics for std calculation
            batch_metrics.append({
                'loss': total_loss.item(),
                'l1': l1_loss.item(),
                'ssim': ssim_score.item(),
                'grad': grad_loss_val.item()
            })
            
            # Accumulate metrics
            metrics['test_loss'] += total_loss.item()
            metrics['test_l1'] += l1_loss.item()
            metrics['test_ssim'] += ssim_score.item()
            metrics['test_grad'] += grad_loss_val.item()
    
    # Average metrics
    for k in metrics:
        metrics[k] /= len(test_loader)
    
    # Calculate standard deviations
    metrics['test_loss_std'] = np.std([b['loss'] for b in batch_metrics])
    metrics['test_l1_std'] = np.std([b['l1'] for b in batch_metrics])
    metrics['test_ssim_std'] = np.std([b['ssim'] for b in batch_metrics])
    metrics['test_grad_std'] = np.std([b['grad'] for b in batch_metrics])
    
    return metrics


def save_sample_outputs(model, test_loader, device, output_dir, num_samples=10):
    """
    Generate and save sample outputs from the test set.
    
    Args:
        model: Generator model
        test_loader: DataLoader for test dataset
        device: Device to run on
        output_dir: Directory to save samples
        num_samples: Number of samples to generate
    """
    model.eval()
    sample_dir = output_dir / 'evaluation_samples'
    sample_dir.mkdir(exist_ok=True)
    
    samples_saved = 0
    
    with torch.no_grad():
        for batch_idx, (rgb, nir) in enumerate(test_loader):
            if samples_saved >= num_samples:
                break
                
            rgb, nir = rgb.to(device), nir.to(device)
            fake = model(rgb)
            
            # Save individual samples from batch
            batch_size = rgb.size(0)
            for i in range(min(batch_size, num_samples - samples_saved)):
                sample_idx = samples_saved + i
                
                # Convert from [-1,1] to [0,1] for saving
                rgb_img = (rgb[i] + 1) / 2
                nir_real = (nir[i] + 1) / 2
                nir_fake = (fake[i] + 1) / 2
                
                # Create comparison image: RGB | Real NIR | Fake NIR
                comparison = torch.cat([rgb_img, nir_real.repeat(3,1,1), nir_fake.repeat(3,1,1)], dim=2)
                
                save_image(comparison, sample_dir / f'sample_{sample_idx:03d}_comparison.png')
                save_image(rgb_img, sample_dir / f'sample_{sample_idx:03d}_rgb.png')
                save_image(nir_real, sample_dir / f'sample_{sample_idx:03d}_nir_real.png')
                save_image(nir_fake, sample_dir / f'sample_{sample_idx:03d}_nir_fake.png')
            
            samples_saved += min(batch_size, num_samples - samples_saved)
    
    print(f"Saved {samples_saved} sample outputs to {sample_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained RGB-to-NIR translation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
========
# Evaluate best model
python evaluate_model.py --data_root DATA/data_04_06_2025 --checkpoint training_output/best_model.pt

# Evaluate with custom test split
python evaluate_model.py --data_root DATA/data_04_06_2025 --checkpoint best_model.pt --test_size 0.2

# Generate sample outputs
python evaluate_model.py --data_root DATA/data_04_06_2025 --checkpoint best_model.pt --save_samples --num_samples 20

# Evaluate multiple checkpoints
python evaluate_model.py --data_root DATA/data_04_06_2025 --checkpoint training_output/checkpoint_epoch_*.pt
        """)
    
    # Data arguments
    parser.add_argument('--data_root', required=True,
                       help='Path to data directory')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--rgb_subdir', default=None,
                       help='RGB subdirectory name (auto-detected if not specified)')
    parser.add_argument('--nir_subdir', default=None,
                       help='NIR subdirectory name (auto-detected if not specified)')
    
    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--test_size', type=float, default=0.1,
                       help='Percentage of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Percentage of data to use for validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='evaluation_output',
                       help='Directory for evaluation outputs')
    parser.add_argument('--save_samples', action='store_true',
                       help='Save sample output images')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of sample images to save')
    
    args = parser.parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("RGB-to-NIR Model Evaluation")
    print("="*60)
    
    # Validate data structure
    try:
        validation_result = validate_data_structure(
            Path(args.data_root),
            rgb_subdir=args.rgb_subdir,
            nir_subdir=args.nir_subdir,
            verbose=True
        )
        
        if not validation_result['valid']:
            print("\nERROR: Data validation failed!")
            print("Issues found:")
            for issue in validation_result['issues']:
                print(f"  - {issue}")
            return 1
            
    except Exception as e:
        print(f"ERROR setting up data: {e}")
        return 1
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint file not found: {checkpoint_path}")
        return 1
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Extract training configuration
        if 'args' in checkpoint:
            train_args = checkpoint['args']
            size = train_args.size
            base_channels = train_args.base_channels
            lambda_l1 = train_args.lambda_l1
            lambda_ssim = train_args.lambda_ssim
            lambda_grad = train_args.lambda_grad
            print(f"Model config: size={size}, base_channels={base_channels}")
            print(f"Loss weights: L1={lambda_l1}, SSIM={lambda_ssim}, Grad={lambda_grad}")
        else:
            # Default values if training args not available
            size = 512
            base_channels = 64
            lambda_l1 = 100.0
            lambda_ssim = 10.0
            lambda_grad = 10.0
            print("Warning: Using default model configuration (training args not found in checkpoint)")
            
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return 1
    
    # Create dataset
    print(f"\nSetting up dataset:")
    print(f"  Structure: {validation_result['structure_type']}")
    print(f"  Total pairs: {validation_result['pair_count']}")
    
    full_dataset = PairedRGBNIR(
        args.data_root,
        size=size,
        rgb_subdir=validation_result.get('rgb_subdir'),
        nir_subdir=validation_result.get('nir_subdir')
    )
    
    # Set random seed for reproducible splits
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Calculate sizes for train/val/test splits (same as training)
    dataset_size = len(full_dataset)
    test_size = int(dataset_size * args.test_size)
    val_size = int(dataset_size * args.val_size)
    train_size = dataset_size - test_size - val_size
    
    # Split dataset (we only need test set, but keep same split for consistency)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"  Test set: {test_size} pairs ({test_size/dataset_size:.1%})")
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    # Initialize model
    print(f"\nInitializing model (base_channels={base_channels})...")
    model = GeneratorUNet(base=base_channels).to(device)
    
    # Load model state
    model.load_state_dict(checkpoint['generator_state_dict'])
    print(f"Model loaded successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run evaluation
    print(f"\nEvaluating on {len(test_loader)} batches...")
    test_metrics = evaluate_model(
        model, test_loader, device,
        lambda_l1=lambda_l1,
        lambda_ssim=lambda_ssim,
        lambda_grad=lambda_grad
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Set Performance:")
    print(f"  Total Loss:  {test_metrics['test_loss']:.6f} ± {test_metrics['test_loss_std']:.6f}")
    print(f"  L1 Loss:     {test_metrics['test_l1']:.6f} ± {test_metrics['test_l1_std']:.6f}")
    print(f"  SSIM:        {test_metrics['test_ssim']:.6f} ± {test_metrics['test_ssim_std']:.6f}")
    print(f"  Grad Loss:   {test_metrics['test_grad']:.6f} ± {test_metrics['test_grad_std']:.6f}")
    
    # Save detailed results
    results_file = output_dir / f'evaluation_results_{checkpoint_path.stem}.txt'
    with open(results_file, 'w') as f:
        f.write("RGB-to-NIR Model Evaluation Results\n")
        f.write("="*40 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Data: {args.data_root}\n")
        f.write(f"Test samples: {test_size}\n")
        f.write(f"Model epoch: {checkpoint['epoch']}\n\n")
        
        f.write("Metrics (mean ± std):\n")
        for metric in ['test_loss', 'test_l1', 'test_ssim', 'test_grad']:
            mean_val = test_metrics[metric]
            std_val = test_metrics[f'{metric}_std']
            f.write(f"{metric}: {mean_val:.6f} ± {std_val:.6f}\n")
        
        f.write(f"\nLoss weights used:\n")
        f.write(f"lambda_l1: {lambda_l1}\n")
        f.write(f"lambda_ssim: {lambda_ssim}\n")
        f.write(f"lambda_grad: {lambda_grad}\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Generate sample outputs if requested
    if args.save_samples:
        print(f"\nGenerating {args.num_samples} sample outputs...")
        save_sample_outputs(model, test_loader, device, output_dir, args.num_samples)
    
    print("\nEvaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
