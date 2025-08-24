"""
Comprehensive evaluation script for CAS-GAN models
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import datetime
from models import create_model
from data import create_dataset
from util.visualizer import save_images
from util.util import tensor2im
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips
import csv
import json
from tqdm import tqdm

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate CAS-GAN models")
    
    # Basic options
    parser.add_argument('--dataroot', required=True, help='path to images')
    parser.add_argument('--name', required=True, help='name of the experiment')
    parser.add_argument('--model', type=str, required=True, help='model type')
    parser.add_argument('--netG', type=str, default='integrated', help='generator architecture')
    parser.add_argument('--netD', type=str, default='frequency_aware', help='discriminator architecture')
    parser.add_argument('--dataset_mode', type=str, default='aligned', help='dataset mode')
    parser.add_argument('--direction', type=str, default='AtoB', help='translation direction')
    parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop to this size')
    parser.add_argument('--input_nc', type=int, default=1, help='number of input channels')
    parser.add_argument('--output_nc', type=int, default=1, help='number of output channels')
    parser.add_argument('--num_bands', type=int, default=3, help='number of frequency bands')
    parser.add_argument('--num_regions', type=int, default=4, help='number of intensity regions')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
    parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load')
    
    # Evaluation options
    parser.add_argument('--eval_batch_size', type=int, default=4, help='batch size for evaluation')
    parser.add_argument('--num_test', type=int, default=float("inf"), help='how many test images to run')
    parser.add_argument('--save_images', action='store_true', help='save test images')
    parser.add_argument('--results_dir', type=str, default='./results/', help='directory to save results')
    parser.add_argument('--compare_models', action='store_true', help='compare multiple models')
    parser.add_argument('--model_list', type=str, default='', help='comma-separated list of model names to compare')
    parser.add_argument('--compute_fid', action='store_true', help='compute FID score')
    
    return parser.parse_args()

def prepare_model(opt):
    """Create and prepare model for evaluation"""
    # Set isTrain attribute
    opt.isTrain = False  # Add this line
    
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    return model

def compute_metrics(real_tensor, fake_tensor, lpips_model=None):
    """Compute evaluation metrics for a batch of images"""
    # Convert tensors to numpy arrays
    real_np = tensor2im(real_tensor)
    fake_np = tensor2im(fake_tensor)
    
    # Initialize metrics
    batch_size = real_tensor.size(0)
    metrics = {
        'lpips': [],
        'mae': [],
        'ssim': [],
        'psnr': []
    }
    
    # Compute metrics for each image in batch
    for i in range(batch_size):
        # Extract single images
        real_img = real_np[i]
        fake_img = fake_np[i]
        
        # LPIPS (perceptual similarity)
        if lpips_model is not None:
            # Prepare tensors for LPIPS
            real_lpips = real_tensor[i:i+1].clone()
            fake_lpips = fake_tensor[i:i+1].clone()
            # Make 3-channel if needed
            if real_lpips.size(1) == 1:
                real_lpips = real_lpips.repeat(1, 3, 1, 1)
                fake_lpips = fake_lpips.repeat(1, 3, 1, 1)
            # Compute LPIPS
            with torch.no_grad():
                lpips_value = lpips_model(real_lpips, fake_lpips).item()
            metrics['lpips'].append(lpips_value)
        
        # MAE (Mean Absolute Error)
        mae = np.abs(real_img.astype(float) - fake_img.astype(float)).mean()
        metrics['mae'].append(mae)
        
        # SSIM (Structural Similarity Index)
        ssim = structural_similarity(real_img, fake_img, data_range=255, multichannel=(real_img.ndim > 2))
        metrics['ssim'].append(ssim)
        
        # PSNR (Peak Signal-to-Noise Ratio)
        psnr = peak_signal_noise_ratio(real_img, fake_img, data_range=255)
        metrics['psnr'].append(psnr)
    
    return metrics

def compute_stats(metrics_list):
    """Compute statistics for a list of metrics"""
    # Convert list of dicts to dict of lists
    metrics_dict = {}
    for key in metrics_list[0].keys():
        metrics_dict[key] = []
        for metrics in metrics_list:
            metrics_dict[key].extend(metrics[key])
    
    # Compute mean and std for each metric
    stats = {}
    for key, values in metrics_dict.items():
        values_array = np.array(values)
        stats[key] = {
            'mean': values_array.mean(),
            'std': values_array.std(),
            'min': values_array.min(),
            'max': values_array.max()
        }
    
    return stats

def save_results(results_dir, model_name, stats, args=None):
    """Save evaluation results"""
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Save statistics as JSON
    stats_file = os.path.join(results_dir, f"{model_name}_stats_{timestamp}.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save arguments
    if args is not None:
        args_file = os.path.join(results_dir, f"{model_name}_args_{timestamp}.json")
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Create summary CSV with mean and std
    csv_file = os.path.join(results_dir, f"{model_name}_summary_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max'])
        # Data
        for metric, values in stats.items():
            writer.writerow([metric, values['mean'], values['std'], values['min'], values['max']])
    
    # Print summary
    print(f"\nResults for {model_name}:")
    print("-" * 50)
    for metric, values in stats.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f} (min: {values['min']:.4f}, max: {values['max']:.4f})")
    
    return csv_file

def create_comparison_plot(results_dir, model_results):
    """Create comparison plot for multiple models"""
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Plot metrics comparison
    metrics = list(next(iter(model_results.values()))['stats'].keys())
    num_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5))
    
    for i, metric in enumerate(metrics):
        # Extract mean and std for each model
        model_names = []
        means = []
        stds = []
        
        for model_name, results in model_results.items():
            model_names.append(model_name)
            means.append(results['stats'][metric]['mean'])
            stds.append(results['stats'][metric]['std'])
        
        # Create bar plot
        ax = axes[i]
        x = np.arange(len(model_names))
        ax.bar(x, means, yerr=stds, alpha=0.7, capsize=10)
        ax.set_title(f"{metric.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Set y-axis limits based on metric
        if metric.lower() == 'lpips' or metric.lower() == 'mae':
            # Lower is better
            ax.set_ylim(0, max(means) * 1.2)
            # Add "lower is better" text
            ax.text(0.5, 0.95, "Lower is better", transform=ax.transAxes, 
                    ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            # Higher is better
            ax.set_ylim(min(min(means) * 0.9, 0.8), min(max(means) * 1.1, 1.0) if metric.lower() == 'ssim' else max(means) * 1.1)
            # Add "higher is better" text
            ax.text(0.5, 0.05, "Higher is better", transform=ax.transAxes, 
                    ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(results_dir, f"model_comparison_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def evaluate_model(opt, model_name=None):
    """Evaluate a single model"""
    # Update model name if provided
    if model_name is not None:
        opt.name = model_name
    
    # Prepare model
    model = prepare_model(opt)
    
    # Create dataset
    opt.phase = 'test'
    opt.batch_size = opt.eval_batch_size
    opt.serial_batches = True  # Disable shuffling for reproducibility
    dataset = create_dataset(opt)
    print(f"# Test images = {len(dataset)}")
    
    # Initialize LPIPS model
    try:
        lpips_model = lpips.LPIPS(net='alex').to(next(model.parameters()).device)
    except:
        print("Warning: Could not initialize LPIPS model. LPIPS will not be computed.")
        lpips_model = None
    
    # Create results directory
    results_dir = os.path.join(opt.results_dir, opt.name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate on dataset
    all_metrics = []
    
    for i, data in tqdm(enumerate(dataset), total=min(len(dataset), opt.num_test)):
        if i >= opt.num_test:
            break
        
        # Forward pass
        model.set_input(data)
        model.test()
        
        # Compute metrics
        visuals = model.get_current_visuals()
        metrics = compute_metrics(visuals['real_B'], visuals['fake_B'], lpips_model)
        all_metrics.append(metrics)
        
        # Save images if requested
        if opt.save_images:
            img_dir = os.path.join(results_dir, 'images')
            os.makedirs(img_dir, exist_ok=True)
            save_images(img_dir, visuals, i)
    
    # Compute statistics
    stats = compute_stats(all_metrics)
    
    # Save results
    results_file = save_results(results_dir, opt.name, stats, opt)
    
    return {
        'name': opt.name,
        'stats': stats,
        'results_file': results_file
    }

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Convert gpu_ids to list
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    
    # Compare multiple models if requested
    if args.compare_models:
        if not args.model_list:
            print("Error: --model_list is required when --compare_models is enabled.")
            return
        
        # Parse model list
        model_names = args.model_list.split(',')
        
        # Evaluate each model
        model_results = {}
        for model_name in model_names:
            print(f"\nEvaluating model: {model_name}")
            result = evaluate_model(args, model_name)
            model_results[model_name] = result
        
        # Create comparison plot
        plot_file = create_comparison_plot(args.results_dir, model_results)
        print(f"\nComparison plot saved to: {plot_file}")
    else:
        # Evaluate single model
        evaluate_model(args)

if __name__ == '__main__':
    main()
