import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from collections import defaultdict
import re

def load_image(image_path):
    """Load and convert image to numpy array"""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)

def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index"""
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = np.mean(img1, axis=2)
        img2_gray = np.mean(img2, axis=2)
    else:
        img1_gray = img1
        img2_gray = img2
    
    return ssim(img1_gray, img2_gray, data_range=255)

def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio"""
    return psnr(img1, img2, data_range=255)

def calculate_lpips(img1, img2, lpips_model):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity)"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Convert numpy arrays to PIL Images then to tensors
    img1_pil = Image.fromarray(img1.astype(np.uint8))
    img2_pil = Image.fromarray(img2.astype(np.uint8))
    
    img1_tensor = transform(img1_pil).unsqueeze(0)
    img2_tensor = transform(img2_pil).unsqueeze(0)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    
    with torch.no_grad():
        lpips_score = lpips_model(img1_tensor, img2_tensor)
    
    return lpips_score.item()

def parse_filename(filename):
    """Parse filename to extract base identifier and type"""
    # Extract the base part before the last underscore and file extension
    parts = filename.split('_')
    if len(parts) >= 3:
        # Join all parts except the last 2 (which should be type and extension)
        base_id = '_'.join(parts[:-2])
        img_type = parts[-2]  # real or fake
        modality = parts[-1].split('.')[0]  # A or B
        return base_id, img_type, modality
    return None, None, None

def group_images(image_dir):
    """Group images by their base identifier"""
    image_groups = defaultdict(list)
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            base_id, img_type, modality = parse_filename(filename)
            if base_id and img_type and modality:
                image_groups[base_id].append({
                    'filename': filename,
                    'type': img_type,
                    'modality': modality,
                    'path': os.path.join(image_dir, filename)
                })
    
    return image_groups

def calculate_metrics(image_dir):
    """Calculate all metrics for PET images (modality B)"""
    print("Initializing LPIPS model...")
    # Initialize LPIPS model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_model = lpips.LPIPS(net='vgg').to(device)
    
    print("Grouping images...")
    image_groups = group_images(image_dir)
    
    metrics = {
        'lpips': [],
        'mae': [],
        'ssim': [],
        'psnr': []
    }
    
    valid_pairs = 0
    total_groups = len(image_groups)
    
    print(f"Processing {total_groups} image groups...")
    
    for i, (base_id, images) in enumerate(image_groups.items()):
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_groups} groups...")
        
        # Find real_B and fake_B images
        real_b = None
        fake_b = None
        
        for img_info in images:
            if img_info['type'] == 'real' and img_info['modality'] == 'B':
                real_b = img_info['path']
            elif img_info['type'] == 'fake' and img_info['modality'] == 'B':
                fake_b = img_info['path']
        
        if real_b and fake_b:
            try:
                # Load images
                real_img = load_image(real_b)
                fake_img = load_image(fake_b)
                
                # Ensure images have the same dimensions
                if real_img.shape != fake_img.shape:
                    print(f"Warning: Shape mismatch for {base_id}. Skipping...")
                    continue
                
                # Calculate metrics
                mae_score = calculate_mae(real_img, fake_img)
                ssim_score = calculate_ssim(real_img, fake_img)
                psnr_score = calculate_psnr(real_img, fake_img)
                lpips_score = calculate_lpips(real_img, fake_img, lpips_model)
                
                # Store metrics
                metrics['mae'].append(mae_score)
                metrics['ssim'].append(ssim_score)
                metrics['psnr'].append(psnr_score)
                metrics['lpips'].append(lpips_score)
                
                valid_pairs += 1
                
            except Exception as e:
                print(f"Error processing {base_id}: {str(e)}")
                continue
        else:
            print(f"Warning: Could not find both real_B and fake_B for {base_id}")
    
    return metrics, valid_pairs

def print_results(metrics, valid_pairs):
    """Print the calculated metrics"""
    print(f"\n{'='*50}")
    print(f"RESULTS ({valid_pairs} valid image pairs)")
    print(f"{'='*50}")
    
    for metric_name, values in metrics.items():
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name.upper():>6}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{metric_name.upper():>6}: No valid values")
    
    print(f"{'='*50}")

def main():
    """Main function"""
    # Set your image directory path here
    image_directory = input("Enter the path to your image directory: ").strip()
    
    if not os.path.exists(image_directory):
        print(f"Error: Directory '{image_directory}' does not exist!")
        return
    
    print(f"Processing images in: {image_directory}")
    
    try:
        metrics, valid_pairs = calculate_metrics(image_directory)
        print_results(metrics, valid_pairs)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Required packages installation note
    print("Required packages: pip install torch torchvision lpips pillow scikit-image")
    print("="*70)
    main()
