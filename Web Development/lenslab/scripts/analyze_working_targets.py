#!/usr/bin/env python3
"""
Analyze and compare original source image with generated working targets.
This script examines edge characteristics, image properties, and blur application.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def load_image(filepath):
    """Load image and return both color and grayscale versions."""
    img_color = cv2.imread(str(filepath))
    if img_color is None:
        raise ValueError(f"Could not load image: {filepath}")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return img_color, img_gray

def analyze_image_properties(img_gray, name):
    """Analyze basic image properties."""
    properties = {
        'name': name,
        'shape': img_gray.shape,
        'dtype': str(img_gray.dtype),
        'min_value': int(np.min(img_gray)),
        'max_value': int(np.max(img_gray)),
        'mean_value': float(np.mean(img_gray)),
        'std_value': float(np.std(img_gray))
    }
    return properties

def find_edge_location(img_gray):
    """Find the primary edge location using gradient analysis."""
    # Apply Sobel operators
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Find the location of maximum gradient
    max_loc = np.unravel_index(np.argmax(gradient_mag), gradient_mag.shape)
    
    # Calculate edge angle using gradient direction
    edge_angle = np.degrees(np.arctan2(sobely[max_loc], sobelx[max_loc]))
    
    return max_loc, edge_angle, gradient_mag

def extract_edge_profile(img_gray, center_row=None):
    """Extract intensity profile across the edge."""
    if center_row is None:
        center_row = img_gray.shape[0] // 2
    
    # Get horizontal profile at center
    profile = img_gray[center_row, :]
    
    # Find edge position by looking for maximum gradient
    gradient = np.abs(np.diff(profile))
    edge_pos = np.argmax(gradient)
    
    return profile, edge_pos, gradient

def measure_edge_sharpness(profile, edge_pos):
    """Measure edge sharpness using various metrics."""
    # Extract region around edge
    start = max(0, edge_pos - 50)
    end = min(len(profile), edge_pos + 50)
    edge_region = profile[start:end]
    
    # Calculate gradient
    gradient = np.abs(np.diff(edge_region))
    max_gradient = np.max(gradient) if len(gradient) > 0 else 0
    
    # Estimate FWHM of gradient (edge spread)
    half_max = max_gradient / 2
    indices = np.where(gradient > half_max)[0]
    if len(indices) > 0:
        fwhm = indices[-1] - indices[0]
    else:
        fwhm = 0
    
    # Calculate contrast - use larger window to ensure we capture the transition
    left_start = max(0, edge_pos - 30)
    left_end = max(0, edge_pos - 5)
    right_start = min(len(profile), edge_pos + 5)
    right_end = min(len(profile), edge_pos + 30)
    
    if left_end > left_start and right_end > right_start:
        left_mean = np.mean(profile[left_start:left_end])
        right_mean = np.mean(profile[right_start:right_end])
        contrast = abs(right_mean - left_mean)
    else:
        # Fallback to simple before/after
        left_mean = np.mean(profile[:edge_pos]) if edge_pos > 0 else 0
        right_mean = np.mean(profile[edge_pos:]) if edge_pos < len(profile) else 0
        contrast = abs(right_mean - left_mean)
    
    return {
        'max_gradient': float(max_gradient),
        'fwhm': int(fwhm),
        'contrast': float(contrast),
        'left_intensity': float(left_mean),
        'right_intensity': float(right_mean)
    }

def create_comparison_visualization(original, working, original_name, working_name, output_path):
    """Create comprehensive visualization comparing two images."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Comparison: {original_name} vs {working_name}', fontsize=16)
    
    # Row 1: Original images
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title(f'{original_name}')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(working, cmap='gray')
    axes[0, 1].set_title(f'{working_name}')
    axes[0, 1].axis('off')
    
    # Difference image
    diff = cv2.absdiff(original, working)
    axes[0, 2].imshow(diff, cmap='hot')
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].axis('off')
    
    # Row 2: Edge profiles
    # Extract profiles from multiple rows
    rows_to_check = [original.shape[0]//4, original.shape[0]//2, 3*original.shape[0]//4]
    colors = ['red', 'green', 'blue']
    
    for idx, (row, color) in enumerate(zip(rows_to_check, colors)):
        profile_orig, edge_pos_orig, _ = extract_edge_profile(original, row)
        profile_work, edge_pos_work, _ = extract_edge_profile(working, row)
        
        axes[1, 0].plot(profile_orig, label=f'Row {row}', color=color, alpha=0.7)
        axes[1, 1].plot(profile_work, label=f'Row {row}', color=color, alpha=0.7)
        
        # Mark edge positions
        axes[1, 0].axvline(edge_pos_orig, color=color, linestyle='--', alpha=0.5)
        axes[1, 1].axvline(edge_pos_work, color=color, linestyle='--', alpha=0.5)
    
    axes[1, 0].set_title(f'{original_name} - Intensity Profiles')
    axes[1, 0].set_xlabel('Pixel Position')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title(f'{working_name} - Intensity Profiles')
    axes[1, 1].set_xlabel('Pixel Position')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Overlay comparison at center row
    center_row = original.shape[0] // 2
    profile_orig_center, _, _ = extract_edge_profile(original, center_row)
    profile_work_center, _, _ = extract_edge_profile(working, center_row)
    
    axes[1, 2].plot(profile_orig_center, label=original_name, linewidth=2)
    axes[1, 2].plot(profile_work_center, label=working_name, linewidth=2, linestyle='--')
    axes[1, 2].set_title('Center Row Comparison')
    axes[1, 2].set_xlabel('Pixel Position')
    axes[1, 2].set_ylabel('Intensity')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Row 3: Gradient analysis
    _, _, grad_orig = find_edge_location(original)
    _, _, grad_work = find_edge_location(working)
    
    axes[2, 0].imshow(grad_orig, cmap='hot')
    axes[2, 0].set_title(f'{original_name} - Gradient Magnitude')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(grad_work, cmap='hot')
    axes[2, 1].set_title(f'{working_name} - Gradient Magnitude')
    axes[2, 1].axis('off')
    
    # Histogram comparison
    axes[2, 2].hist(original.ravel(), bins=256, alpha=0.5, label=original_name, density=True)
    axes[2, 2].hist(working.ravel(), bins=256, alpha=0.5, label=working_name, density=True)
    axes[2, 2].set_title('Intensity Histograms')
    axes[2, 2].set_xlabel('Intensity Value')
    axes[2, 2].set_ylabel('Frequency')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set up paths
    base_dir = Path("/Users/codyhatch/Documents/Github Projects/Projects/Web Development/lenslab")
    original_path = base_dir / "rotated_images" / "Slant-Edge-Target_rotated.png"
    working_dir = base_dir / "working_targets"
    output_dir = base_dir / "results" / "target_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original image
    print("Loading original image...")
    _, original_gray = load_image(original_path)
    
    # Analyze original
    print("\nAnalyzing original image properties:")
    orig_props = analyze_image_properties(original_gray, "Original")
    print(json.dumps(orig_props, indent=2))
    
    # Find edge in original
    orig_edge_loc, orig_edge_angle, _ = find_edge_location(original_gray)
    print(f"\nOriginal edge location: {orig_edge_loc}")
    print(f"Original edge angle: {orig_edge_angle:.2f} degrees")
    
    # Extract profile and measure sharpness
    orig_profile, orig_edge_pos, _ = extract_edge_profile(original_gray)
    orig_sharpness = measure_edge_sharpness(orig_profile, orig_edge_pos)
    print(f"\nOriginal edge sharpness metrics:")
    print(json.dumps(orig_sharpness, indent=2))
    
    # Analyze working targets
    target_files = ["working_perfect.png", "working_sigma_1.5.png", "working_sigma_3.0.png"]
    results = {"original": orig_props}
    
    for target_file in target_files:
        target_path = working_dir / target_file
        if not target_path.exists():
            print(f"\nSkipping {target_file} - file not found")
            continue
            
        print(f"\n{'='*60}")
        print(f"Analyzing {target_file}...")
        
        # Load target image
        _, target_gray = load_image(target_path)
        
        # Analyze properties
        target_props = analyze_image_properties(target_gray, target_file)
        print(json.dumps(target_props, indent=2))
        
        # Compare dimensions
        if target_gray.shape != original_gray.shape:
            print(f"WARNING: Dimension mismatch! Original: {original_gray.shape}, Target: {target_gray.shape}")
        
        # Find edge
        target_edge_loc, target_edge_angle, _ = find_edge_location(target_gray)
        print(f"\nTarget edge location: {target_edge_loc}")
        print(f"Target edge angle: {target_edge_angle:.2f} degrees")
        print(f"Edge angle difference: {abs(target_edge_angle - orig_edge_angle):.2f} degrees")
        
        # Extract profile and measure sharpness
        target_profile, target_edge_pos, _ = extract_edge_profile(target_gray)
        target_sharpness = measure_edge_sharpness(target_profile, target_edge_pos)
        print(f"\nTarget edge sharpness metrics:")
        print(json.dumps(target_sharpness, indent=2))
        
        # Compare sharpness
        print(f"\nSharpness comparison:")
        if orig_sharpness['max_gradient'] > 0:
            print(f"  Gradient ratio: {target_sharpness['max_gradient'] / orig_sharpness['max_gradient']:.3f}")
        else:
            print(f"  Gradient ratio: N/A (original gradient is 0)")
            
        if orig_sharpness['fwhm'] > 0:
            print(f"  FWHM ratio: {target_sharpness['fwhm'] / orig_sharpness['fwhm']:.3f}")
        else:
            print(f"  FWHM ratio: N/A (original FWHM is 0)")
            
        if orig_sharpness['contrast'] > 0:
            print(f"  Contrast ratio: {target_sharpness['contrast'] / orig_sharpness['contrast']:.3f}")
        else:
            print(f"  Contrast: Original={orig_sharpness['contrast']:.1f}, Target={target_sharpness['contrast']:.1f}")
        
        # Create visualization
        viz_path = output_dir / f"comparison_{target_file.replace('.png', '')}.png"
        create_comparison_visualization(original_gray, target_gray, "Original", target_file, viz_path)
        print(f"\nVisualization saved to: {viz_path}")
        
        # Store results
        results[target_file] = {
            "properties": target_props,
            "edge_location": [int(target_edge_loc[0]), int(target_edge_loc[1])],
            "edge_angle": float(target_edge_angle),
            "sharpness": target_sharpness,
            "matches_original_dims": target_gray.shape == original_gray.shape
        }
    
    # Save comprehensive results
    results_path = output_dir / "target_analysis_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nComplete results saved to: {results_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS:")
    print("="*60)
    
    # Check if blur is working
    if "working_sigma_1.5.png" in results and "working_perfect.png" in results:
        perfect_sharp = results["working_perfect.png"]["sharpness"]["max_gradient"]
        blurred_sharp = results["working_sigma_1.5.png"]["sharpness"]["max_gradient"]
        blur_ratio = blurred_sharp / perfect_sharp
        print(f"\nBlur effectiveness: {(1 - blur_ratio)*100:.1f}% reduction in edge sharpness")
        if blur_ratio < 0.8:
            print("✓ Blur is working effectively")
        else:
            print("✗ Blur may not be working as expected")
    
    # Check edge preservation
    edge_preserved = all(results[f]["matches_original_dims"] for f in results if f != "original")
    print(f"\nDimensions preserved: {'✓ Yes' if edge_preserved else '✗ No'}")
    
    # Check contrast levels
    print("\nContrast levels:")
    for filename in results:
        if filename != "original" and "sharpness" in results[filename]:
            contrast = results[filename]["sharpness"]["contrast"]
            print(f"  {filename}: {contrast:.1f}")

if __name__ == "__main__":
    main()