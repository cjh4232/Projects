#!/usr/bin/env python3
"""Quick visual check of the images to understand their content."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    base_dir = Path("/Users/codyhatch/Documents/Github Projects/Projects/Web Development/lenslab")
    
    # Images to check
    images = [
        ("rotated_images/Slant-Edge-Target_rotated.png", "Original Rotated"),
        ("working_targets/working_perfect.png", "Working Perfect"),
        ("working_targets/working_sigma_1.5.png", "Working Sigma 1.5"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (img_path, title) in enumerate(images):
        full_path = base_dir / img_path
        img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Could not load: {img_path}")
            continue
            
        # Show image
        axes[0, idx].imshow(img, cmap='gray')
        axes[0, idx].set_title(title)
        axes[0, idx].axis('off')
        
        # Show center row profile
        center_row = img.shape[0] // 2
        profile = img[center_row, :]
        axes[1, idx].plot(profile)
        axes[1, idx].set_title(f"{title} - Center Row Profile")
        axes[1, idx].set_xlabel('Pixel Position')
        axes[1, idx].set_ylabel('Intensity')
        axes[1, idx].grid(True, alpha=0.3)
        axes[1, idx].set_ylim(-5, 260)
        
        # Print some stats
        print(f"\n{title}:")
        print(f"  Shape: {img.shape}")
        print(f"  Min/Max: {img.min()}/{img.max()}")
        print(f"  Mean: {img.mean():.2f}")
        print(f"  Unique values: {len(np.unique(img))}")
        
        # Check if it's mostly black
        black_pixels = np.sum(img < 10)
        white_pixels = np.sum(img > 245)
        print(f"  Black pixels (<10): {black_pixels} ({100*black_pixels/img.size:.1f}%)")
        print(f"  White pixels (>245): {white_pixels} ({100*white_pixels/img.size:.1f}%)")
    
    plt.tight_layout()
    output_path = base_dir / "results" / "target_analysis" / "quick_image_check.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()
    
    # Also check what's in the working_targets_summary.txt
    summary_path = base_dir / "working_targets" / "working_targets_summary.txt"
    if summary_path.exists():
        print(f"\n{'='*60}")
        print("Content of working_targets_summary.txt:")
        print('='*60)
        with open(summary_path, 'r') as f:
            print(f.read())

if __name__ == "__main__":
    main()