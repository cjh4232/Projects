#!/usr/bin/env python3
"""
Analyze edge patterns to understand the actual edge structure in our images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

def find_all_edges(img_gray, threshold=20):
    """Find all significant edges in the image."""
    # Apply Sobel to find edges
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Find significant edges
    edge_mask = gradient_mag > threshold
    
    return gradient_mag, edge_mask, sobelx, sobely

def analyze_edge_structure(img_gray, name):
    """Analyze the edge structure in detail."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print('='*60)
    
    # Get image stats
    h, w = img_gray.shape
    print(f"Image shape: {h}x{w}")
    
    # Find edges
    gradient_mag, edge_mask, sobelx, sobely = find_all_edges(img_gray)
    
    # Count edge pixels
    edge_pixels = np.sum(edge_mask)
    print(f"Edge pixels: {edge_pixels} ({100*edge_pixels/(h*w):.2f}% of image)")
    
    # Find the primary edge by looking at different approaches
    # 1. Look for vertical edges (high horizontal gradient)
    vertical_edges = np.abs(sobelx) > 20
    
    # 2. Look for horizontal edges (high vertical gradient)  
    horizontal_edges = np.abs(sobely) > 20
    
    # 3. Find the strongest gradients
    max_gradient_loc = np.unravel_index(np.argmax(gradient_mag), gradient_mag.shape)
    print(f"Maximum gradient at: row={max_gradient_loc[0]}, col={max_gradient_loc[1]}")
    print(f"Maximum gradient value: {gradient_mag[max_gradient_loc]:.2f}")
    
    # Analyze edge orientation
    if gradient_mag[max_gradient_loc] > 0:
        angle = np.degrees(np.arctan2(sobely[max_gradient_loc], sobelx[max_gradient_loc]))
        print(f"Edge angle at max gradient: {angle:.2f} degrees")
    
    # Look at row and column profiles
    # Sum gradients along rows and columns to find dominant edges
    row_gradient_sum = np.sum(gradient_mag, axis=1)
    col_gradient_sum = np.sum(gradient_mag, axis=0)
    
    max_row = np.argmax(row_gradient_sum)
    max_col = np.argmax(col_gradient_sum)
    
    print(f"Row with most edge activity: {max_row}")
    print(f"Column with most edge activity: {max_col}")
    
    # Look for the main transition in the image
    # Check multiple rows
    print("\nEdge positions in different rows:")
    rows_to_check = [h//4, h//2, 3*h//4]
    for row in rows_to_check:
        profile = img_gray[row, :]
        gradient = np.abs(np.diff(profile))
        if len(gradient) > 0 and np.max(gradient) > 10:
            edge_pos = np.argmax(gradient)
            print(f"  Row {row}: edge at column {edge_pos}, gradient={gradient[edge_pos]:.1f}")
    
    # Check for slanted edge pattern
    # A slanted edge should show edge positions that change with row
    edge_positions = []
    for row in range(0, h, 10):  # Sample every 10 rows
        profile = img_gray[row, :]
        gradient = np.abs(np.diff(profile))
        if len(gradient) > 0 and np.max(gradient) > 20:
            edge_pos = np.argmax(gradient)
            edge_positions.append((row, edge_pos))
    
    if len(edge_positions) > 2:
        rows = [p[0] for p in edge_positions]
        cols = [p[1] for p in edge_positions]
        
        # Fit a line to see if it's slanted
        if len(set(cols)) > 1:  # Check if columns vary
            coeffs = np.polyfit(rows, cols, 1)
            slope = coeffs[0]
            angle = np.degrees(np.arctan(slope))
            print(f"\nSlanted edge detected:")
            print(f"  Slope: {slope:.4f} pixels/row")
            print(f"  Angle: {angle:.2f} degrees from vertical")
    
    return gradient_mag, edge_mask

def create_detailed_visualization(images_data, output_path):
    """Create detailed visualization of edge patterns."""
    n_images = len(images_data)
    fig, axes = plt.subplots(n_images, 4, figsize=(16, 4*n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (img, gradient_mag, name) in enumerate(images_data):
        # Original image
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title(f'{name} - Original')
        axes[idx, 0].axis('off')
        
        # Gradient magnitude
        axes[idx, 1].imshow(gradient_mag, cmap='hot')
        axes[idx, 1].set_title(f'{name} - Gradient Magnitude')
        axes[idx, 1].axis('off')
        
        # Multiple row profiles
        h, w = img.shape
        rows = [h//4, h//2, 3*h//4]
        colors = ['red', 'green', 'blue']
        
        for row, color in zip(rows, colors):
            profile = img[row, :]
            axes[idx, 2].plot(profile, label=f'Row {row}', color=color)
            
            # Mark detected edge
            gradient = np.abs(np.diff(profile))
            if len(gradient) > 0:
                edge_pos = np.argmax(gradient)
                axes[idx, 2].axvline(edge_pos, color=color, linestyle='--', alpha=0.5)
        
        axes[idx, 2].set_title(f'{name} - Row Profiles')
        axes[idx, 2].set_xlabel('Column')
        axes[idx, 2].set_ylabel('Intensity')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True, alpha=0.3)
        
        # Column profiles
        cols = [w//4, w//2, 3*w//4]
        
        for col, color in zip(cols, colors):
            profile = img[:, col]
            axes[idx, 3].plot(profile, label=f'Col {col}', color=color)
        
        axes[idx, 3].set_title(f'{name} - Column Profiles')
        axes[idx, 3].set_xlabel('Row')
        axes[idx, 3].set_ylabel('Intensity')
        axes[idx, 3].legend()
        axes[idx, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    base_dir = Path("/Users/codyhatch/Documents/Github Projects/Projects/Web Development/lenslab")
    output_dir = base_dir / "results" / "target_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Images to analyze
    image_files = [
        ("rotated_images/Slant-Edge-Target_rotated.png", "Original"),
        ("working_targets/working_perfect.png", "Working Perfect"),
        ("working_targets/working_sigma_1.5.png", "Working Sigma 1.5"),
    ]
    
    images_data = []
    
    for img_path, name in image_files:
        full_path = base_dir / img_path
        img = cv2.imread(str(full_path), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Could not load: {img_path}")
            continue
        
        # Analyze edge structure
        gradient_mag, edge_mask = analyze_edge_structure(img, name)
        
        images_data.append((img, gradient_mag, name))
    
    # Create visualization
    viz_path = output_dir / "edge_pattern_analysis.png"
    create_detailed_visualization(images_data, viz_path)
    print(f"\nVisualization saved to: {viz_path}")
    
    # Additional analysis: Check if the images are truly different
    if len(images_data) >= 2:
        print("\n" + "="*60)
        print("IMAGE COMPARISON")
        print("="*60)
        
        original = images_data[0][0]
        perfect = images_data[1][0]
        
        # Check if they're identical
        if np.array_equal(original, perfect):
            print("✓ Original and Working Perfect are IDENTICAL (as expected)")
        else:
            diff = cv2.absdiff(original, perfect)
            print(f"✗ Original and Working Perfect differ!")
            print(f"  Max difference: {np.max(diff)}")
            print(f"  Number of different pixels: {np.sum(diff > 0)}")
        
        if len(images_data) >= 3:
            blurred = images_data[2][0]
            
            # Compare with perfect
            if np.array_equal(perfect, blurred):
                print("✗ Working Perfect and Sigma 1.5 are IDENTICAL (blur not applied!)")
            else:
                diff = cv2.absdiff(perfect, blurred)
                print("✓ Blur was applied to Sigma 1.5")
                print(f"  Max difference: {np.max(diff)}")
                print(f"  Number of different pixels: {np.sum(diff > 0)}")
                
                # Check if blur reduced edge sharpness
                perfect_grad = images_data[1][1]
                blurred_grad = images_data[2][1]
                
                perfect_max = np.max(perfect_grad)
                blurred_max = np.max(blurred_grad)
                
                print(f"\n  Perfect max gradient: {perfect_max:.2f}")
                print(f"  Blurred max gradient: {blurred_max:.2f}")
                
                if blurred_max < perfect_max:
                    print(f"  ✓ Blur reduced max gradient by {100*(1-blurred_max/perfect_max):.1f}%")
                else:
                    print(f"  ✗ Blur did not reduce gradient (increased by {100*(blurred_max/perfect_max-1):.1f}%)")

if __name__ == "__main__":
    main()