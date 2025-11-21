#!/usr/bin/env python3
"""
Generate test targets with known Gaussian blur for FWHM validation
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the src/python directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'src' / 'python'))

from rotate_and_blur import read_file

def generate_test_targets(input_path, output_dir, sigma_values, angle=8.0):
    """
    Generate test targets with multiple sigma values for FWHM validation.
    
    Parameters:
    input_path (str): Path to input slant edge target
    output_dir (str): Directory to save test targets
    sigma_values (list): List of sigma values to test
    angle (float): Rotation angle in degrees
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the input file
    print(f"Reading input file: {input_path}")
    img = read_file(input_path)
    
    if img is None:
        raise ValueError("Failed to read input file")
    
    height, width = img.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Rotate image once
    print(f"Rotating image by {angle} degrees")
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height),
                             flags=cv2.INTER_LINEAR, borderValue=255)
    
    # Create circular mask
    center = (width//2, height//2)
    radius = min(width, height)//2
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    
    # Apply mask to rotated image
    masked = cv2.bitwise_and(rotated, mask)
    masked[mask == 0] = 255  # Set outside to white
    
    # Save unblurred rotated target
    base_name = Path(input_path).stem
    rotated_path = os.path.join(output_dir, f"{base_name}_test_rotated.png")
    cv2.imwrite(rotated_path, masked)
    print(f"Saved rotated image to: {rotated_path}")
    
    generated_files = []
    
    # Generate blurred versions for each sigma
    for sigma in sigma_values:
        print(f"Applying Gaussian blur with sigma={sigma}")
        blurred = cv2.GaussianBlur(masked, (0, 0), sigma)
        
        # Calculate theoretical FWHM
        theoretical_fwhm = 2.355 * sigma
        
        # Save blurred image
        blurred_path = os.path.join(output_dir, f"{base_name}_sigma_{sigma:.1f}_blurred.png")
        cv2.imwrite(blurred_path, blurred)
        
        generated_files.append({
            'path': blurred_path,
            'sigma': sigma,
            'theoretical_fwhm': theoretical_fwhm
        })
        
        print(f"Saved blurred image (σ={sigma}, theoretical FWHM={theoretical_fwhm:.3f}): {blurred_path}")
    
    return generated_files

def main():
    # Configuration
    input_path = "rotated_images/Slant-Edge-Target_rotated.png"
    output_dir = "test_targets"
    sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    angle = 8.0
    
    try:
        generated_files = generate_test_targets(input_path, output_dir, sigma_values, angle)
        
        print("\n" + "="*60)
        print("TEST TARGETS GENERATED SUCCESSFULLY")
        print("="*60)
        print("File\t\t\t\tSigma\tTheoretical FWHM")
        print("-"*60)
        
        for file_info in generated_files:
            filename = Path(file_info['path']).name
            print(f"{filename:<30}\t{file_info['sigma']:.1f}\t{file_info['theoretical_fwhm']:.3f}")
        
        print("\nThese targets can be used to validate FWHM calculations.")
        print("Expected: Measured FWHM should match theoretical FWHM (±5%)")
        
    except Exception as e:
        print(f"Error generating test targets: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())