#!/usr/bin/env python3
"""
Working Target Generator

Uses the user's proven working target design and applies different blur levels.
This avoids trying to recreate the geometry and just builds on what already works.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the src/python directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'src' / 'python'))
from rotate_and_blur import read_file

def apply_controlled_blur(image, blur_sigma):
    """Apply controlled Gaussian blur to the working target."""
    if blur_sigma <= 0:
        return image
        
    print(f"Applying controlled blur: sigma={blur_sigma}")
    
    # Use a properly sized kernel
    ksize = int(6 * blur_sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    
    blurred = cv2.GaussianBlur(image, (ksize, ksize), blur_sigma)
    return blurred

def generate_working_test_suite(input_path, output_dir="working_targets"):
    """
    Generate test suite based on the user's proven working target.
    
    This takes the target that we KNOW works with the MTF analyzer
    and creates versions with different blur levels.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the proven working target
    print(f"Reading proven working target: {input_path}")
    base_image = read_file(input_path)
    
    if base_image is None:
        raise ValueError("Failed to read working target")
    
    print(f"Working target dimensions: {base_image.shape}")
    
    # Test configurations using the proven target
    test_configs = [
        {"blur": 0.0, "name": "working_perfect"},
        {"blur": 0.5, "name": "working_sigma_0.5"},
        {"blur": 1.0, "name": "working_sigma_1.0"},
        {"blur": 1.5, "name": "working_sigma_1.5"},
        {"blur": 2.0, "name": "working_sigma_2.0"},
        {"blur": 2.5, "name": "working_sigma_2.5"},
        {"blur": 3.0, "name": "working_sigma_3.0"},
    ]
    
    generated_files = []
    
    for config in test_configs:
        blur = config["blur"]
        name = config["name"]
        
        print(f"\n=== Generating {name} ===")
        
        # Apply blur to the working target
        if blur == 0.0:
            target = base_image.copy()
        else:
            target = apply_controlled_blur(base_image, blur)
        
        # Save target
        filename = f"{name}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, target)
        
        # Calculate theoretical FWHM
        theoretical_fwhm = 2.355 * blur if blur > 0 else 0.0
        
        generated_files.append({
            'path': filepath,
            'filename': filename,
            'blur_sigma': blur,
            'theoretical_fwhm': theoretical_fwhm,
            'name': name
        })
        
        print(f"Saved: {filename} (σ={blur}, theoretical FWHM={theoretical_fwhm:.3f})")
    
    # Create summary
    create_working_summary_report(generated_files, output_dir)
    
    return generated_files

def create_working_summary_report(generated_files, output_dir):
    """Create summary of working targets"""
    summary_path = os.path.join(output_dir, "working_targets_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("WORKING TARGET TEST SUITE\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("APPROACH:\n")
        f.write("✅ Uses the user's proven working target as base\n")
        f.write("✅ Applies controlled blur levels for validation\n")
        f.write("✅ Maintains the exact edge pattern that works\n")
        f.write("✅ Avoids rotation artifacts\n")
        f.write("✅ Provides known ground truth for accuracy testing\n\n")
        
        f.write("GENERATED TARGETS:\n")
        f.write("-" * 30 + "\n")
        
        for file_info in generated_files:
            f.write(f"\nFile: {file_info['filename']}\n")
            f.write(f"  Blur sigma: {file_info['blur_sigma']:.1f}\n")
            if file_info['theoretical_fwhm'] > 0:
                f.write(f"  Theoretical FWHM: {file_info['theoretical_fwhm']:.3f} pixels\n")
            else:
                f.write(f"  Theoretical FWHM: Perfect edge (no blur)\n")
            f.write(f"  Purpose: {file_info['name'].replace('_', ' ').title()}\n")
    
    print(f"\nWorking targets summary saved to: {summary_path}")

def main():
    """Generate targets based on the proven working design"""
    
    # Use the target that we KNOW works
    input_path = "rotated_images/Slant-Edge-Target_rotated.png"
    
    print("=" * 50)
    print("WORKING TARGET GENERATOR")
    print("Based on proven working target design")
    print("=" * 50)
    
    try:
        generated_files = generate_working_test_suite(input_path)
        
        print("\n" + "=" * 50)
        print("WORKING TARGETS GENERATED SUCCESSFULLY")
        print("=" * 50)
        print(f"Total files generated: {len(generated_files)}")
        print("\nThese targets use your proven working design!")
        print("\nNext steps:")
        print("1. Test with coordinate-fixed MTF analyzer")
        print("2. Validate FWHM accuracy against theoretical values")
        print("3. Compare measurement consistency")
        
        return 0
        
    except Exception as e:
        print(f"Error generating working targets: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())