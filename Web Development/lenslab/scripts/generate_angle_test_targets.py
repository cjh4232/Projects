#!/usr/bin/env python3
"""
Generate comprehensive test targets with various angles for validation
Creates both optimal and suboptimal angles to test angle validation logic
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the src/python directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'src' / 'python'))

from rotate_and_blur import read_file

def generate_angle_test_targets(input_path, output_dir):
    """
    Generate test targets with various angles for comprehensive validation.
    
    Test Matrix:
    - Optimal angles: 8°, 10°, 12°, 15° (should pass with highest accuracy)
    - Acceptable angles: 5°, 18°, 20° (should pass with warning)
    - Invalid angles: 3°, 25°, 45°, 90° (should be rejected)
    - Multiple blur levels for each angle
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Test configuration
    angle_categories = {
        'optimal': [8.0, 10.0, 12.0, 15.0],      # Should pass with best accuracy
        'acceptable': [5.0, 18.0, 20.0],         # Should pass with warning  
        'invalid': [3.0, 25.0, 45.0, 90.0]       # Should be rejected
    }
    
    sigma_values = [1.0, 1.5, 2.0]  # Representative blur levels
    
    # Read the input file
    print(f"Reading input file: {input_path}")
    img = read_file(input_path)
    
    if img is None:
        raise ValueError("Failed to read input file")
    
    height, width = img.shape
    print(f"Image dimensions: {width}x{height}")
    
    generated_files = []
    
    for category, angles in angle_categories.items():
        print(f"\n=== Generating {category.upper()} angle targets ===")
        
        for angle in angles:
            for sigma in sigma_values:
                print(f"Creating target: angle={angle}°, sigma={sigma}, category={category}")
                
                # Rotate image
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
                
                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(masked, (0, 0), sigma)
                
                # Calculate theoretical FWHM
                theoretical_fwhm = 2.355 * sigma
                
                # Generate filename
                filename = f"test_angle_{angle:04.1f}deg_sigma_{sigma:.1f}_{category}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save image
                cv2.imwrite(filepath, blurred)
                
                generated_files.append({
                    'path': filepath,
                    'filename': filename,
                    'angle': angle,
                    'sigma': sigma,
                    'theoretical_fwhm': theoretical_fwhm,
                    'category': category,
                    'expected_result': 'PASS' if category in ['optimal', 'acceptable'] else 'REJECT'
                })
                
                print(f"  Saved: {filename}")
    
    return generated_files

def create_test_summary(generated_files, output_dir):
    """Create a summary report of generated test files"""
    summary_path = os.path.join(output_dir, "test_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("ANGLE VALIDATION TEST TARGETS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Test Categories:\n")
        f.write("- OPTIMAL (8-15°): Should pass with highest accuracy\n")
        f.write("- ACCEPTABLE (5°, 18-20°): Should pass with warning\n") 
        f.write("- INVALID (3°, 25°, 45°, 90°): Should be rejected\n\n")
        
        f.write("Expected Results:\n")
        f.write("- OPTIMAL: <3% FWHM error, no warnings\n")
        f.write("- ACCEPTABLE: <5% FWHM error, angle warning\n")
        f.write("- INVALID: Analysis rejection due to poor angle\n\n")
        
        categories = {}
        for file_info in generated_files:
            cat = file_info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(file_info)
        
        for category in ['optimal', 'acceptable', 'invalid']:
            if category in categories:
                f.write(f"\n{category.upper()} ANGLES:\n")
                f.write("-" * 30 + "\n")
                
                for file_info in categories[category]:
                    f.write(f"File: {file_info['filename']}\n")
                    f.write(f"  Angle: {file_info['angle']:.1f}°\n")
                    f.write(f"  Sigma: {file_info['sigma']:.1f}\n")
                    f.write(f"  Theoretical FWHM: {file_info['theoretical_fwhm']:.3f}\n")
                    f.write(f"  Expected: {file_info['expected_result']}\n\n")
    
    print(f"\nTest summary saved to: {summary_path}")

def main():
    # Configuration
    input_path = "rotated_images/Slant-Edge-Target_rotated.png"
    output_dir = "angle_test_targets"
    
    try:
        generated_files = generate_angle_test_targets(input_path, output_dir)
        create_test_summary(generated_files, output_dir)
        
        print("\n" + "="*60)
        print("ANGLE TEST TARGETS GENERATED SUCCESSFULLY")
        print("="*60)
        
        categories = {}
        for file_info in generated_files:
            cat = file_info['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"Total files generated: {len(generated_files)}")
        for category, count in categories.items():
            print(f"  {category.capitalize()}: {count} files")
        
        print("\nNext steps:")
        print("1. Compile improved MTF analyzer")
        print("2. Run validation on all test targets")
        print("3. Verify angle detection and FWHM accuracy")
        
    except Exception as e:
        print(f"Error generating test targets: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())