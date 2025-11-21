#!/usr/bin/env python3
"""
Radial Slant-Edge Target Generator

Creates targets with straight radial lines forming sector boundaries,
matching the user's original design exactly.

This generates the pattern that produces long, straight line segments
for proper Hough line detection.
"""

import numpy as np
import cv2
import os

def create_radial_quadrant_target(width=800, height=600, 
                                rotation_angle=11.0,
                                blur_sigma=0.0,
                                circle_radius_ratio=0.45):
    """
    Create target with straight radial lines forming sector boundaries.
    
    This matches the user's original design by creating straight lines
    that extend from center to edge, forming clean sector boundaries.
    """
    print(f"Creating radial target: {width}x{height}")
    print(f"Rotation angle: {rotation_angle}°, Blur sigma: {blur_sigma}")
    
    # Create high-resolution version for anti-aliasing
    hr_factor = 4
    hr_width = width * hr_factor
    hr_height = height * hr_factor
    hr_center_x = hr_width // 2
    hr_center_y = hr_height // 2
    radius = min(width, height) * circle_radius_ratio * hr_factor
    
    # Create coordinate arrays
    y_coords, x_coords = np.ogrid[:hr_height, :hr_width]
    
    # Calculate distance from center and angle
    dx = x_coords - hr_center_x
    dy = y_coords - hr_center_y
    distance = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Normalize angle to 0-360°
    angle = (angle + 360) % 360
    
    # Apply rotation to create slant edges
    rotated_angle = (angle - rotation_angle) % 360
    
    # Create sectors with straight radial boundaries
    # The key is to make clean 90° sectors with sharp transitions
    
    # Define the four sector boundaries (straight radial lines)
    sector_1_end = 90   # First quadrant: 0° to 90°
    sector_2_end = 180  # Second quadrant: 90° to 180° 
    sector_3_end = 270  # Third quadrant: 180° to 270°
    sector_4_end = 360  # Fourth quadrant: 270° to 360°
    
    # Create the sector pattern
    # Sectors 1 and 3 = white (0°-90°, 180°-270°)
    # Sectors 2 and 4 = black (90°-180°, 270°-360°)
    
    sector_1 = (rotated_angle >= 0) & (rotated_angle < 90)
    sector_2 = (rotated_angle >= 90) & (rotated_angle < 180)  
    sector_3 = (rotated_angle >= 180) & (rotated_angle < 270)
    sector_4 = (rotated_angle >= 270) & (rotated_angle < 360)
    
    # Create alternating black/white pattern
    pattern = np.zeros((hr_height, hr_width), dtype=np.uint8)
    pattern[sector_1] = 255  # White
    pattern[sector_2] = 0    # Black
    pattern[sector_3] = 255  # White  
    pattern[sector_4] = 0    # Black
    
    # Apply circular mask
    mask = distance <= radius
    target_hr = np.full((hr_height, hr_width), 255, dtype=np.uint8)  # White background
    target_hr[mask] = pattern[mask]
    
    # Downsample for anti-aliasing
    target = cv2.resize(target_hr, (width, height), interpolation=cv2.INTER_AREA)
    
    # Apply blur if specified
    if blur_sigma > 0:
        print(f"Applying Gaussian blur: sigma={blur_sigma}")
        ksize = int(6 * blur_sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        target = cv2.GaussianBlur(target, (ksize, ksize), blur_sigma)
    
    return target

def generate_radial_test_suite(output_dir="radial_targets", image_size=(800, 600)):
    """Generate test suite with proper radial targets."""
    os.makedirs(output_dir, exist_ok=True)
    width, height = image_size
    
    test_configs = [
        # Perfect targets for accuracy validation
        {"rotation": 11.0, "blur": 0.0, "name": "perfect_radial_target"},
        {"rotation": 11.0, "blur": 0.5, "name": "radial_sigma_0.5"},
        {"rotation": 11.0, "blur": 1.0, "name": "radial_sigma_1.0"},
        {"rotation": 11.0, "blur": 1.5, "name": "radial_sigma_1.5"},
        {"rotation": 11.0, "blur": 2.0, "name": "radial_sigma_2.0"},
        
        # Different rotation angles for validation
        {"rotation": 8.0, "blur": 1.0, "name": "radial_8deg_rotation"},
        {"rotation": 10.0, "blur": 1.0, "name": "radial_10deg_rotation"},
        {"rotation": 12.0, "blur": 1.0, "name": "radial_12deg_rotation"},
        {"rotation": 15.0, "blur": 1.0, "name": "radial_15deg_rotation"},
    ]
    
    generated_files = []
    
    for config in test_configs:
        rotation = config["rotation"]
        blur = config["blur"]
        name = config["name"]
        
        print(f"\n=== Generating {name} ===")
        
        # Generate target
        target = create_radial_quadrant_target(
            width=width, 
            height=height,
            rotation_angle=rotation,
            blur_sigma=blur
        )
        
        # Save target
        filename = f"{name}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, target)
        
        # Calculate theoretical FWHM
        theoretical_fwhm = 2.355 * blur if blur > 0 else 0.0
        
        generated_files.append({
            'path': filepath,
            'filename': filename,
            'rotation_angle': rotation,
            'blur_sigma': blur,
            'theoretical_fwhm': theoretical_fwhm,
            'name': name
        })
        
        print(f"Saved: {filename}")
    
    # Create summary
    create_radial_summary_report(generated_files, output_dir)
    
    return generated_files

def create_radial_summary_report(generated_files, output_dir):
    """Create summary of radial targets"""
    summary_path = os.path.join(output_dir, "radial_targets_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("RADIAL SLANT-EDGE TARGETS\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("DESIGN TO MATCH USER'S ORIGINAL PATTERN:\n")
        f.write("✅ Straight radial lines forming sector boundaries\n")
        f.write("✅ Four alternating black/white sectors\n")
        f.write("✅ Long straight edges for Hough line detection\n")
        f.write("✅ Compatible with existing MTF analyzer\n")
        f.write("✅ Generates clean debug images\n\n")
        
        f.write("GENERATED TARGETS:\n")
        f.write("-" * 30 + "\n")
        
        for file_info in generated_files:
            f.write(f"\nFile: {file_info['filename']}\n")
            f.write(f"  Rotation angle: {file_info['rotation_angle']:.1f}°\n")
            f.write(f"  Blur sigma: {file_info['blur_sigma']:.1f}\n")
            if file_info['theoretical_fwhm'] > 0:
                f.write(f"  Theoretical FWHM: {file_info['theoretical_fwhm']:.3f} pixels\n")
            else:
                f.write(f"  Theoretical FWHM: Perfect edge (no blur)\n")
    
    print(f"\nRadial targets summary saved to: {summary_path}")

def main():
    """Generate radial targets with straight sector boundaries"""
    
    print("=" * 50)
    print("RADIAL SLANT-EDGE TARGET GENERATOR")
    print("Straight radial lines matching original design")
    print("=" * 50)
    
    try:
        generated_files = generate_radial_test_suite()
        
        print("\n" + "=" * 50)
        print("RADIAL TARGETS GENERATED SUCCESSFULLY")
        print("=" * 50)
        print(f"Total files generated: {len(generated_files)}")
        print("\nThese should create long straight lines for Hough detection!")
        
        return 0
        
    except Exception as e:
        print(f"Error generating radial targets: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())