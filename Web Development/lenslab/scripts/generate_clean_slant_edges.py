#!/usr/bin/env python3
"""
Native Slant-Edge Target Generator

Generates clean, artifact-free slant-edge targets using analytical functions
instead of rotation. Follows ISO 12233 methodology for proper MTF testing.

Key improvements over rotation-based generation:
1. Native edge generation (no interpolation artifacts)
2. Sub-pixel anti-aliasing for smooth edges
3. Realistic blur application in spatial domain
4. Quadrant-based design for sagittal/tangential measurements
5. Known ground truth for validation
"""

import numpy as np
import cv2
import os

def create_analytical_edge(width, height, angle_deg, edge_position=0.5, subpixel_factor=4):
    """
    Create a clean slant edge using analytical function (no rotation artifacts).
    
    Parameters:
    width, height: Image dimensions
    angle_deg: Edge angle in degrees
    edge_position: Position of edge (0.0 = left/top, 1.0 = right/bottom)
    subpixel_factor: Oversampling factor for anti-aliasing
    
    Returns:
    numpy.ndarray: Clean binary edge image (0-255)
    """
    print(f"Generating analytical edge: angle={angle_deg}°, position={edge_position}")
    
    # Create high-resolution grid for anti-aliasing
    hr_width = width * subpixel_factor
    hr_height = height * subpixel_factor
    
    # Create coordinate grids
    x = np.linspace(0, width, hr_width, endpoint=False)
    y = np.linspace(0, height, hr_height, endpoint=False)
    X, Y = np.meshgrid(x, y)
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle_deg)
    
    # Calculate edge line parameters
    # Edge goes through center point at specified position
    center_x = width * edge_position
    center_y = height * 0.5
    
    # Line equation: normal_x * (x - center_x) + normal_y * (y - center_y) = 0
    normal_x = np.sin(angle_rad)  # Normal to the edge
    normal_y = -np.cos(angle_rad)
    
    # Calculate signed distance from each point to the edge line
    distance = normal_x * (X - center_x) + normal_y * (Y - center_y)
    
    # Create binary edge (step function)
    edge_hr = (distance > 0).astype(np.float64) * 255.0
    
    # Anti-alias by downsampling
    edge = cv2.resize(edge_hr, (width, height), interpolation=cv2.INTER_AREA)
    
    return edge.astype(np.uint8)

def apply_realistic_blur(image, sigma):
    """
    Apply Gaussian blur using spatial domain for realistic edge characteristics.
    
    Spatial domain blur creates more realistic edges that match real-world
    imaging conditions, unlike frequency domain which is too mathematically perfect.
    
    Parameters:
    image: Input image
    sigma: Gaussian blur standard deviation
    
    Returns:
    numpy.ndarray: Blurred image
    """
    if sigma <= 0:
        return image
        
    print(f"Applying realistic spatial domain blur: sigma={sigma}")
    
    # Use OpenCV's GaussianBlur for realistic edge characteristics
    # This matches real-world imaging better than frequency domain
    ksize = int(6 * sigma + 1)  # Kernel size (must be odd)
    if ksize % 2 == 0:
        ksize += 1
    
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    
    return blurred

def create_quadrant_slant_edge_target(width=800, height=600, 
                                     horizontal_angle=11.0, vertical_angle=79.0,
                                     blur_sigma=0.0):
    """
    Create a quadrant-based slant-edge target for sagittal/tangential MTF measurement.
    
    Each quadrant contains a single, clean slant edge at the appropriate angle:
    - Top-left & bottom-left: Horizontal edges (~11°) for sagittal measurement
    - Top-right & bottom-right: Vertical edges (~79°) for tangential measurement
    
    Parameters:
    width, height: Target dimensions
    horizontal_angle: Angle for horizontal edges (sagittal)
    vertical_angle: Angle for vertical edges (tangential) 
    blur_sigma: Gaussian blur to apply
    
    Returns:
    numpy.ndarray: Complete quadrant target
    """
    print(f"Creating quadrant target: {width}x{height}")
    print(f"Horizontal edges: {horizontal_angle}°, Vertical edges: {vertical_angle}°")
    
    # Create the four quadrants
    quad_width = width // 2
    quad_height = height // 2
    
    # Top-left: Horizontal edge (sagittal)
    tl_edge = create_analytical_edge(quad_width, quad_height, horizontal_angle, edge_position=0.5)
    
    # Top-right: Vertical edge (tangential)  
    tr_edge = create_analytical_edge(quad_width, quad_height, vertical_angle, edge_position=0.5)
    
    # Bottom-left: Horizontal edge (sagittal)
    bl_edge = create_analytical_edge(quad_width, quad_height, horizontal_angle, edge_position=0.5)
    
    # Bottom-right: Vertical edge (tangential)
    br_edge = create_analytical_edge(quad_width, quad_height, vertical_angle, edge_position=0.5)
    
    # Combine quadrants
    target = np.zeros((height, width), dtype=np.uint8)
    target[0:quad_height, 0:quad_width] = tl_edge
    target[0:quad_height, quad_width:width] = tr_edge
    target[quad_height:height, 0:quad_width] = bl_edge
    target[quad_height:height, quad_width:width] = br_edge
    
    # Apply blur if specified
    if blur_sigma > 0:
        target = apply_realistic_blur(target, blur_sigma)
    
    return target

def generate_test_suite(output_dir="clean_targets", image_size=(800, 600)):
    """
    Generate a comprehensive test suite with clean targets.
    
    Parameters:
    output_dir: Directory to save targets
    image_size: (width, height) of generated targets
    """
    os.makedirs(output_dir, exist_ok=True)
    width, height = image_size
    
    # Test configurations
    test_configs = [
        # Perfect targets for accuracy validation
        {"angles": (11.0, 79.0), "blur": 0.0, "name": "perfect_target"},
        {"angles": (11.0, 79.0), "blur": 0.5, "name": "target_sigma_0.5"},
        {"angles": (11.0, 79.0), "blur": 1.0, "name": "target_sigma_1.0"},
        {"angles": (11.0, 79.0), "blur": 1.5, "name": "target_sigma_1.5"},
        {"angles": (11.0, 79.0), "blur": 2.0, "name": "target_sigma_2.0"},
        
        # Angle validation targets
        {"angles": (8.0, 82.0), "blur": 1.0, "name": "optimal_angle_8deg"},
        {"angles": (10.0, 80.0), "blur": 1.0, "name": "optimal_angle_10deg"},
        {"angles": (12.0, 78.0), "blur": 1.0, "name": "optimal_angle_12deg"},
        {"angles": (15.0, 75.0), "blur": 1.0, "name": "optimal_angle_15deg"},
        
        {"angles": (5.0, 85.0), "blur": 1.0, "name": "acceptable_angle_5deg"},
        {"angles": (18.0, 72.0), "blur": 1.0, "name": "acceptable_angle_18deg"},
        {"angles": (20.0, 70.0), "blur": 1.0, "name": "acceptable_angle_20deg"},
    ]
    
    generated_files = []
    
    for config in test_configs:
        h_angle, v_angle = config["angles"]
        blur = config["blur"]
        name = config["name"]
        
        print(f"\n=== Generating {name} ===")
        
        # Generate target
        target = create_quadrant_slant_edge_target(
            width=width, 
            height=height,
            horizontal_angle=h_angle,
            vertical_angle=v_angle,
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
            'horizontal_angle': h_angle,
            'vertical_angle': v_angle,
            'blur_sigma': blur,
            'theoretical_fwhm': theoretical_fwhm,
            'name': name
        })
        
        print(f"Saved: {filename}")
    
    # Create summary
    create_summary_report(generated_files, output_dir)
    
    return generated_files

def create_summary_report(generated_files, output_dir):
    """Create a detailed summary of generated clean targets"""
    summary_path = os.path.join(output_dir, "clean_targets_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("CLEAN SLANT-EDGE TARGETS - ARTIFACT-FREE GENERATION\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("IMPROVEMENTS OVER ROTATION-BASED GENERATION:\n")
        f.write("✅ Native analytical edge generation (no interpolation artifacts)\n")
        f.write("✅ Sub-pixel anti-aliasing for smooth edges\n")
        f.write("✅ Realistic blur in spatial domain (matches real-world imaging)\n")
        f.write("✅ ISO 12233 compliant edge profiles\n")
        f.write("✅ Known ground truth for validation\n")
        f.write("✅ Quadrant design for sagittal/tangential measurement\n\n")
        
        f.write("TARGET DESIGN:\n")
        f.write("- Top-left & Bottom-left: Horizontal edges (sagittal measurement)\n")
        f.write("- Top-right & Bottom-right: Vertical edges (tangential measurement)\n")
        f.write("- Complementary angles ensure proper cross-pattern analysis\n\n")
        
        f.write("GENERATED TARGETS:\n")
        f.write("-" * 40 + "\n")
        
        for file_info in generated_files:
            f.write(f"\nFile: {file_info['filename']}\n")
            f.write(f"  Horizontal angle: {file_info['horizontal_angle']:.1f}°\n")
            f.write(f"  Vertical angle: {file_info['vertical_angle']:.1f}°\n")
            f.write(f"  Blur sigma: {file_info['blur_sigma']:.1f}\n")
            if file_info['theoretical_fwhm'] > 0:
                f.write(f"  Theoretical FWHM: {file_info['theoretical_fwhm']:.3f} pixels\n")
            else:
                f.write(f"  Theoretical FWHM: Perfect edge (no blur)\n")
            f.write(f"  Purpose: {file_info['name'].replace('_', ' ').title()}\n")
    
    print(f"\nClean targets summary saved to: {summary_path}")

def main():
    """Generate clean, artifact-free slant-edge targets"""
    
    print("=" * 60)
    print("CLEAN SLANT-EDGE TARGET GENERATOR")
    print("Artifact-free native generation using analytical functions")
    print("=" * 60)
    
    try:
        generated_files = generate_test_suite()
        
        print("\n" + "=" * 60)
        print("CLEAN TARGETS GENERATED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total files generated: {len(generated_files)}")
        print("\nTargets ready for MTF analysis validation!")
        print("\nNext steps:")
        print("1. Test with coordinate-fixed MTF analyzer")
        print("2. Compare against theoretical FWHM values")
        print("3. Validate measurement consistency")
        
        return 0
        
    except Exception as e:
        print(f"Error generating clean targets: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())