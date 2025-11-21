#!/usr/bin/env python3
"""
Simple validation of Gaussian blur application.
Creates a perfect step edge and applies known blur levels.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_perfect_step_edge(width=500, height=500, angle_deg=5):
    """Create a perfect black-to-white slant edge."""
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Create slant edge - similar to MTF test patterns
    center_x = width // 2
    center_y = height // 2
    angle_rad = np.radians(angle_deg)
    
    for y in range(height):
        for x in range(width):
            # Calculate distance from center line
            dx = x - center_x
            dy = y - center_y
            
            # Rotate coordinate system
            rotated_x = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
            
            # Create edge based on rotated x coordinate
            if rotated_x > 0:
                image[y, x] = 255
    
    return image

def apply_blur_and_analyze(image, sigma):
    """Apply blur and analyze the edge profile."""
    # Apply Gaussian blur
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    
    # Extract horizontal profile through the middle
    middle_row = blurred.shape[0] // 2
    profile = blurred[middle_row, :].astype(float) / 255.0
    
    # Find edge center
    edge_center = len(profile) // 2
    
    # Extract profile around edge (±50 pixels)
    start = max(0, edge_center - 50)
    end = min(len(profile), edge_center + 50)
    x = np.arange(start, end) - edge_center
    y = profile[start:end]
    
    # Calculate LSF (derivative of profile)
    lsf = np.gradient(y)
    
    # Find FWHM of LSF
    peak_idx = np.argmax(lsf)
    peak_val = lsf[peak_idx]
    half_max = peak_val / 2
    
    # Find half-maximum crossings
    left_idx = None
    right_idx = None
    
    for i in range(peak_idx):
        if lsf[i] <= half_max and lsf[i+1] > half_max:
            left_idx = i
    
    for i in range(peak_idx, len(lsf)-1):
        if lsf[i] > half_max and lsf[i+1] <= half_max:
            right_idx = i
            break
    
    if left_idx is not None and right_idx is not None:
        fwhm = x[right_idx] - x[left_idx]
    else:
        fwhm = None
    
    return blurred, profile, lsf, fwhm, x, y

def main():
    """Test blur application with known theoretical values."""
    print("Validating Gaussian blur application...")
    
    # Create perfect step edge
    perfect_edge = create_perfect_step_edge()
    
    # Test different sigma values
    sigmas = [0.5, 1.0, 1.5, 2.0]
    
    plt.figure(figsize=(15, 10))
    
    for i, sigma in enumerate(sigmas):
        theoretical_fwhm = 2.355 * sigma
        
        # Apply blur and analyze
        blurred, profile, lsf, measured_fwhm, x, y = apply_blur_and_analyze(perfect_edge, sigma)
        
        # Save blurred image
        cv2.imwrite(f'test_blur_sigma_{sigma}.png', blurred)
        
        # Plot results
        plt.subplot(2, 2, i+1)
        plt.plot(x, lsf, 'b-', label=f'LSF (σ={sigma})')
        plt.axhline(y=np.max(lsf)/2, color='r', linestyle='--', alpha=0.5, label='Half Maximum')
        plt.title(f'σ={sigma}: Theory={theoretical_fwhm:.3f}, Measured={measured_fwhm:.3f}' if measured_fwhm else f'σ={sigma}: Theory={theoretical_fwhm:.3f}, Measured=FAILED')
        plt.xlabel('Distance (pixels)')
        plt.ylabel('LSF')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Print results
        error = abs(measured_fwhm - theoretical_fwhm) / theoretical_fwhm * 100 if measured_fwhm else float('inf')
        print(f"σ={sigma}: Theoretical FWHM={theoretical_fwhm:.3f}, Measured FWHM={measured_fwhm:.3f}, Error={error:.1f}%")
    
    plt.tight_layout()
    plt.savefig('blur_validation_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to blur_validation_results.png")
    print("Test images saved as test_blur_sigma_*.png")

if __name__ == "__main__":
    main()