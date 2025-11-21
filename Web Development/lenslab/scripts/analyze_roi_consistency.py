#!/usr/bin/env python3

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_roi_image(image_path, roi_name):
    """Analyze a single ROI image for edge characteristics"""
    if not os.path.exists(image_path):
        return None
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Basic statistics
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    min_intensity = np.min(img)
    max_intensity = np.max(img)
    
    # Edge detection using Sobel
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Edge strength metrics
    max_gradient = np.max(gradient_magnitude)
    mean_gradient = np.mean(gradient_magnitude)
    edge_pixels = np.sum(gradient_magnitude > 50)  # Count strong edge pixels
    
    # Find the steepest transition (should represent the main edge)
    max_grad_pos = np.unravel_index(np.argmax(gradient_magnitude), gradient_magnitude.shape)
    
    # Analyze edge profile along the steepest direction
    if grad_x[max_grad_pos] != 0 or grad_y[max_grad_pos] != 0:
        # Get edge direction (perpendicular to gradient)
        edge_angle = np.arctan2(-grad_x[max_grad_pos], grad_y[max_grad_pos])
        
        # Sample profile perpendicular to edge
        center_y, center_x = max_grad_pos
        profile_length = min(img.shape) // 2
        
        cos_angle = np.cos(edge_angle)
        sin_angle = np.sin(edge_angle)
        
        profile = []
        profile_positions = []
        
        for i in range(-profile_length, profile_length):
            x = int(center_x + i * cos_angle)
            y = int(center_y + i * sin_angle)
            
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                profile.append(img[y, x])
                profile_positions.append(i)
        
        profile = np.array(profile)
        
        # Calculate edge sharpness (rise distance)
        if len(profile) > 10:
            # Find 10% and 90% intensity points
            intensity_range = np.max(profile) - np.min(profile)
            low_thresh = np.min(profile) + 0.1 * intensity_range
            high_thresh = np.min(profile) + 0.9 * intensity_range
            
            # Find positions where profile crosses these thresholds
            low_crossings = np.where(np.diff(np.signbit(profile - low_thresh)))[0]
            high_crossings = np.where(np.diff(np.signbit(profile - high_thresh)))[0]
            
            edge_sharpness = None
            if len(low_crossings) > 0 and len(high_crossings) > 0:
                # Distance between 10% and 90% points (smaller = sharper)
                edge_sharpness = abs(high_crossings[0] - low_crossings[0])
        else:
            edge_sharpness = None
            profile = []
    else:
        edge_sharpness = None
        profile = []
    
    return {
        'name': roi_name,
        'size': img.shape,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'intensity_range': max_intensity - min_intensity,
        'max_gradient': max_gradient,
        'mean_gradient': mean_gradient,
        'edge_pixels': edge_pixels,
        'edge_sharpness': edge_sharpness,
        'profile': profile,
        'image': img
    }

def analyze_all_rois():
    """Analyze all ROI debug images"""
    print("COMPREHENSIVE ROI ANALYSIS")
    print("=" * 60)
    print("Analyzing edge characteristics in each ROI debug image...")
    print()
    
    roi_files = [
        'debug_roi_0.png',
        'debug_roi_1.png', 
        'debug_roi_2.png',
        'debug_roi_3.png'
    ]
    
    results = []
    
    for i, roi_file in enumerate(roi_files):
        result = analyze_roi_image(roi_file, f"ROI {i}")
        if result is not None:
            results.append(result)
        else:
            print(f"❌ Could not analyze {roi_file}")
    
    if not results:
        print("❌ No ROI images found to analyze")
        return
    
    print("INDIVIDUAL ROI ANALYSIS:")
    print("-" * 50)
    
    for result in results:
        print(f"{result['name']}:")
        print(f"  Size: {result['size']}")
        print(f"  Intensity - Mean: {result['mean_intensity']:.1f}, Range: {result['intensity_range']:.1f}")
        print(f"  Edge Metrics - Max Gradient: {result['max_gradient']:.1f}, Strong Edge Pixels: {result['edge_pixels']}")
        if result['edge_sharpness'] is not None:
            print(f"  Edge Sharpness (10-90% rise): {result['edge_sharpness']:.1f} pixels (lower=sharper)")
        else:
            print(f"  Edge Sharpness: Could not calculate")
        print()
    
    # Comparative analysis
    print("COMPARATIVE ANALYSIS:")
    print("-" * 50)
    
    # Compare edge sharpness
    sharpness_values = [r['edge_sharpness'] for r in results if r['edge_sharpness'] is not None]
    if len(sharpness_values) > 1:
        sharpness_range = max(sharpness_values) - min(sharpness_values)
        sharpness_cv = (np.std(sharpness_values) / np.mean(sharpness_values)) * 100
        
        print(f"Edge Sharpness Consistency:")
        print(f"  Values: {[f'{s:.1f}' for s in sharpness_values]} pixels")
        print(f"  Range: {sharpness_range:.1f} pixels")
        print(f"  Coefficient of Variation: {sharpness_cv:.1f}%")
        
        if sharpness_cv < 20:
            print(f"  Assessment: ✅ CONSISTENT edge sharpness")
        elif sharpness_cv < 50:
            print(f"  Assessment: ⚠️ MODERATE variation in edge sharpness")
        else:
            print(f"  Assessment: ❌ HIGH variation in edge sharpness - confirms different blur levels")
        print()
    
    # Compare gradient strengths
    max_gradients = [r['max_gradient'] for r in results]
    grad_range = max(max_gradients) - min(max_gradients)
    grad_cv = (np.std(max_gradients) / np.mean(max_gradients)) * 100
    
    print(f"Gradient Strength Consistency:")
    print(f"  Max Gradients: {[f'{g:.1f}' for g in max_gradients]}")
    print(f"  Range: {grad_range:.1f}")
    print(f"  Coefficient of Variation: {grad_cv:.1f}%")
    
    if grad_cv < 20:
        print(f"  Assessment: ✅ CONSISTENT gradient strength")
    elif grad_cv < 50:
        print(f"  Assessment: ⚠️ MODERATE variation in gradient strength")
    else:
        print(f"  Assessment: ❌ HIGH variation in gradient strength")
    print()
    
    # Compare intensity ranges
    intensity_ranges = [r['intensity_range'] for r in results]
    range_cv = (np.std(intensity_ranges) / np.mean(intensity_ranges)) * 100
    
    print(f"Intensity Range Consistency:")
    print(f"  Ranges: {[f'{ir:.1f}' for ir in intensity_ranges]}")
    print(f"  Coefficient of Variation: {range_cv:.1f}%")
    
    if range_cv < 10:
        print(f"  Assessment: ✅ CONSISTENT intensity ranges")
    else:
        print(f"  Assessment: ⚠️ Variation in intensity ranges")
    print()
    
    # Identify potential issues
    print("ISSUE IDENTIFICATION:")
    print("-" * 50)
    
    # Find the outlier ROI
    if len(sharpness_values) > 1:
        mean_sharpness = np.mean(sharpness_values)
        for i, result in enumerate(results):
            if result['edge_sharpness'] is not None:
                deviation = abs(result['edge_sharpness'] - mean_sharpness)
                if deviation > mean_sharpness * 0.5:  # More than 50% deviation
                    print(f"⚠️ {result['name']} appears to be an outlier:")
                    print(f"  Edge sharpness: {result['edge_sharpness']:.1f} vs mean {mean_sharpness:.1f}")
                    print(f"  This ROI may be capturing a different edge feature")
    
    # Check for ROI size inconsistencies
    sizes = [r['size'] for r in results]
    if len(set(sizes)) > 1:
        print(f"⚠️ ROI sizes are inconsistent: {sizes}")
        print(f"  This suggests ROI generation issues")
    
    print(f"\nCONCLUSIONS:")
    print("-" * 50)
    print(f"• If edge sharpness varies significantly (CV > 50%), the blur is genuinely different")
    print(f"• If edge sharpness is consistent but FWHM varies, the issue is in our LSF processing")
    print(f"• ROI size inconsistencies indicate ROI generation problems")
    print(f"• You're right to question uniform blur - synthetic images should be uniform")

if __name__ == "__main__":
    analyze_all_rois()