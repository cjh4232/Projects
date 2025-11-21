#!/usr/bin/env python3

import subprocess
import re
import os

def analyze_roi_patterns():
    """Analyze ROI quality patterns to understand why consistency is poor"""
    print("ROI QUALITY PATTERN ANALYSIS")
    print("=" * 60)
    print("Analyzing why some ROIs give good FWHM and others don't...")
    print()
    
    test_file = "test_targets/Slant-Edge-Target_rotated_sigma_1.5_blurred.png"
    executable = "./mtf_analyzer_6_real_fwhm"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
        
    if not os.path.exists(executable):
        print(f"❌ Executable not found: {executable}")
        return
    
    try:
        result = subprocess.run([executable, test_file], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        # Extract ROI information
        print("DETAILED ROI ANALYSIS:")
        print("-" * 40)
        
        # Look for angle information
        angle_pattern = r'Found edge with angle: ([-\d.]+), raw vector: \((\d+), ([-\d]+)\)'
        angle_matches = re.findall(angle_pattern, result.stdout)
        
        # Look for FWHM values
        fwhm_pattern = r'=== REAL IMAGE FWHM ANALYSIS ===\s*\n\s*Measured FWHM from image: ([\d.]+) pixels'
        fwhm_matches = re.findall(fwhm_pattern, result.stdout)
        
        # Look for profile collection info
        profile_pattern = r'Total profiles collected: (\d+)'
        profile_matches = re.findall(profile_pattern, result.stdout)
        
        print(f"Found {len(angle_matches)} edge detections:")
        for i, (angle, x, y) in enumerate(angle_matches):
            print(f"  Edge {i+1}: Angle {angle}°, Vector ({x}, {y})")
        
        print(f"\nFound {len(fwhm_matches)} FWHM measurements:")
        for i, fwhm in enumerate(fwhm_matches):
            quality = "GOOD" if 2.0 <= float(fwhm) <= 5.0 else "POOR"
            print(f"  ROI {i+1}: {fwhm} pixels ({quality})")
        
        print(f"\nFound {len(profile_matches)} profile collections:")
        for i, profiles in enumerate(profile_matches):
            adequacy = "ADEQUATE" if int(profiles) >= 20 else "SPARSE"
            print(f"  ROI {i+1}: {profiles} profiles ({adequacy})")
        
        # Analysis
        print("\nPATTERN ANALYSIS:")
        print("-" * 40)
        
        if len(fwhm_matches) >= 4:
            fwhm_values = [float(f) for f in fwhm_matches]
            good_rois = [i for i, f in enumerate(fwhm_values) if 2.0 <= f <= 5.0]
            poor_rois = [i for i, f in enumerate(fwhm_values) if f < 2.0 or f > 5.0]
            
            print(f"Good ROIs (2-5 pixels): {len(good_rois)}/4 - Indices: {good_rois}")
            print(f"Poor ROIs (<2 or >5 pixels): {len(poor_rois)}/4 - Indices: {poor_rois}")
            
            if len(angle_matches) >= 4:
                print("\nROI Quality vs Angle Correlation:")
                for i in range(min(4, len(fwhm_values), len(angle_matches))):
                    angle = float(angle_matches[i][0])
                    fwhm = fwhm_values[i]
                    quality = "GOOD" if 2.0 <= fwhm <= 5.0 else "POOR"
                    print(f"  ROI {i+1}: Angle {angle:6.1f}° → FWHM {fwhm:5.3f} ({quality})")
        
        # Look for specific issues
        print("\nPOTENTIAL ISSUES IDENTIFIED:")
        print("-" * 40)
        
        if "Error:" in result.stdout:
            print("❌ Processing errors detected")
            
        # Check for angle clustering
        if len(angle_matches) >= 4:
            angles = [abs(float(a[0])) for a in angle_matches]
            horizontal_angles = [a for a in angles if a < 45 or a > 315]
            vertical_angles = [a for a in angles if 45 <= a <= 135 or 225 <= a <= 315]
            
            print(f"• Horizontal-ish edges: {len(horizontal_angles)}")
            print(f"• Vertical-ish edges: {len(vertical_angles)}")
            
            if len(horizontal_angles) != len(vertical_angles):
                print("⚠️ Unbalanced edge orientation distribution")
        
        # Check for very small FWHM values (sampling issues)
        if fwhm_matches:
            tiny_fwhm = [f for f in fwhm_matches if float(f) < 1.0]
            if tiny_fwhm:
                print(f"⚠️ {len(tiny_fwhm)} ROIs have suspiciously small FWHM (<1 pixel)")
                print("   This suggests sub-pixel sampling or edge detection issues")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        print("1. Implement ROI quality scoring based on:")
        print("   - Edge strength/contrast")
        print("   - Line straightness/linearity") 
        print("   - Noise level assessment")
        print("   - Profile count adequacy")
        print("2. Filter out poor-quality ROIs before FWHM calculation")
        print("3. Enhance edge detection robustness")
        print("4. Add sub-pixel interpolation validation")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    analyze_roi_patterns()