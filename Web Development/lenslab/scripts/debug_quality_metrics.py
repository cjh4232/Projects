#!/usr/bin/env python3

import subprocess
import re
import os

def debug_quality_metrics():
    """Debug why all ROIs get similar quality scores"""
    print("QUALITY METRICS DEBUGGING")
    print("=" * 60)
    print("Analyzing why quality scores are too similar...")
    print()
    
    test_file = "test_targets/Slant-Edge-Target_rotated_sigma_1.5_blurred.png"
    executable = "./mtf_analyzer_6_filtered"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
        
    if not os.path.exists(executable):
        print(f"❌ Executable not found: {executable}")
        return
    
    try:
        result = subprocess.run([executable, test_file], 
                              capture_output=True, text=True, cwd=os.getcwd())
        
        # Extract detailed quality information
        fwhm_pattern = r'Measured FWHM from image: ([\d.]+) pixels'
        quality_pattern = r'ROI Quality Score: ([\d.]+)/100'
        status_pattern = r'Quality Status: ([^\\n]+)'
        
        fwhm_matches = re.findall(fwhm_pattern, result.stdout)
        quality_matches = re.findall(quality_pattern, result.stdout)
        status_matches = re.findall(status_pattern, result.stdout)
        
        print("DETAILED ROI COMPARISON:")
        print("-" * 50)
        
        for i in range(len(fwhm_matches)):
            fwhm = float(fwhm_matches[i])
            quality = float(quality_matches[i]) if i < len(quality_matches) else 0
            status = status_matches[i] if i < len(status_matches) else "Unknown"
            
            # Assess FWHM reasonableness (expected: 3.532)
            fwhm_reasonableness = "GOOD" if 2.0 <= fwhm <= 5.0 else "POOR"
            
            print(f"ROI {i+1}:")
            print(f"  FWHM: {fwhm:.3f} pixels ({fwhm_reasonableness})")
            print(f"  Quality Score: {quality:.1f}/100")
            print(f"  Status: {status}")
            print()
        
        # Analysis
        print("ANALYSIS:")
        print("-" * 50)
        
        if quality_matches:
            quality_values = [float(q) for q in quality_matches]
            quality_range = max(quality_values) - min(quality_values)
            
            print(f"Quality score range: {min(quality_values):.1f} - {max(quality_values):.1f}")
            print(f"Quality variation: {quality_range:.1f} points")
            
            if quality_range < 5:
                print("❌ PROBLEM: Quality scores are too similar (variation < 5 points)")
                print("   This suggests the quality metrics aren't discriminating properly.")
            
            # Check if quality correlates with FWHM reasonableness
            fwhm_values = [float(f) for f in fwhm_matches]
            reasonable_rois = [i for i, f in enumerate(fwhm_values) if 2.0 <= f <= 5.0]
            unreasonable_rois = [i for i, f in enumerate(fwhm_values) if f < 2.0 or f > 5.0]
            
            print(f"\nFWHM reasonableness check:")
            print(f"Reasonable ROIs (2-5 pixels): {len(reasonable_rois)} - Indices: {reasonable_rois}")
            print(f"Unreasonable ROIs: {len(unreasonable_rois)} - Indices: {unreasonable_rois}")
            
            if reasonable_rois and unreasonable_rois:
                reasonable_quality = [quality_values[i] for i in reasonable_rois]
                unreasonable_quality = [quality_values[i] for i in unreasonable_rois]
                
                avg_reasonable = sum(reasonable_quality) / len(reasonable_quality)
                avg_unreasonable = sum(unreasonable_quality) / len(unreasonable_quality)
                
                print(f"Average quality - Reasonable ROIs: {avg_reasonable:.1f}")
                print(f"Average quality - Unreasonable ROIs: {avg_unreasonable:.1f}")
                print(f"Quality difference: {abs(avg_reasonable - avg_unreasonable):.1f} points")
                
                if abs(avg_reasonable - avg_unreasonable) < 10:
                    print("❌ PROBLEM: Quality metrics don't correlate with FWHM reasonableness")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 50)
        print("1. Quality metrics need better discrimination")
        print("2. Consider different weight distributions in quality scoring")
        print("3. May need to examine individual quality components:")
        print("   - Edge strength")
        print("   - Linearity score") 
        print("   - Noise level")
        print("   - Profile adequacy")
        print("4. Consider using relative quality ranking instead of absolute thresholds")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_quality_metrics()