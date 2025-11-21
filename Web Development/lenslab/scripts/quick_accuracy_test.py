#!/usr/bin/env python3
"""
Quick test to check if the corrected hybrid system improved FWHM accuracy
"""

import subprocess
import re

def test_analyzer(analyzer, image_path, sigma, theoretical_fwhm):
    """Test a single analyzer and return FWHM accuracy"""
    cmd = [analyzer, image_path, "--gaussian-sigma", str(sigma), "--debug"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None, "Analysis failed"
        
        # Count ROIs
        roi_match = re.search(r'Found (\d+) edges at target angles', result.stdout)
        rois = int(roi_match.group(1)) if roi_match else 0
        
        # Extract all FWHM values
        fwhm_pattern = r'Final LSF Statistics:\s*FWHM:\s*([0-9]+\.?[0-9]*)'
        fwhm_values = [float(m) for m in re.findall(fwhm_pattern, result.stdout)]
        
        if not fwhm_values:
            return None, "No FWHM values found"
        
        # Calculate average FWHM and error
        avg_fwhm = sum(fwhm_values) / len(fwhm_values)
        error_percent = abs((avg_fwhm - theoretical_fwhm) / theoretical_fwhm) * 100
        
        return {
            "rois": rois,
            "fwhm_values": fwhm_values,
            "avg_fwhm": avg_fwhm,
            "error_percent": error_percent,
            "passed": error_percent < 5.0
        }, None
        
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)

def main():
    test_cases = [
        {"image": "test_targets/Slant-Edge-Target_rotated_sigma_1.5_blurred.png", "sigma": 1.5, "theoretical": 3.532},
        {"image": "test_targets/Slant-Edge-Target_rotated_sigma_2.0_blurred.png", "sigma": 2.0, "theoretical": 4.710},
    ]
    
    analyzers = [
        ("Corrected Hybrid", "./mtf_analyzer_6_hybrid_fixed"),
        ("Previous Hybrid", "./mtf_analyzer_6_hybrid"),
    ]
    
    print("QUICK ACCURACY COMPARISON")
    print("=" * 50)
    
    for test_case in test_cases:
        print(f"\nTest: σ={test_case['sigma']}, Expected FWHM={test_case['theoretical']:.3f}")
        print("-" * 40)
        
        for name, analyzer in analyzers:
            result, error = test_analyzer(analyzer, test_case["image"], test_case["sigma"], test_case["theoretical"])
            
            if result:
                print(f"{name}:")
                print(f"  ROIs: {result['rois']}")
                print(f"  Avg FWHM: {result['avg_fwhm']:.3f}")
                print(f"  Error: {result['error_percent']:.1f}%")
                print(f"  Status: {'✓ PASS' if result['passed'] else '✗ FAIL'}")
            else:
                print(f"{name}: ✗ {error}")
    
    print("\n" + "=" * 50)
    print("Key improvement: Corrected sampling interval scaling")

if __name__ == "__main__":
    main()