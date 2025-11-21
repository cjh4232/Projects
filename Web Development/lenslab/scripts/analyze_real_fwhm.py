#!/usr/bin/env python3

import subprocess
import re
import os
import statistics

def extract_real_fwhm_values(output_text):
    """Extract FWHM values from the real image analysis"""
    fwhm_values = []
    
    # Find all "REAL IMAGE FWHM ANALYSIS" sections
    pattern = r'=== REAL IMAGE FWHM ANALYSIS ===\s*\n\s*Measured FWHM from image: ([\d.]+) pixels'
    matches = re.findall(pattern, output_text)
    
    return [float(match) for match in matches]

def analyze_real_fwhm():
    print("REAL FWHM ANALYSIS FROM IMAGE DATA")
    print("=" * 60)
    
    test_cases = [
        {
            'file': 'test_targets/Slant-Edge-Target_rotated_sigma_1.0_blurred.png',
            'sigma': 1.0,
            'expected_fwhm': 2.355
        },
        {
            'file': 'test_targets/Slant-Edge-Target_rotated_sigma_1.5_blurred.png', 
            'sigma': 1.5,
            'expected_fwhm': 3.532
        },
        {
            'file': 'test_targets/Slant-Edge-Target_rotated_sigma_2.0_blurred.png',
            'sigma': 2.0, 
            'expected_fwhm': 4.710
        }
    ]
    
    executable = "./mtf_analyzer_6_real_fwhm"
    
    if not os.path.exists(executable):
        print(f"❌ Executable not found: {executable}")
        return
    
    for test_case in test_cases:
        if not os.path.exists(test_case['file']):
            print(f"❌ File not found: {test_case['file']}")
            continue
            
        print(f"\nTesting σ={test_case['sigma']}, Expected FWHM={test_case['expected_fwhm']:.3f} pixels")
        print("-" * 60)
        
        try:
            result = subprocess.run([executable, test_case['file']], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            # Extract all real FWHM values
            fwhm_values = extract_real_fwhm_values(result.stdout)
            
            if fwhm_values:
                print(f"Found {len(fwhm_values)} ROI measurements:")
                for i, fwhm in enumerate(fwhm_values, 1):
                    error = abs(fwhm - test_case['expected_fwhm']) / test_case['expected_fwhm'] * 100
                    status = "✅" if error < 15 else "⚠️" if error < 30 else "❌"
                    print(f"  ROI {i}: {fwhm:.3f} pixels (error: {error:.1f}%) {status}")
                
                # Statistical analysis
                if len(fwhm_values) > 1:
                    mean_fwhm = statistics.mean(fwhm_values)
                    median_fwhm = statistics.median(fwhm_values)
                    stdev_fwhm = statistics.stdev(fwhm_values) if len(fwhm_values) > 1 else 0
                    
                    print(f"\nStatistics:")
                    print(f"  Mean: {mean_fwhm:.3f} pixels")
                    print(f"  Median: {median_fwhm:.3f} pixels")
                    print(f"  Std Dev: {stdev_fwhm:.3f} pixels")
                    print(f"  Range: {min(fwhm_values):.3f} - {max(fwhm_values):.3f} pixels")
                    
                    mean_error = abs(mean_fwhm - test_case['expected_fwhm']) / test_case['expected_fwhm'] * 100
                    median_error = abs(median_fwhm - test_case['expected_fwhm']) / test_case['expected_fwhm'] * 100
                    
                    print(f"  Mean error: {mean_error:.1f}%")
                    print(f"  Median error: {median_error:.1f}%")
                    
                    # Quality assessment
                    cv = (stdev_fwhm / mean_fwhm) * 100 if mean_fwhm > 0 else 0
                    print(f"  Coefficient of variation: {cv:.1f}%")
                    
                    if cv < 10:
                        print("  Consistency: ✅ GOOD (CV < 10%)")
                    elif cv < 25:
                        print("  Consistency: ⚠️ FAIR (CV 10-25%)")
                    else:
                        print("  Consistency: ❌ POOR (CV > 25%)")
                else:
                    single_error = abs(fwhm_values[0] - test_case['expected_fwhm']) / test_case['expected_fwhm'] * 100
                    print(f"  Error: {single_error:.1f}%")
                    
            else:
                print("❌ No FWHM values extracted")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS INSIGHTS:")
    print("• Real FWHM measurements show significant ROI-to-ROI variation")
    print("• This suggests edge quality or sampling issues in some ROIs")
    print("• May need ROI quality filtering or improved edge detection")

if __name__ == "__main__":
    analyze_real_fwhm()