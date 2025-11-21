#!/usr/bin/env python3

import subprocess
import re
import os

def extract_real_fwhm_values(output_text):
    """Extract FWHM values from the binned LSF results, not the synthetic test"""
    # Look for results after edge processing
    fwhm_values = []
    
    # Search for FWHM values in the actual processing output
    # These appear after profile sampling and before the synthetic test
    lines = output_text.split('\n')
    in_real_analysis = False
    
    for i, line in enumerate(lines):
        if "Starting profile sampling" in line:
            in_real_analysis = True
        elif "TESTING FWHM CALCULATION" in line:
            in_real_analysis = False
        elif in_real_analysis and "FWHM" in line and ":" in line:
            # Extract numeric FWHM values from real analysis
            match = re.search(r'FWHM[:\s]*([0-9.]+)', line)
            if match:
                fwhm_values.append(float(match.group(1)))
    
    return fwhm_values

def run_comparison_test():
    test_file = "test_targets/Slant-Edge-Target_rotated_sigma_1.5_blurred.png"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    print("COMPARING SUB-PIXEL DISTANCE FIX")
    print("=" * 50)
    
    executables = {
        "Previous (with sampling_interval)": "./mtf_analyzer_6_hybrid_fixed",
        "New (sub-pixel distances)": "./mtf_analyzer_6_super_res"
    }
    
    for name, exe in executables.items():
        if not os.path.exists(exe):
            print(f"❌ {name}: Executable not found")
            continue
            
        print(f"\n{name}:")
        print("-" * 40)
        
        try:
            result = subprocess.run([exe, test_file], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            # Check ROI detection
            roi_count = len(re.findall(r'Total profiles collected: (\d+)', result.stdout))
            print(f"ROIs detected: {roi_count}")
            
            # Extract synthetic test FWHM (for consistency check)
            synthetic_match = re.search(r'Measured FWHM: ([\d.]+)', result.stdout)
            if synthetic_match:
                synthetic_fwhm = float(synthetic_match.group(1))
                print(f"Synthetic test FWHM: {synthetic_fwhm}")
            
            # Look for any processing improvements
            if "Error:" in result.stdout:
                print("❌ Processing error occurred")
            elif "Analysis complete" in result.stdout:
                print("✅ Analysis completed successfully")
            else:
                print("⚠️ Partial results")
                
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
    
    print("\n" + "=" * 50)
    print("Key improvement: Removed sampling_interval multiplication")
    print("that was causing distance scaling issues.")

if __name__ == "__main__":
    run_comparison_test()