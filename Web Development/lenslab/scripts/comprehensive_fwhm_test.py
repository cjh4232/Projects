#!/usr/bin/env python3

import subprocess
import re
import os

def run_comprehensive_test():
    print("COMPREHENSIVE FWHM ACCURACY TEST")
    print("=" * 60)
    print("Testing sub-pixel resolution improvements")
    print()
    
    # Test cases with known Gaussian blur
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
        },
        {
            'file': 'test_targets/Slant-Edge-Target_rotated_sigma_2.5_blurred.png',
            'sigma': 2.5, 
            'expected_fwhm': 5.888
        }
    ]
    
    executable = "./mtf_analyzer_6_sub_pixel"
    
    if not os.path.exists(executable):
        print(f"❌ Executable not found: {executable}")
        return
    
    results = []
    
    for test_case in test_cases:
        if not os.path.exists(test_case['file']):
            print(f"❌ File not found: {test_case['file']}")
            continue
            
        print(f"Testing σ={test_case['sigma']}, Expected FWHM={test_case['expected_fwhm']:.3f}")
        print("-" * 50)
        
        try:
            result = subprocess.run([executable, test_case['file']], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            # Extract synthetic test FWHM
            fwhm_match = re.search(r'Measured FWHM: ([\d.]+)', result.stdout)
            if fwhm_match:
                measured_fwhm = float(fwhm_match.group(1))
                error_percent = abs(measured_fwhm - test_case['expected_fwhm']) / test_case['expected_fwhm'] * 100
                
                status = "✅ EXCELLENT" if error_percent < 5 else ("✅ GOOD" if error_percent < 10 else ("⚠️ ACCEPTABLE" if error_percent < 15 else "❌ POOR"))
                
                print(f"Measured FWHM: {measured_fwhm:.3f}")
                print(f"Error: {error_percent:.1f}%")
                print(f"Status: {status}")
                
                results.append({
                    'sigma': test_case['sigma'],
                    'expected': test_case['expected_fwhm'],
                    'measured': measured_fwhm,
                    'error': error_percent
                })
            else:
                print("❌ Could not extract FWHM value")
                
            # Check ROI detection
            roi_count = len(re.findall(r'Total profiles collected: (\d+)', result.stdout))
            print(f"ROIs detected: {roi_count}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    # Summary
    if results:
        print("SUMMARY")
        print("=" * 60)
        avg_error = sum(r['error'] for r in results) / len(results)
        min_error = min(r['error'] for r in results)
        max_error = max(r['error'] for r in results)
        
        print(f"Tests completed: {len(results)}")
        print(f"Average error: {avg_error:.1f}%")
        print(f"Min error: {min_error:.1f}%") 
        print(f"Max error: {max_error:.1f}%")
        
        excellent_count = sum(1 for r in results if r['error'] < 5)
        good_count = sum(1 for r in results if 5 <= r['error'] < 10)
        
        print(f"Excellent (<5% error): {excellent_count}/{len(results)}")
        print(f"Good (5-10% error): {good_count}/{len(results)}")
        
        if avg_error < 5:
            print("🎉 EXCELLENT: Average error under 5% target!")
        elif avg_error < 10:
            print("✅ GOOD: Average error under 10%")
        else:
            print("⚠️ NEEDS IMPROVEMENT: Average error above 10%")
        
        print("\nKey improvements achieved:")
        print("• Implemented sub-pixel distance scaling")
        print("• Removed erroneous sampling_interval multiplication")
        print("• Added 4x super-resolution as per ISO 12233 standard")

if __name__ == "__main__":
    run_comprehensive_test()