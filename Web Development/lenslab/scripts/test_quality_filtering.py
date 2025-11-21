#!/usr/bin/env python3

import subprocess
import re
import os
import statistics

def test_quality_filtering():
    """Test ROI quality filtering effectiveness"""
    print("ROI QUALITY FILTERING TEST")
    print("=" * 60)
    
    test_file = "test_targets/Slant-Edge-Target_rotated_sigma_1.5_blurred.png"
    
    # Test configurations
    tests = {
        "Without Quality Filtering": "./mtf_analyzer_6_real_fwhm",
        "With Quality Filtering": "./mtf_analyzer_6_filtered"
    }
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    results = {}
    
    for name, executable in tests.items():
        if not os.path.exists(executable):
            print(f"❌ {name}: Executable not found")
            continue
        
        print(f"\n{name}:")
        print("-" * 50)
        
        try:
            result = subprocess.run([executable, test_file], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            # Extract FWHM values
            fwhm_pattern = r'Measured FWHM from image: ([\d.]+) pixels'
            fwhm_matches = re.findall(fwhm_pattern, result.stdout)
            
            # Extract quality scores
            quality_pattern = r'ROI Quality Score: ([\d.]+)/100'
            quality_matches = re.findall(quality_pattern, result.stdout)
            
            # Extract skipped ROIs
            skipped_count = len(re.findall(r'Skipping ROI due to poor quality', result.stdout))
            
            fwhm_values = [float(f) for f in fwhm_matches]
            quality_scores = [float(q) for q in quality_matches]
            
            print(f"Total ROIs processed: {len(fwhm_values)}")
            print(f"ROIs skipped due to quality: {skipped_count}")
            print(f"Quality scores: {[f'{q:.1f}' for q in quality_scores]}")
            
            if fwhm_values:
                print(f"FWHM values: {[f'{f:.3f}' for f in fwhm_values]}")
                
                # Calculate statistics
                mean_fwhm = statistics.mean(fwhm_values)
                if len(fwhm_values) > 1:
                    stdev_fwhm = statistics.stdev(fwhm_values)
                    cv = (stdev_fwhm / mean_fwhm) * 100
                    
                    print(f"Mean FWHM: {mean_fwhm:.3f} pixels")
                    print(f"Std Dev: {stdev_fwhm:.3f} pixels")
                    print(f"Coefficient of Variation: {cv:.1f}%")
                    
                    # Quality assessment
                    if cv < 10:
                        consistency = "✅ EXCELLENT"
                    elif cv < 25:
                        consistency = "✅ GOOD"
                    elif cv < 50:
                        consistency = "⚠️ FAIR"
                    else:
                        consistency = "❌ POOR"
                    
                    print(f"Consistency: {consistency}")
                    
                    # Expected value comparison (σ=1.5 → FWHM=3.532)
                    expected_fwhm = 3.532
                    error = abs(mean_fwhm - expected_fwhm) / expected_fwhm * 100
                    print(f"Error vs expected (3.532): {error:.1f}%")
                    
                    results[name] = {
                        'roi_count': len(fwhm_values),
                        'skipped_count': skipped_count,
                        'mean_fwhm': mean_fwhm,
                        'cv': cv,
                        'error': error,
                        'consistency': consistency
                    }
                else:
                    single_error = abs(fwhm_values[0] - 3.532) / 3.532 * 100
                    print(f"Single ROI error vs expected: {single_error:.1f}%")
                    
                    results[name] = {
                        'roi_count': 1,
                        'skipped_count': skipped_count,
                        'mean_fwhm': fwhm_values[0],
                        'cv': 0,
                        'error': single_error,
                        'consistency': "N/A (single ROI)"
                    }
            else:
                print("❌ No FWHM measurements obtained")
                results[name] = {
                    'roi_count': 0,
                    'skipped_count': skipped_count,
                    'mean_fwhm': 0,
                    'cv': 0,
                    'error': 100,
                    'consistency': "❌ NO DATA"
                }
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Comparison
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("-" * 60)
        
        names = list(results.keys())
        before = results[names[0]]
        after = results[names[1]]
        
        print(f"ROI Count: {before['roi_count']} → {after['roi_count']}")
        print(f"Skipped ROIs: {before['skipped_count']} → {after['skipped_count']}")
        print(f"Mean FWHM: {before['mean_fwhm']:.3f} → {after['mean_fwhm']:.3f}")
        print(f"Consistency (CV): {before['cv']:.1f}% → {after['cv']:.1f}%")
        print(f"Accuracy (Error): {before['error']:.1f}% → {after['error']:.1f}%")
        
        # Improvements
        if after['cv'] < before['cv']:
            cv_improvement = before['cv'] - after['cv']
            print(f"✅ Consistency improved by {cv_improvement:.1f} percentage points")
        
        if after['error'] < before['error']:
            error_improvement = before['error'] - after['error']
            print(f"✅ Accuracy improved by {error_improvement:.1f} percentage points")
        
        print(f"\nOverall Assessment:")
        print(f"Quality filtering {'✅ IMPROVED' if after['cv'] < before['cv'] or after['error'] < before['error'] else '⚠️ MIXED RESULTS'} the measurements")

if __name__ == "__main__":
    test_quality_filtering()