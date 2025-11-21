#!/usr/bin/env python3

import subprocess
import re
import os
import statistics

def test_polarity_fix():
    """Test edge polarity normalization improvements"""
    print("EDGE POLARITY NORMALIZATION TEST")
    print("=" * 60)
    
    test_file = "test_targets/Slant-Edge-Target_rotated_sigma_1.5_blurred.png"
    
    # Test configurations
    tests = {
        "Before Polarity Fix": "./mtf_analyzer_6_fixed_quality",
        "After Polarity Fix": "./mtf_analyzer_6_polarity_fixed"
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
            
            # Extract ROI sizes
            roi_size_pattern = r'ROI size: \[(\d+) x (\d+)\]'
            roi_size_matches = re.findall(roi_size_pattern, result.stdout)
            
            # Extract polarity information
            polarity_flipped = len(re.findall(r'flipped bright→dark to dark→bright', result.stdout))
            polarity_no_flip = len(re.findall(r'already dark→bright \(no flip needed\)', result.stdout))
            
            fwhm_values = [float(f) for f in fwhm_matches]
            
            print(f"ROI Sizes: {roi_size_matches}")
            print(f"FWHM values: {[f'{f:.3f}' for f in fwhm_values]}")
            
            if polarity_flipped + polarity_no_flip > 0:
                print(f"Edge Polarity: {polarity_flipped} flipped, {polarity_no_flip} unchanged")
            
            if fwhm_values and len(fwhm_values) > 1:
                mean_fwhm = statistics.mean(fwhm_values)
                stdev_fwhm = statistics.stdev(fwhm_values)
                cv = (stdev_fwhm / mean_fwhm) * 100
                min_fwhm = min(fwhm_values)
                max_fwhm = max(fwhm_values)
                
                print(f"Mean FWHM: {mean_fwhm:.3f} pixels")
                print(f"Std Dev: {stdev_fwhm:.3f} pixels")
                print(f"Range: {min_fwhm:.3f} - {max_fwhm:.3f} pixels")
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
                    'mean_fwhm': mean_fwhm,
                    'cv': cv,
                    'error': error,
                    'consistency': consistency,
                    'roi_sizes': roi_size_matches,
                    'range': max_fwhm - min_fwhm,
                    'polarity_info': (polarity_flipped, polarity_no_flip)
                }
            else:
                print("❌ Insufficient FWHM measurements")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Comparison
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("POLARITY NORMALIZATION IMPACT")
        print("-" * 60)
        
        names = list(results.keys())
        before = results[names[0]]
        after = results[names[1]]
        
        print(f"Consistency (CV): {before['cv']:.1f}% → {after['cv']:.1f}%")
        cv_improvement = before['cv'] - after['cv']
        
        print(f"Accuracy (Error): {before['error']:.1f}% → {after['error']:.1f}%")
        error_change = after['error'] - before['error']
        
        print(f"FWHM Range: {before['range']:.3f} → {after['range']:.3f} pixels")
        range_improvement = before['range'] - after['range']
        
        print(f"Mean FWHM: {before['mean_fwhm']:.3f} → {after['mean_fwhm']:.3f}")
        
        if 'polarity_info' in after:
            flipped, unchanged = after['polarity_info']
            print(f"Edge Polarity Normalization: {flipped} ROIs flipped, {unchanged} unchanged")
        
        # Assessment
        print(f"\nIMPROVEMENT ASSESSMENT:")
        
        if cv_improvement > 20:
            print(f"🎉 MAJOR consistency improvement: {cv_improvement:.1f} percentage points")
        elif cv_improvement > 10:
            print(f"✅ SIGNIFICANT consistency improvement: {cv_improvement:.1f} percentage points")
        elif cv_improvement > 0:
            print(f"✅ Consistency improvement: {cv_improvement:.1f} percentage points")
        else:
            print(f"⚠️ No consistency improvement (change: {cv_improvement:+.1f}%)")
        
        if range_improvement > 1.0:
            print(f"✅ FWHM range significantly reduced by {range_improvement:.3f} pixels")
        elif range_improvement > 0:
            print(f"✅ FWHM range reduced by {range_improvement:.3f} pixels")
        
        # Overall assessment
        if cv_improvement > 20 and after['cv'] < 25:
            print(f"🎉 EXCELLENT: Edge polarity fix achieved target consistency!")
        elif cv_improvement > 10:
            print(f"✅ GOOD: Meaningful improvement achieved")
        elif cv_improvement > 0:
            print(f"⚠️ MODEST: Some improvement, may need additional fixes")
        else:
            print(f"❌ INEFFECTIVE: Polarity fix didn't improve consistency")

if __name__ == "__main__":
    test_polarity_fix()