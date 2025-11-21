#!/usr/bin/env python3

import subprocess
import re
import os
import statistics

def test_consistent_rois():
    """Test ROI consistency improvements"""
    print("CONSISTENT ROI SIZE TEST")
    print("=" * 60)
    
    test_file = "test_targets/Slant-Edge-Target_rotated_sigma_1.5_blurred.png"
    
    # Test configurations
    tests = {
        "Variable ROI Sizes (Original)": "./mtf_analyzer_6_fixed_quality",
        "Consistent ROI Sizes (Fixed)": "./mtf_analyzer_6_consistent_rois"
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
            
            # Extract quality scores
            quality_pattern = r'ROI Quality Score: ([\d.]+)/100'
            quality_matches = re.findall(quality_pattern, result.stdout)
            
            fwhm_values = [float(f) for f in fwhm_matches]
            quality_scores = [float(q) for q in quality_matches]
            
            print(f"ROI Sizes: {roi_size_matches}")
            print(f"FWHM values: {[f'{f:.3f}' for f in fwhm_values]}")
            print(f"Quality scores: {[f'{q:.1f}' for q in quality_scores]}")
            
            if fwhm_values and len(fwhm_values) > 1:
                mean_fwhm = statistics.mean(fwhm_values)
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
                    'mean_fwhm': mean_fwhm,
                    'cv': cv,
                    'error': error,
                    'consistency': consistency,
                    'roi_sizes': roi_size_matches
                }
            else:
                print("❌ Insufficient FWHM measurements")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Comparison
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("IMPROVEMENT ANALYSIS")
        print("-" * 60)
        
        names = list(results.keys())
        original = results[names[0]]
        improved = results[names[1]]
        
        print(f"ROI Sizes:")
        print(f"  Original: {original['roi_sizes']}")
        print(f"  Improved: {improved['roi_sizes']}")
        
        print(f"Consistency (CV): {original['cv']:.1f}% → {improved['cv']:.1f}%")
        cv_improvement = original['cv'] - improved['cv']
        
        print(f"Accuracy (Error): {original['error']:.1f}% → {improved['error']:.1f}%")
        error_change = improved['error'] - original['error']
        
        print(f"Mean FWHM: {original['mean_fwhm']:.3f} → {improved['mean_fwhm']:.3f}")
        
        # Assessment
        print(f"\nIMPROVEMENT ASSESSMENT:")
        if cv_improvement > 10:
            print(f"✅ MAJOR consistency improvement: {cv_improvement:.1f} percentage points")
        elif cv_improvement > 0:
            print(f"✅ Consistency improvement: {cv_improvement:.1f} percentage points")
        else:
            print(f"⚠️ No significant consistency improvement")
        
        if abs(error_change) < 5:
            print(f"✅ Accuracy maintained (change: {error_change:+.1f}%)")
        elif error_change < 0:
            print(f"✅ Accuracy improved by {-error_change:.1f}%")
        else:
            print(f"⚠️ Accuracy decreased by {error_change:.1f}%")
        
        # Overall assessment
        if cv_improvement > 10 and abs(error_change) < 10:
            print(f"🎉 EXCELLENT: Consistent ROI sizes significantly improved measurements!")
        elif cv_improvement > 5:
            print(f"✅ GOOD: Meaningful improvement achieved")
        else:
            print(f"⚠️ MIXED: Some improvement but more work needed")

if __name__ == "__main__":
    test_consistent_rois()