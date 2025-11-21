#!/usr/bin/env python3

import subprocess
import re
import os
import statistics

def test_outlier_filtering():
    """Test outlier filtering effectiveness"""
    print("OUTLIER FILTERING TEST")
    print("=" * 60)
    print("Testing if removing extreme FWHM values improves consistency...")
    print()
    
    test_file = "test_targets/Slant-Edge-Target_rotated_sigma_1.5_blurred.png"
    
    # Test configurations
    tests = {
        "Before Outlier Filtering": "./mtf_analyzer_6_roi_debug",
        "After Outlier Filtering": "./mtf_analyzer_6_outlier_filtered"
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
            
            # Count skipped ROIs
            skipped_quality = len(re.findall(r'Skipping ROI due to poor quality', result.stdout))
            skipped_fwhm = len(re.findall(r'Skipping ROI due to unrealistic FWHM', result.stdout))
            
            fwhm_values = [float(f) for f in fwhm_matches]
            
            print(f"FWHM values: {[f'{f:.3f}' for f in fwhm_values]}")
            print(f"ROIs processed: {len(fwhm_values)}")
            print(f"ROIs skipped - Quality: {skipped_quality}, FWHM outliers: {skipped_fwhm}")
            
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
                    'range': max_fwhm - min_fwhm,
                    'processed_count': len(fwhm_values),
                    'skipped_quality': skipped_quality,
                    'skipped_fwhm': skipped_fwhm
                }
            elif len(fwhm_values) == 1:
                single_error = abs(fwhm_values[0] - 3.532) / 3.532 * 100
                print(f"Single ROI error vs expected: {single_error:.1f}%")
                
                results[name] = {
                    'mean_fwhm': fwhm_values[0],
                    'cv': 0,
                    'error': single_error,
                    'consistency': "N/A (single ROI)",
                    'range': 0,
                    'processed_count': 1,
                    'skipped_quality': skipped_quality,
                    'skipped_fwhm': skipped_fwhm
                }
            else:
                print("❌ No FWHM measurements obtained")
                results[name] = {
                    'processed_count': 0,
                    'skipped_quality': skipped_quality,
                    'skipped_fwhm': skipped_fwhm
                }
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Comparison
    if len(results) == 2:
        print("\n" + "=" * 60)
        print("OUTLIER FILTERING IMPACT")
        print("-" * 60)
        
        names = list(results.keys())
        before = results[names[0]]
        after = results[names[1]]
        
        print(f"ROIs processed: {before.get('processed_count', 0)} → {after.get('processed_count', 0)}")
        print(f"ROIs skipped (FWHM): {before.get('skipped_fwhm', 0)} → {after.get('skipped_fwhm', 0)}")
        
        if before.get('processed_count', 0) > 1 and after.get('processed_count', 0) > 1:
            print(f"Consistency (CV): {before['cv']:.1f}% → {after['cv']:.1f}%")
            cv_improvement = before['cv'] - after['cv']
            
            print(f"Accuracy (Error): {before['error']:.1f}% → {after['error']:.1f}%")
            error_change = after['error'] - before['error']
            
            print(f"FWHM Range: {before['range']:.3f} → {after['range']:.3f} pixels")
            range_improvement = before['range'] - after['range']
            
            print(f"Mean FWHM: {before['mean_fwhm']:.3f} → {after['mean_fwhm']:.3f}")
            
            # Assessment
            print(f"\nIMPROVEMENT ASSESSMENT:")
            
            if cv_improvement > 50:
                print(f"🎉 MAJOR consistency improvement: {cv_improvement:.1f} percentage points")
            elif cv_improvement > 20:
                print(f"✅ SIGNIFICANT consistency improvement: {cv_improvement:.1f} percentage points")
            elif cv_improvement > 0:
                print(f"✅ Consistency improvement: {cv_improvement:.1f} percentage points")
            else:
                print(f"⚠️ No consistency improvement (change: {cv_improvement:+.1f}%)")
            
            if range_improvement > 3.0:
                print(f"✅ FWHM range significantly reduced by {range_improvement:.3f} pixels")
            elif range_improvement > 0:
                print(f"✅ FWHM range reduced by {range_improvement:.3f} pixels")
            
            # Overall assessment
            if cv_improvement > 50 and after['cv'] < 25:
                print(f"🎉 EXCELLENT: Outlier filtering achieved target consistency!")
            elif cv_improvement > 20:
                print(f"✅ GOOD: Meaningful improvement achieved")
            elif cv_improvement > 0:
                print(f"⚠️ MODEST: Some improvement, filtering helped")
            else:
                print(f"❌ INEFFECTIVE: Outlier filtering didn't improve consistency")
                
            print(f"\nKey insight: ROI 2 (FWHM=6.087) was likely the outlier causing inconsistency")

if __name__ == "__main__":
    test_outlier_filtering()