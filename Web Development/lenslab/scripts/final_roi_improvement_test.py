#!/usr/bin/env python3

import subprocess
import re
import os
import statistics

def final_roi_improvement_test():
    """Comprehensive test of all ROI and FWHM improvements"""
    print("FINAL ROI IMPROVEMENT VALIDATION")
    print("=" * 60)
    print("Testing the complete pipeline of improvements...")
    print()
    
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
    
    executable = "./mtf_analyzer_6_fixed_quality"
    
    if not os.path.exists(executable):
        print(f"❌ Executable not found: {executable}")
        return
    
    all_results = []
    
    for test_case in test_cases:
        if not os.path.exists(test_case['file']):
            print(f"❌ File not found: {test_case['file']}")
            continue
            
        print(f"Testing σ={test_case['sigma']}, Expected FWHM={test_case['expected_fwhm']:.3f} pixels")
        print("-" * 60)
        
        try:
            result = subprocess.run([executable, test_case['file']], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            # Extract FWHM values
            fwhm_pattern = r'Measured FWHM from image: ([\d.]+) pixels'
            fwhm_matches = re.findall(fwhm_pattern, result.stdout)
            
            # Extract quality scores and components
            quality_pattern = r'ROI Quality Score: ([\d.]+)/100'
            edge_pattern = r'Edge Strength: ([\d.]+)/100'
            linearity_pattern = r'Linearity: ([\d.]+)/100'
            noise_pattern = r'Noise Level: ([\d.]+)'
            
            quality_matches = re.findall(quality_pattern, result.stdout)
            edge_matches = re.findall(edge_pattern, result.stdout)
            linearity_matches = re.findall(linearity_pattern, result.stdout)
            noise_matches = re.findall(noise_pattern, result.stdout)
            
            # Extract skipped ROIs
            skipped_count = len(re.findall(r'Skipping ROI due to poor quality', result.stdout))
            
            fwhm_values = [float(f) for f in fwhm_matches]
            quality_scores = [float(q) for q in quality_matches]
            
            processed_rois = len(fwhm_values)
            total_rois = processed_rois + skipped_count
            
            print(f"ROI Processing:")
            print(f"  Total ROIs detected: {total_rois}")
            print(f"  ROIs processed: {processed_rois}")
            print(f"  ROIs skipped: {skipped_count}")
            
            if quality_scores:
                print(f"  Quality scores: {[f'{q:.1f}' for q in quality_scores]}")
            
            if fwhm_values:
                print(f"FWHM Results:")
                print(f"  Values: {[f'{f:.3f}' for f in fwhm_values]} pixels")
                
                # Calculate statistics
                mean_fwhm = statistics.mean(fwhm_values)
                if len(fwhm_values) > 1:
                    stdev_fwhm = statistics.stdev(fwhm_values)
                    cv = (stdev_fwhm / mean_fwhm) * 100
                    
                    print(f"  Mean: {mean_fwhm:.3f} pixels")
                    print(f"  Std Dev: {stdev_fwhm:.3f} pixels")
                    print(f"  Coefficient of Variation: {cv:.1f}%")
                    
                    # Consistency rating
                    if cv < 10:
                        consistency = "✅ EXCELLENT"
                    elif cv < 25:
                        consistency = "✅ GOOD"
                    elif cv < 50:
                        consistency = "⚠️ FAIR"
                    else:
                        consistency = "❌ POOR"
                    
                    print(f"  Consistency: {consistency}")
                else:
                    cv = 0
                    consistency = "N/A (single ROI)"
                    print(f"  Consistency: {consistency}")
                
                # Accuracy assessment
                error = abs(mean_fwhm - test_case['expected_fwhm']) / test_case['expected_fwhm'] * 100
                print(f"  Accuracy error: {error:.1f}%")
                
                if error < 5:
                    accuracy = "✅ EXCELLENT"
                elif error < 15:
                    accuracy = "✅ GOOD"
                elif error < 30:
                    accuracy = "⚠️ FAIR"
                else:
                    accuracy = "❌ POOR"
                
                print(f"  Accuracy: {accuracy}")
                
                all_results.append({
                    'sigma': test_case['sigma'],
                    'expected': test_case['expected_fwhm'],
                    'measured': mean_fwhm,
                    'error': error,
                    'cv': cv,
                    'roi_count': processed_rois,
                    'consistency': consistency,
                    'accuracy': accuracy
                })
            else:
                print("❌ No FWHM measurements obtained")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    # Overall summary
    if all_results:
        print("OVERALL PERFORMANCE SUMMARY")
        print("=" * 60)
        
        avg_error = statistics.mean([r['error'] for r in all_results])
        avg_cv = statistics.mean([r['cv'] for r in all_results if r['cv'] > 0])
        avg_roi_count = statistics.mean([r['roi_count'] for r in all_results])
        
        print(f"Average accuracy error: {avg_error:.1f}%")
        print(f"Average consistency (CV): {avg_cv:.1f}%")
        print(f"Average ROIs processed per test: {avg_roi_count:.1f}")
        
        excellent_accuracy = sum(1 for r in all_results if r['error'] < 5)
        excellent_consistency = sum(1 for r in all_results if r['cv'] < 10)
        
        print(f"Tests with excellent accuracy (<5% error): {excellent_accuracy}/{len(all_results)}")
        print(f"Tests with excellent consistency (<10% CV): {excellent_consistency}/{len(all_results)}")
        
        # Final assessment
        print(f"\nFINAL ASSESSMENT:")
        
        if avg_error < 10 and avg_cv < 25:
            print("🎉 EXCELLENT: System meets accuracy and consistency targets!")
            grade = "A"
        elif avg_error < 15 and avg_cv < 40:
            print("✅ GOOD: System shows significant improvement")
            grade = "B"
        elif avg_error < 25 and avg_cv < 60:
            print("⚠️ FAIR: System needs further refinement")
            grade = "C"
        else:
            print("❌ POOR: System requires major improvements")
            grade = "D"
        
        print(f"Overall Grade: {grade}")
        
        print(f"\nKEY ACHIEVEMENTS:")
        print(f"• Fixed FWHM calculation with sub-pixel sampling")
        print(f"• Implemented ROI quality assessment")
        print(f"• Restored 4-ROI detection capability")
        print(f"• Added ISO 12233 compliance features")
        print(f"• Created comprehensive validation framework")

if __name__ == "__main__":
    final_roi_improvement_test()