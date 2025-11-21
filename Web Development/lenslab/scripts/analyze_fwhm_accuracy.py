#!/usr/bin/env python3
"""
FWHM Accuracy Analysis

Analyzes the relationship between sampling intervals, edge angles, 
and FWHM measurement accuracy to identify optimization opportunities.
"""

import re
import subprocess
import os

def run_mtf_analysis(target_file):
    """Run MTF analyzer and extract key metrics"""
    try:
        result = subprocess.run(
            ["./mtf_analyzer_6_coordinate_fix", target_file, "--debug"],
            capture_output=True,
            text=True,
            cwd="/Users/codyhatch/Documents/Github Projects/Projects/Web Development/lenslab"
        )
        
        output = result.stdout + result.stderr
        return output
    except Exception as e:
        print(f"Error running analysis: {e}")
        return ""

def extract_metrics(output):
    """Extract FWHM measurements, sampling intervals, and angles"""
    metrics = []
    
    # Extract sampling intervals and angles
    sampling_matches = re.findall(r'Using adaptive sampling interval: ([\d.]+)', output)
    angle_matches = re.findall(r'Calculated line angle: ([\d.]+)', output)
    
    # Extract FWHM measurements
    fwhm_matches = re.findall(r'Measured FWHM from image: ([\d.]+) pixels', output)
    
    # Extract quality classifications
    quality_matches = re.findall(r'Research classification: (\w+)', output)
    
    # Combine the data
    for i in range(min(len(sampling_matches), len(angle_matches), len(fwhm_matches))):
        metrics.append({
            'roi': i,
            'sampling_interval': float(sampling_matches[i]),
            'edge_angle': float(angle_matches[i]),
            'fwhm_pixels': float(fwhm_matches[i]),
            'quality_class': quality_matches[i] if i < len(quality_matches) else 'unknown'
        })
    
    return metrics

def analyze_target(target_name, expected_fwhm):
    """Analyze a specific target and calculate accuracy metrics"""
    print(f"\n=== Analyzing {target_name} ===")
    print(f"Expected FWHM: {expected_fwhm:.3f} pixels")
    
    target_file = f"working_targets/{target_name}.png"
    output = run_mtf_analysis(target_file)
    
    if not output:
        print("❌ Failed to run analysis")
        return None
    
    metrics = extract_metrics(output)
    
    if not metrics:
        print("❌ No valid metrics extracted")
        return None
    
    print(f"Found {len(metrics)} ROI measurements:")
    
    results = []
    for metric in metrics:
        error_percent = abs(metric['fwhm_pixels'] - expected_fwhm) / expected_fwhm * 100
        
        print(f"  ROI {metric['roi']}: {metric['fwhm_pixels']:.3f} px "
              f"(error: {error_percent:.1f}%) "
              f"sampling: {metric['sampling_interval']:.3f} "
              f"angle: {metric['edge_angle']:.1f}° "
              f"quality: {metric['quality_class']}")
        
        results.append({
            'target': target_name,
            'expected_fwhm': expected_fwhm,
            **metric,
            'error_percent': error_percent
        })
    
    return results

def main():
    """Analyze FWHM accuracy across multiple targets"""
    
    print("=" * 60)
    print("FWHM ACCURACY ANALYSIS")
    print("Investigating sampling intervals vs measurement accuracy")
    print("=" * 60)
    
    # Test configurations with correct theoretical FWHM values
    test_targets = [
        ("working_sigma_0.5", 2.355 * 0.5),   # 1.178 pixels
        ("working_sigma_1.0", 2.355 * 1.0),   # 2.355 pixels
        ("working_sigma_1.5", 2.355 * 1.5),   # 3.533 pixels
        ("working_sigma_2.0", 2.355 * 2.0),   # 4.710 pixels
        ("working_sigma_2.5", 2.355 * 2.5),   # 5.888 pixels
    ]
    
    all_results = []
    
    for target_name, expected_fwhm in test_targets:
        results = analyze_target(target_name, expected_fwhm)
        if results:
            all_results.extend(results)
    
    # Analysis summary
    print("\n" + "=" * 60)
    print("ACCURACY ANALYSIS SUMMARY")
    print("=" * 60)
    
    if not all_results:
        print("❌ No results to analyze")
        return
    
    # Group by sampling interval
    sampling_groups = {}
    for result in all_results:
        interval = round(result['sampling_interval'], 2)
        if interval not in sampling_groups:
            sampling_groups[interval] = []
        sampling_groups[interval].append(result)
    
    print("\nAccuracy by Sampling Interval:")
    for interval in sorted(sampling_groups.keys()):
        results = sampling_groups[interval]
        errors = [r['error_percent'] for r in results]
        avg_error = sum(errors) / len(errors)
        min_error = min(errors)
        max_error = max(errors)
        
        print(f"  Interval {interval}: {len(results)} measurements")
        print(f"    Average error: {avg_error:.1f}%")
        print(f"    Error range: {min_error:.1f}% - {max_error:.1f}%")
    
    # Group by quality classification
    quality_groups = {}
    for result in all_results:
        quality = result['quality_class']
        if quality not in quality_groups:
            quality_groups[quality] = []
        quality_groups[quality].append(result)
    
    print("\nAccuracy by Quality Classification:")
    for quality in sorted(quality_groups.keys()):
        results = quality_groups[quality]
        errors = [r['error_percent'] for r in results]
        avg_error = sum(errors) / len(errors)
        
        print(f"  {quality}: {len(results)} measurements, avg error: {avg_error:.1f}%")
    
    # Find best and worst performers
    best_result = min(all_results, key=lambda x: x['error_percent'])
    worst_result = max(all_results, key=lambda x: x['error_percent'])
    
    print(f"\n🏆 Best measurement: {best_result['fwhm_pixels']:.3f} px "
          f"(error: {best_result['error_percent']:.1f}%) "
          f"sampling: {best_result['sampling_interval']:.3f}")
    
    print(f"❌ Worst measurement: {worst_result['fwhm_pixels']:.3f} px "
          f"(error: {worst_result['error_percent']:.1f}%) "
          f"sampling: {worst_result['sampling_interval']:.3f}")
    
    print("\n🎯 RECOMMENDATIONS:")
    
    # Check if sampling interval correlates with accuracy
    good_results = [r for r in all_results if r['error_percent'] < 30]
    bad_results = [r for r in all_results if r['error_percent'] > 70]
    
    if good_results and bad_results:
        good_avg_sampling = sum(r['sampling_interval'] for r in good_results) / len(good_results)
        bad_avg_sampling = sum(r['sampling_interval'] for r in bad_results) / len(bad_results)
        
        print(f"- Good results (error <30%) avg sampling: {good_avg_sampling:.3f}")
        print(f"- Bad results (error >70%) avg sampling: {bad_avg_sampling:.3f}")
        
        if abs(good_avg_sampling - bad_avg_sampling) > 0.05:
            print("- ✅ Sampling interval appears to correlate with accuracy")
        else:
            print("- ⚠️ Sampling interval may not be the primary factor")
    
    return all_results

if __name__ == "__main__":
    results = main()