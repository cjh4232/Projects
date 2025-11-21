#!/usr/bin/env python3
"""
Validate MTF results against theoretical Gaussian MTF curves.
For Gaussian blur with sigma σ, the theoretical MTF is: MTF(f) = exp(-2π²σ²f²)
"""

import numpy as np
import subprocess
import re
import matplotlib.pyplot as plt

def theoretical_mtf(freq, sigma):
    """Calculate theoretical MTF for Gaussian blur."""
    return np.exp(-2 * np.pi**2 * sigma**2 * freq**2)

def find_mtf_frequency(freqs, mtf_values, target_mtf):
    """Find frequency where MTF crosses target value."""
    try:
        # Find crossing point using linear interpolation
        for i in range(len(mtf_values) - 1):
            if mtf_values[i] >= target_mtf and mtf_values[i+1] < target_mtf:
                # Linear interpolation
                ratio = (target_mtf - mtf_values[i+1]) / (mtf_values[i] - mtf_values[i+1])
                freq_interp = freqs[i+1] + ratio * (freqs[i] - freqs[i+1])
                return freq_interp
        return None
    except:
        return None

def extract_mtf_metrics(target_file):
    """Extract MTF50 values from analyzer output."""
    try:
        # Run MTF analyzer and capture output
        result = subprocess.run(['./mtf_analyzer_6_final', target_file], 
                              capture_output=True, text=True)
        
        # Extract MTF50 values from final results output
        mtf50_pattern = r'MTF50: ([\d\.]+) cycles/pixel'
        mtf50_values = re.findall(mtf50_pattern, result.stdout + result.stderr)
        
        if mtf50_values:
            # Return the best (highest) MTF50 value - likely from main edge
            return max(float(val) for val in mtf50_values)
        
        return None
    except Exception as e:
        print(f"Error processing {target_file}: {e}")
        return None

def validate_mtf_results():
    """Validate MTF results against theoretical predictions."""
    print("=== MTF THEORETICAL VALIDATION ===")
    print("Comparing measured MTF50 vs theoretical Gaussian MTF\n")
    
    # Test different sigma values
    test_cases = [
        (0.5, 1.177),   # sigma, theoretical FWHM
        (1.0, 2.355),
        (1.5, 3.532),
        (2.0, 4.710)
    ]
    
    results = []
    
    print("| Sigma | Theoretical MTF50 | Measured MTF50 | Error | Status |")
    print("|-------|------------------|----------------|-------|--------|")
    
    for sigma, theoretical_fwhm in test_cases:
        target_file = f"working_targets/working_sigma_{sigma}.png"
        
        # Calculate theoretical MTF50 (frequency where MTF = 0.5)
        # For Gaussian: MTF(f) = exp(-2π²σ²f²) = 0.5
        # Solving: f = sqrt(-ln(0.5)/(2π²σ²))
        theoretical_mtf50 = np.sqrt(-np.log(0.5) / (2 * np.pi**2 * sigma**2))
        
        # Get measured MTF50
        measured_mtf50 = extract_mtf_metrics(target_file)
        
        if measured_mtf50:
            error = ((measured_mtf50 - theoretical_mtf50) / theoretical_mtf50) * 100
            
            if abs(error) < 20:
                status = "✅ GOOD"
            elif abs(error) < 50:
                status = "⚠️ FAIR"  
            else:
                status = "❌ POOR"
                
            print(f"| {sigma:4.1f} | {theoretical_mtf50:15.6f} | {measured_mtf50:13.6f} | {error:+5.1f}% | {status} |")
            
            results.append({
                'sigma': sigma,
                'theoretical': theoretical_mtf50, 
                'measured': measured_mtf50,
                'error': error
            })
        else:
            print(f"| {sigma:4.1f} | {theoretical_mtf50:15.6f} | FAILED         | -     | ❌ FAIL |")
    
    # Create validation plot
    if results:
        create_mtf_validation_plot(results)
        
    print(f"\n=== SUMMARY ===")
    print(f"Processed {len(results)} successful measurements")
    if results:
        avg_error = np.mean([abs(r['error']) for r in results])
        print(f"Average absolute error: {avg_error:.1f}%")
        
        if avg_error < 20:
            print("✅ MTF measurements are theoretically sound!")
        elif avg_error < 50:
            print("⚠️ MTF measurements are reasonable but could be improved")
        else:
            print("❌ MTF measurements need significant improvement")

def create_mtf_validation_plot(results):
    """Create validation plot comparing theoretical vs measured MTF50."""
    plt.figure(figsize=(10, 6))
    
    sigmas = [r['sigma'] for r in results]
    theoretical = [r['theoretical'] for r in results]
    measured = [r['measured'] for r in results]
    
    plt.subplot(1, 2, 1)
    plt.plot(sigmas, theoretical, 'b-o', label='Theoretical MTF50')
    plt.plot(sigmas, measured, 'r-s', label='Measured MTF50')
    plt.xlabel('Blur Sigma')
    plt.ylabel('MTF50 (cycles/pixel)')
    plt.title('MTF50 vs Blur Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    errors = [r['error'] for r in results]
    plt.bar(sigmas, errors, alpha=0.7, color=['green' if abs(e) < 20 else 'orange' if abs(e) < 50 else 'red' for e in errors])
    plt.xlabel('Blur Sigma')
    plt.ylabel('Error (%)')
    plt.title('MTF50 Measurement Error')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='±20% threshold')
    plt.axhline(y=-20, color='orange', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mtf_theoretical_validation.png', dpi=150, bbox_inches='tight')
    print("Validation plot saved to mtf_theoretical_validation.png")

if __name__ == "__main__":
    validate_mtf_results()