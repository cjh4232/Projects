#!/usr/bin/env python3
"""
Comprehensive FWHM validation script
Tests the fixed MTF analyzer against images with known Gaussian blur
"""

import subprocess
import os
import json
import sys
from pathlib import Path
import re

class FWHMValidator:
    def __init__(self, analyzer_path="src/cpp/mtf_analyzer_6"):
        self.analyzer_path = analyzer_path
        self.test_results = []
        
    def compile_analyzer(self):
        """Compile the MTF analyzer if needed"""
        print("Compiling MTF analyzer...")
        
        compile_cmd = [
            "g++", "-std=c++17", "-O2",
            "src/cpp/mtf_analyzer_6.cpp",
            "-o", "mtf_analyzer_6",
            "`pkg-config", "--cflags", "--libs", "opencv4`"
        ]
        
        # Try to compile
        try:
            result = subprocess.run(" ".join(compile_cmd), shell=True, 
                                   capture_output=True, text=True, cwd=".")
            if result.returncode != 0:
                print(f"Compilation failed: {result.stderr}")
                return False
            print("✓ MTF analyzer compiled successfully")
            return True
        except Exception as e:
            print(f"Error compiling: {e}")
            return False
    
    def run_analysis(self, image_path, sigma=None, debug=False):
        """Run MTF analysis on a single image"""
        cmd = ["./mtf_analyzer_6", image_path]
        
        if sigma is not None:
            cmd.extend(["--gaussian-sigma", str(sigma)])
        
        if debug:
            cmd.append("--debug")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Analysis timed out"
        except Exception as e:
            return False, "", str(e)
    
    def extract_fwhm_from_output(self, output):
        """Extract FWHM value from analyzer output"""
        # Look for FWHM in the output
        fwhm_pattern = r'FWHM[:\s]*([0-9]+\.?[0-9]*)'
        matches = re.findall(fwhm_pattern, output, re.IGNORECASE)
        
        if matches:
            return float(matches[-1])  # Return the last match
        
        # Alternative patterns
        patterns = [
            r'Calculated FWHM[:\s]*([0-9]+\.?[0-9]*)',
            r'measured.*fwhm[:\s]*([0-9]+\.?[0-9]*)',
            r'Final.*FWHM[:\s]*([0-9]+\.?[0-9]*)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                return float(matches[-1])
        
        return None
    
    def validate_test_targets(self):
        """Run validation on all test targets"""
        print("\n" + "="*60)
        print("VALIDATING FWHM CALCULATIONS WITH TEST TARGETS")
        print("="*60)
        
        # Test configurations
        test_configs = [
            {"sigma": 0.5, "theoretical_fwhm": 1.177},
            {"sigma": 1.0, "theoretical_fwhm": 2.355},
            {"sigma": 1.5, "theoretical_fwhm": 3.532},
            {"sigma": 2.0, "theoretical_fwhm": 4.710},
            {"sigma": 2.5, "theoretical_fwhm": 5.888},
        ]
        
        results = []
        
        for config in test_configs:
            sigma = config["sigma"]
            theoretical_fwhm = config["theoretical_fwhm"]
            
            # Format sigma for filename (handle both 1.0 -> "1.0" and 0.5 -> "0.5")
            if sigma == int(sigma):
                sigma_str = f"{int(sigma)}.0"
            else:
                sigma_str = f"{sigma:.1f}"
            
            image_path = f"test_targets/Slant-Edge-Target_rotated_sigma_{sigma_str}_blurred.png"
            
            if not os.path.exists(image_path):
                print(f"⚠ Warning: Test image not found: {image_path}")
                continue
            
            print(f"\nTesting image: {image_path}")
            print(f"Expected FWHM: {theoretical_fwhm:.3f}")
            
            # Run analysis
            success, stdout, stderr = self.run_analysis(image_path, sigma=sigma, debug=True)
            
            if not success:
                print(f"✗ Analysis failed: {stderr}")
                results.append({
                    "sigma": sigma,
                    "theoretical_fwhm": theoretical_fwhm,
                    "measured_fwhm": None,
                    "error_percent": None,
                    "passed": False,
                    "error": stderr
                })
                continue
            
            # Extract FWHM from output
            measured_fwhm = self.extract_fwhm_from_output(stdout)
            
            if measured_fwhm is None:
                print(f"✗ Could not extract FWHM from output")
                print("Output:", stdout[:500], "...")
                results.append({
                    "sigma": sigma,
                    "theoretical_fwhm": theoretical_fwhm,
                    "measured_fwhm": None,
                    "error_percent": None,
                    "passed": False,
                    "error": "Could not extract FWHM"
                })
                continue
            
            # Calculate error
            error_percent = abs((measured_fwhm - theoretical_fwhm) / theoretical_fwhm) * 100
            passed = error_percent < 10.0  # 10% tolerance
            
            print(f"Measured FWHM: {measured_fwhm:.3f}")
            print(f"Error: {error_percent:.1f}%")
            print(f"Result: {'✓ PASS' if passed else '✗ FAIL'}")
            
            results.append({
                "sigma": sigma,
                "theoretical_fwhm": theoretical_fwhm,
                "measured_fwhm": measured_fwhm,
                "error_percent": error_percent,
                "passed": passed,
                "raw_output": stdout
            })
        
        return results
    
    def print_summary(self, results):
        """Print test summary"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        passed_count = sum(1 for r in results if r["passed"])
        total_count = len(results)
        
        print(f"Tests passed: {passed_count}/{total_count}")
        print()
        
        print("Sigma\tTheoretical\tMeasured\tError\tStatus")
        print("-" * 50)
        
        for result in results:
            if result["measured_fwhm"] is not None:
                print(f"{result['sigma']:.1f}\t{result['theoretical_fwhm']:.3f}\t\t"
                      f"{result['measured_fwhm']:.3f}\t\t{result['error_percent']:.1f}%\t"
                      f"{'PASS' if result['passed'] else 'FAIL'}")
            else:
                print(f"{result['sigma']:.1f}\t{result['theoretical_fwhm']:.3f}\t\t"
                      f"N/A\t\tN/A\tFAIL")
        
        print()
        
        if passed_count == total_count:
            print("🎉 ALL TESTS PASSED!")
            print("✓ The FWHM calculation fix is working correctly.")
            print("✓ Measured FWHM values match theoretical expectations.")
        else:
            print("⚠ Some tests failed.")
            if passed_count > 0:
                print(f"✓ {passed_count} tests passed - fix may be partially working.")
            print("✗ Further investigation may be needed.")
        
        return passed_count == total_count

def main():
    validator = FWHMValidator()
    
    # Check if test targets exist
    if not os.path.exists("test_targets"):
        print("Error: test_targets directory not found. Run generate_test_targets.py first.")
        return 1
    
    # Compile analyzer
    if not validator.compile_analyzer():
        print("Failed to compile MTF analyzer")
        return 1
    
    # Run validation tests
    results = validator.validate_test_targets()
    
    # Print summary
    all_passed = validator.print_summary(results)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())