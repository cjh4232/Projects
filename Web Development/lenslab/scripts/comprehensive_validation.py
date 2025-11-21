#!/usr/bin/env python3
"""
Comprehensive validation of improved MTF analyzer
Tests both FWHM accuracy and angle validation improvements
"""

import subprocess
import os
import json
import sys
import re
from pathlib import Path
import glob

class ComprehensiveValidator:
    def __init__(self, analyzer_path="./mtf_analyzer_6_improved"):
        self.analyzer_path = analyzer_path
        self.results = {
            'synthetic_tests': [],
            'angle_validation': [],
            'accuracy_improvement': {}
        }
        
    def run_analysis(self, image_path, sigma=None, debug=True):
        """Run MTF analysis on a single image"""
        cmd = [self.analyzer_path, image_path]
        
        if sigma is not None:
            cmd.extend(["--gaussian-sigma", str(sigma)])
        
        if debug:
            cmd.append("--debug")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Analysis timed out"
        except Exception as e:
            return False, "", str(e)
    
    def extract_metrics_from_output(self, output):
        """Extract FWHM and other metrics from analyzer output"""
        metrics = {}
        
        # Extract FWHM
        fwhm_patterns = [
            r'FWHM[:\s]*([0-9]+\.?[0-9]*)',
            r'Calculated FWHM[:\s]*([0-9]+\.?[0-9]*)',
            r'Measured FWHM[:\s]*([0-9]+\.?[0-9]*)'
        ]
        
        for pattern in fwhm_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                metrics['fwhm'] = float(matches[-1])
                break
        
        # Extract angle information
        angle_patterns = [
            r'Found.*edge.*angle\s*([0-9]+\.?[0-9]*)°',
            r'Edge angle:\s*([0-9]+\.?[0-9]*)',
            r'angle[:\s]*([0-9]+\.?[0-9]*)\s*degrees'
        ]
        
        for pattern in angle_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                metrics['detected_angle'] = float(matches[-1])
                break
        
        # Check for angle validation messages
        if "OPTIMAL" in output:
            metrics['angle_category'] = 'optimal'
        elif "acceptable" in output.lower():
            metrics['angle_category'] = 'acceptable'
        elif "Rejected" in output or "outside acceptable range" in output:
            metrics['angle_category'] = 'rejected'
        
        # Check for warnings
        metrics['has_warnings'] = "warning" in output.lower() or "WARNING" in output
        
        return metrics
    
    def test_synthetic_fwhm_accuracy(self):
        """Test the built-in synthetic FWHM calculation"""
        print("\n" + "="*60)
        print("TESTING SYNTHETIC FWHM ACCURACY")
        print("="*60)
        
        # Run the built-in synthetic test
        success, stdout, stderr = self.run_analysis("dummy", debug=True)
        
        if "Gaussian Test Results" in stdout:
            # Extract the synthetic test results
            lines = stdout.split('\n')
            for i, line in enumerate(lines):
                if "Input sigma:" in line:
                    try:
                        sigma_match = re.search(r'sigma:\s*([0-9.]+)', line)
                        theoretical_match = re.search(r'Theoretical FWHM:\s*([0-9.]+)', lines[i+1])
                        measured_match = re.search(r'Measured FWHM:\s*([0-9.]+)', lines[i+2])
                        ratio_match = re.search(r'Ratio.*:\s*([0-9.]+)', lines[i+3])
                        
                        if all([sigma_match, theoretical_match, measured_match, ratio_match]):
                            sigma = float(sigma_match.group(1))
                            theoretical = float(theoretical_match.group(1))
                            measured = float(measured_match.group(1))
                            ratio = float(ratio_match.group(1))
                            
                            error_percent = abs((measured - theoretical) / theoretical) * 100
                            
                            result = {
                                'sigma': sigma,
                                'theoretical_fwhm': theoretical,
                                'measured_fwhm': measured,
                                'ratio': ratio,
                                'error_percent': error_percent,
                                'passed': error_percent < 5.0  # 5% threshold
                            }
                            
                            self.results['synthetic_tests'].append(result)
                            
                            print(f"Sigma: {sigma:.1f}")
                            print(f"  Theoretical FWHM: {theoretical:.3f}")
                            print(f"  Measured FWHM: {measured:.3f}")
                            print(f"  Error: {error_percent:.1f}%")
                            print(f"  Status: {'✓ PASS' if result['passed'] else '✗ FAIL'}")
                            
                    except Exception as e:
                        print(f"Error parsing synthetic test results: {e}")
        else:
            print("⚠ Could not extract synthetic test results")
    
    def test_angle_validation(self):
        """Test angle validation on generated test targets"""
        print("\n" + "="*60)
        print("TESTING ANGLE VALIDATION")
        print("="*60)
        
        # Find all angle test targets
        test_files = glob.glob("angle_test_targets/test_angle_*.png")
        test_files.sort()
        
        if not test_files:
            print("⚠ No angle test targets found. Run generate_angle_test_targets.py first.")
            return
        
        for filepath in test_files:
            filename = os.path.basename(filepath)
            
            # Parse filename to extract expected parameters
            match = re.match(r'test_angle_([0-9.]+)deg_sigma_([0-9.]+)_(\w+)\.png', filename)
            if not match:
                continue
                
            expected_angle = float(match.group(1))
            sigma = float(match.group(2))
            category = match.group(3)
            theoretical_fwhm = 2.355 * sigma
            
            print(f"\nTesting: {filename}")
            print(f"  Expected angle: {expected_angle}°")
            print(f"  Category: {category}")
            print(f"  Expected result: {'PASS' if category in ['optimal', 'acceptable'] else 'REJECT'}")
            
            # Run analysis
            success, stdout, stderr = self.run_analysis(filepath, sigma=sigma, debug=True)
            
            if success:
                metrics = self.extract_metrics_from_output(stdout)
                
                result = {
                    'filename': filename,
                    'expected_angle': expected_angle,
                    'sigma': sigma,
                    'category': category,
                    'theoretical_fwhm': theoretical_fwhm,
                    'success': True,
                    'detected_angle': metrics.get('detected_angle'),
                    'measured_fwhm': metrics.get('fwhm'),
                    'angle_category': metrics.get('angle_category'),
                    'has_warnings': metrics.get('has_warnings', False)
                }
                
                if metrics.get('fwhm'):
                    result['fwhm_error_percent'] = abs((metrics['fwhm'] - theoretical_fwhm) / theoretical_fwhm) * 100
                    result['fwhm_passed'] = result['fwhm_error_percent'] < 5.0
                
                # Determine if angle validation worked correctly
                if category == 'optimal':
                    result['angle_validation_correct'] = metrics.get('angle_category') == 'optimal'
                elif category == 'acceptable':
                    result['angle_validation_correct'] = metrics.get('angle_category') == 'acceptable'
                elif category == 'invalid':
                    result['angle_validation_correct'] = metrics.get('angle_category') == 'rejected'
                else:
                    result['angle_validation_correct'] = False
                
                print(f"  Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
                if metrics.get('detected_angle'):
                    print(f"  Detected angle: {metrics['detected_angle']:.1f}°")
                if metrics.get('angle_category'):
                    print(f"  Angle category: {metrics['angle_category']}")
                if metrics.get('fwhm'):
                    print(f"  Measured FWHM: {metrics['fwhm']:.3f} (error: {result.get('fwhm_error_percent', 0):.1f}%)")
                
            else:
                print(f"  Result: ✗ ANALYSIS FAILED")
                result = {
                    'filename': filename,
                    'expected_angle': expected_angle,
                    'sigma': sigma,
                    'category': category,
                    'theoretical_fwhm': theoretical_fwhm,
                    'success': False,
                    'error': stderr
                }
            
            self.results['angle_validation'].append(result)
    
    def print_comprehensive_summary(self):
        """Print comprehensive summary of all tests"""
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION SUMMARY")
        print("="*80)
        
        # Synthetic test summary
        synthetic_tests = self.results['synthetic_tests']
        if synthetic_tests:
            synthetic_passed = sum(1 for t in synthetic_tests if t['passed'])
            print(f"\nSYNTHETIC FWHM TESTS: {synthetic_passed}/{len(synthetic_tests)} passed")
            
            if synthetic_tests:
                avg_error = sum(t['error_percent'] for t in synthetic_tests) / len(synthetic_tests)
                print(f"Average FWHM error: {avg_error:.1f}%")
        
        # Angle validation summary
        angle_tests = self.results['angle_validation']
        if angle_tests:
            successful_analyses = [t for t in angle_tests if t['success']]
            
            print(f"\nANGLE VALIDATION TESTS: {len(successful_analyses)}/{len(angle_tests)} completed")
            
            # Group by category
            categories = {}
            for test in successful_analyses:
                cat = test['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(test)
            
            for category in ['optimal', 'acceptable', 'invalid']:
                if category in categories:
                    tests = categories[category]
                    
                    angle_correct = sum(1 for t in tests if t.get('angle_validation_correct', False))
                    
                    if category != 'invalid':  # Invalid angles shouldn't produce FWHM results
                        fwhm_tests = [t for t in tests if 'fwhm_error_percent' in t]
                        fwhm_passed = sum(1 for t in fwhm_tests if t.get('fwhm_passed', False))
                        avg_fwhm_error = sum(t['fwhm_error_percent'] for t in fwhm_tests) / len(fwhm_tests) if fwhm_tests else 0
                        
                        print(f"\n{category.upper()} angles ({len(tests)} tests):")
                        print(f"  Angle validation: {angle_correct}/{len(tests)} correct")
                        print(f"  FWHM accuracy: {fwhm_passed}/{len(fwhm_tests)} passed (<5% error)")
                        print(f"  Average FWHM error: {avg_fwhm_error:.1f}%")
                    else:
                        print(f"\n{category.upper()} angles ({len(tests)} tests):")
                        print(f"  Correctly rejected: {angle_correct}/{len(tests)}")
        
        # Overall assessment
        print(f"\n{'='*80}")
        
        # Check if we achieved our goals
        synthetic_success = all(t['passed'] for t in synthetic_tests) if synthetic_tests else False
        
        if angle_tests:
            optimal_tests = [t for t in successful_analyses if t['category'] == 'optimal' and 'fwhm_error_percent' in t]
            optimal_accuracy = all(t['fwhm_error_percent'] < 3.0 for t in optimal_tests) if optimal_tests else False
            
            acceptable_tests = [t for t in successful_analyses if t['category'] == 'acceptable' and 'fwhm_error_percent' in t]
            acceptable_accuracy = all(t['fwhm_error_percent'] < 5.0 for t in acceptable_tests) if acceptable_tests else False
            
            invalid_tests = [t for t in successful_analyses if t['category'] == 'invalid']
            invalid_rejection = all(t.get('angle_validation_correct', False) for t in invalid_tests) if invalid_tests else False
        else:
            optimal_accuracy = acceptable_accuracy = invalid_rejection = False
        
        print("GOALS ASSESSMENT:")
        print(f"✓ Synthetic FWHM accuracy: {'ACHIEVED' if synthetic_success else 'NEEDS WORK'}")
        print(f"✓ Optimal angle accuracy (<3%): {'ACHIEVED' if optimal_accuracy else 'NEEDS WORK'}")
        print(f"✓ Acceptable angle accuracy (<5%): {'ACHIEVED' if acceptable_accuracy else 'NEEDS WORK'}")
        print(f"✓ Invalid angle rejection: {'ACHIEVED' if invalid_rejection else 'NEEDS WORK'}")
        
        overall_success = synthetic_success and optimal_accuracy and acceptable_accuracy and invalid_rejection
        print(f"\n🎯 OVERALL RESULT: {'🎉 SUCCESS - All improvements working!' if overall_success else '⚠️ PARTIAL SUCCESS - Some areas need refinement'}")

def main():
    validator = ComprehensiveValidator()
    
    # Test 1: Synthetic FWHM accuracy
    validator.test_synthetic_fwhm_accuracy()
    
    # Test 2: Angle validation
    validator.test_angle_validation()
    
    # Summary
    validator.print_comprehensive_summary()
    
    return 0

if __name__ == "__main__":
    exit(main())