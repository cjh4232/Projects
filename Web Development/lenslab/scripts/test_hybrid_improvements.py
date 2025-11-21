#!/usr/bin/env python3
"""
Test if the hybrid system (4-ROI detection + research validation) 
improved our FWHM accuracy scores compared to the previous version
"""

import subprocess
import os
import re
from pathlib import Path

class HybridValidator:
    def __init__(self):
        self.hybrid_analyzer = "./mtf_analyzer_6_hybrid"
        self.improved_analyzer = "./mtf_analyzer_6_improved"  # Previous version
        
    def run_analysis(self, analyzer_path, image_path, sigma=None):
        """Run MTF analysis and extract FWHM"""
        cmd = [analyzer_path, image_path]
        if sigma:
            cmd.extend(["--gaussian-sigma", str(sigma)])
        cmd.append("--debug")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)
    
    def extract_fwhm_from_output(self, output):
        """Extract FWHM value from analyzer output"""
        fwhm_patterns = [
            r'Final LSF Statistics:\s*FWHM:\s*([0-9]+\.?[0-9]*)',
            r'FWHM:\s*([0-9]+\.?[0-9]*)',
            r'Calculated FWHM:\s*([0-9]+\.?[0-9]*)',
            r'Measured FWHM:\s*([0-9]+\.?[0-9]*)'
        ]
        
        for pattern in fwhm_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])  # Return the last match
                except:
                    continue
        
        return None
    
    def count_rois(self, output):
        """Count number of ROIs detected"""
        roi_match = re.search(r'Found (\d+) edges at target angles', output)
        return int(roi_match.group(1)) if roi_match else 0
    
    def test_comparison_on_known_targets(self):
        """Compare hybrid vs improved system on our test targets"""
        test_targets = [
            {"path": "test_targets/Slant-Edge-Target_rotated_sigma_1.0_blurred.png", "sigma": 1.0, "theoretical_fwhm": 2.355},
            {"path": "test_targets/Slant-Edge-Target_rotated_sigma_1.5_blurred.png", "sigma": 1.5, "theoretical_fwhm": 3.532},
            {"path": "test_targets/Slant-Edge-Target_rotated_sigma_2.0_blurred.png", "sigma": 2.0, "theoretical_fwhm": 4.710},
        ]
        
        print("=" * 80)
        print("COMPARING HYBRID SYSTEM vs PREVIOUS IMPROVED SYSTEM")
        print("=" * 80)
        
        results = []
        
        for target in test_targets:
            if not os.path.exists(target["path"]):
                print(f"⚠ Skipping missing target: {target['path']}")
                continue
                
            print(f"\nTesting: {os.path.basename(target['path'])}")
            print(f"Expected FWHM: {target['theoretical_fwhm']:.3f}")
            
            # Test hybrid system
            hybrid_success, hybrid_out, hybrid_err = self.run_analysis(
                self.hybrid_analyzer, target["path"], target["sigma"]
            )
            
            # Test previous improved system  
            improved_success, improved_out, improved_err = self.run_analysis(
                self.improved_analyzer, target["path"], target["sigma"]
            )
            
            result = {
                "target": target["path"],
                "sigma": target["sigma"],
                "theoretical_fwhm": target["theoretical_fwhm"],
                "hybrid": {
                    "success": hybrid_success,
                    "rois": self.count_rois(hybrid_out) if hybrid_success else 0,
                    "fwhm": self.extract_fwhm_from_output(hybrid_out) if hybrid_success else None,
                    "error": hybrid_err if not hybrid_success else None
                },
                "improved": {
                    "success": improved_success,
                    "rois": self.count_rois(improved_out) if improved_success else 0,
                    "fwhm": self.extract_fwhm_from_output(improved_out) if improved_success else None,
                    "error": improved_err if not improved_success else None
                }
            }
            
            # Calculate errors
            if result["hybrid"]["fwhm"]:
                result["hybrid"]["error_percent"] = abs((result["hybrid"]["fwhm"] - target["theoretical_fwhm"]) / target["theoretical_fwhm"]) * 100
            
            if result["improved"]["fwhm"]:
                result["improved"]["error_percent"] = abs((result["improved"]["fwhm"] - target["theoretical_fwhm"]) / target["theoretical_fwhm"]) * 100
            
            results.append(result)
            
            # Print comparison
            print(f"\nHYBRID SYSTEM:")
            if result["hybrid"]["success"]:
                print(f"  ROIs detected: {result['hybrid']['rois']}")
                print(f"  FWHM: {result['hybrid']['fwhm']:.3f}")
                print(f"  Error: {result['hybrid'].get('error_percent', 0):.1f}%")
                print(f"  Status: {'✓ PASS' if result['hybrid'].get('error_percent', 100) < 5.0 else '✗ FAIL'}")
            else:
                print(f"  Status: ✗ ANALYSIS FAILED")
                print(f"  Error: {result['hybrid']['error']}")
            
            print(f"\nPREVIOUS IMPROVED SYSTEM:")
            if result["improved"]["success"]:
                print(f"  ROIs detected: {result['improved']['rois']}")
                print(f"  FWHM: {result['improved']['fwhm']:.3f}")
                print(f"  Error: {result['improved'].get('error_percent', 0):.1f}%")
                print(f"  Status: {'✓ PASS' if result['improved'].get('error_percent', 100) < 5.0 else '✗ FAIL'}")
            else:
                print(f"  Status: ✗ ANALYSIS FAILED")
                print(f"  Error: {result['improved']['error']}")
        
        return results
    
    def print_summary(self, results):
        """Print comparison summary"""
        print("\n" + "=" * 80)
        print("IMPROVEMENT SUMMARY")
        print("=" * 80)
        
        successful_hybrid = [r for r in results if r["hybrid"]["success"]]
        successful_improved = [r for r in results if r["improved"]["success"]]
        
        print(f"\nSuccessful analyses:")
        print(f"  Hybrid system: {len(successful_hybrid)}/{len(results)}")
        print(f"  Previous system: {len(successful_improved)}/{len(results)}")
        
        # ROI comparison
        if successful_hybrid and successful_improved:
            avg_rois_hybrid = sum(r["hybrid"]["rois"] for r in successful_hybrid) / len(successful_hybrid)
            avg_rois_improved = sum(r["improved"]["rois"] for r in successful_improved) / len(successful_improved)
            
            print(f"\nAverage ROIs detected:")
            print(f"  Hybrid system: {avg_rois_hybrid:.1f}")
            print(f"  Previous system: {avg_rois_improved:.1f}")
            print(f"  Improvement: {'+' if avg_rois_hybrid > avg_rois_improved else ''}{avg_rois_hybrid - avg_rois_improved:.1f} ROIs")
        
        # FWHM accuracy comparison
        hybrid_errors = [r["hybrid"]["error_percent"] for r in successful_hybrid if "error_percent" in r["hybrid"]]
        improved_errors = [r["improved"]["error_percent"] for r in successful_improved if "error_percent" in r["improved"]]
        
        if hybrid_errors and improved_errors:
            avg_error_hybrid = sum(hybrid_errors) / len(hybrid_errors)
            avg_error_improved = sum(improved_errors) / len(improved_errors)
            
            print(f"\nAverage FWHM errors:")
            print(f"  Hybrid system: {avg_error_hybrid:.1f}%")
            print(f"  Previous system: {avg_error_improved:.1f}%")
            print(f"  Improvement: {avg_error_improved - avg_error_hybrid:.1f} percentage points")
            
            # Check if we improved
            if avg_error_hybrid < avg_error_improved:
                print(f"  🎉 HYBRID SYSTEM IS MORE ACCURATE!")
            elif avg_error_hybrid > avg_error_improved:
                print(f"  ⚠ Previous system was more accurate")
            else:
                print(f"  📊 Similar accuracy")
        
        # Pass rate comparison
        hybrid_pass = sum(1 for r in successful_hybrid if r["hybrid"].get("error_percent", 100) < 5.0)
        improved_pass = sum(1 for r in successful_improved if r["improved"].get("error_percent", 100) < 5.0)
        
        print(f"\nTests passing <5% error threshold:")
        print(f"  Hybrid system: {hybrid_pass}/{len(successful_hybrid)}")
        print(f"  Previous system: {improved_pass}/{len(successful_improved)}")
        
        # Overall assessment
        print("\n" + "=" * 80)
        better_success_rate = len(successful_hybrid) > len(successful_improved)
        better_roi_detection = avg_rois_hybrid > avg_rois_improved if successful_hybrid and successful_improved else False
        better_accuracy = avg_error_hybrid < avg_error_improved if hybrid_errors and improved_errors else False
        
        improvements = []
        if better_success_rate:
            improvements.append("✅ Higher success rate")
        if better_roi_detection:
            improvements.append("✅ Better ROI detection") 
        if better_accuracy:
            improvements.append("✅ Improved FWHM accuracy")
        
        if improvements:
            print("HYBRID SYSTEM IMPROVEMENTS:")
            for improvement in improvements:
                print(f"  {improvement}")
        else:
            print("📊 MIXED RESULTS - Some metrics improved, others may need work")
        
        return len(improvements) > 0

def main():
    validator = HybridValidator()
    
    # Check if both analyzers exist
    if not os.path.exists(validator.hybrid_analyzer):
        print(f"Error: Hybrid analyzer not found: {validator.hybrid_analyzer}")
        return 1
        
    if not os.path.exists(validator.improved_analyzer):
        print(f"Error: Previous analyzer not found: {validator.improved_analyzer}")
        print("Note: This comparison requires the previous version for baseline")
        return 1
    
    # Run comparison tests
    results = validator.test_comparison_on_known_targets()
    
    # Print summary
    improved = validator.print_summary(results)
    
    return 0 if improved else 1

if __name__ == "__main__":
    exit(main())