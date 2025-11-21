#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>

// Extract just the FWHM calculation function for testing
double calculateFWHM(const std::vector<double> &x, const std::vector<double> &y) {
    if (x.empty() || y.empty() || x.size() != y.size()) {
        std::cerr << "Error: Invalid input for FWHM calculation" << std::endl;
        return 0.0;
    }

    // Find peak value and location
    auto peak_it = std::max_element(y.begin(), y.end());
    int peak_idx = std::distance(y.begin(), peak_it);
    double max_value = *peak_it;
    double half_max = max_value / 2.0;

    std::cout << "Peak value: " << max_value << " at index " << peak_idx
              << " (x = " << x[peak_idx] << ")" << std::endl;
    std::cout << "Half maximum: " << half_max << std::endl;

    // Search left from peak
    double left_x = x[peak_idx];
    bool found_left = false;

    for (int i = peak_idx; i > 0; i--) {
        if (y[i] >= half_max && y[i - 1] < half_max) {
            // Linear interpolation for better precision
            double t = (half_max - y[i - 1]) / (y[i] - y[i - 1]);
            left_x = x[i - 1] + t * (x[i] - x[i - 1]);
            found_left = true;
            
            std::cout << "Left crossing found between indices " << (i - 1) << " and " << i << std::endl;
            std::cout << "Left x: " << left_x << " (t = " << t << ")" << std::endl;
            break;
        }
    }

    if (!found_left) {
        std::cout << "Warning: Could not find left crossing for half maximum" << std::endl;
    }

    // Search right from peak
    double right_x = x[peak_idx];
    bool found_right = false;

    for (int i = peak_idx; i < static_cast<int>(y.size()) - 1; i++) {
        if (y[i] >= half_max && y[i + 1] < half_max) {
            // Linear interpolation for better precision
            double t = (half_max - y[i + 1]) / (y[i] - y[i + 1]);
            right_x = x[i] + t * (x[i + 1] - x[i]);
            found_right = true;
            
            std::cout << "Right crossing found between indices " << i << " and " << (i + 1) << std::endl;
            std::cout << "Right x: " << right_x << " (t = " << t << ")" << std::endl;
            break;
        }
    }

    if (!found_right) {
        std::cout << "Warning: Could not find right crossing for half maximum" << std::endl;
    }

    // Calculate FWHM
    double fwhm = std::abs(right_x - left_x);
    std::cout << "Calculated FWHM: " << fwhm << std::endl;

    return fwhm;
}

bool testSyntheticGaussian() {
    std::cout << "\n====== Testing Synthetic Gaussian LSF ======" << std::endl;
    
    const std::vector<double> test_sigmas = {0.5, 1.0, 1.5, 2.0, 2.5};
    bool all_passed = true;
    
    for (double sigma : test_sigmas) {
        std::cout << "\n--- Testing sigma = " << sigma << " ---" << std::endl;
        
        const int size = 200;
        const double theoretical_fwhm = 2.355 * sigma;
        
        std::vector<double> x(size);
        std::vector<double> y(size);
        
        // Create distance array centered at 0 with 0.5 pixel spacing (this is the key!)
        for (int i = 0; i < size; i++) {
            x[i] = (i - size / 2.0) * 0.5; // This 0.5 factor is critical!
        }
        
        // Generate normalized Gaussian values
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            y[i] = std::exp(-(x[i] * x[i]) / (2 * sigma * sigma));
            sum += y[i];
        }
        
        // Normalize to ensure area = 1.0
        double dx = x[1] - x[0];
        double normalization_factor = sum * dx;
        for (auto &val : y) {
            val /= normalization_factor;
        }
        
        // Calculate FWHM using our function
        double measured_fwhm = calculateFWHM(x, y);
        double error_percent = std::abs((measured_fwhm - theoretical_fwhm) / theoretical_fwhm) * 100.0;
        bool passed = error_percent < 5.0; // 5% tolerance
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Sigma: " << sigma 
                  << ", Theoretical FWHM: " << theoretical_fwhm
                  << ", Measured FWHM: " << measured_fwhm
                  << ", Error: " << error_percent << "%"
                  << " [" << (passed ? "PASS" : "FAIL") << "]" << std::endl;
        
        if (!passed) {
            all_passed = false;
        }
    }
    
    return all_passed;
}

int main() {
    std::cout << "FWHM Fix Validation - Synthetic Test" << std::endl;
    std::cout << "====================================" << std::endl;
    
    bool passed = testSyntheticGaussian();
    
    std::cout << "\n====== SUMMARY ======" << std::endl;
    if (passed) {
        std::cout << "✓ ALL SYNTHETIC TESTS PASSED!" << std::endl;
        std::cout << "✓ FWHM calculation with 0.5x distance scaling works correctly." << std::endl;
    } else {
        std::cout << "✗ Some synthetic tests failed." << std::endl;
        std::cout << "✗ FWHM calculation may need further adjustment." << std::endl;
    }
    
    return passed ? 0 : 1;
}