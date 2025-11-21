#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <emscripten/emscripten.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace emscripten;

namespace {
    // Helper struct for 2D convolution
    struct Kernel {
        std::vector<double> data;
        int width;
        int height;
    };

    // 2D convolution implementation
    std::vector<double> convolve2D(const std::vector<double>& input, 
                                  int width, int height,
                                  const Kernel& kernel) {
        std::vector<double> output((width - kernel.width + 1) * 
                                 (height - kernel.height + 1));
        
        for (int y = 0; y <= height - kernel.height; y++) {
            for (int x = 0; x <= width - kernel.width; x++) {
                double sum = 0.0;
                
                for (int ky = 0; ky < kernel.height; ky++) {
                    for (int kx = 0; kx < kernel.width; kx++) {
                        int input_idx = (y + ky) * width + (x + kx);
                        int kernel_idx = ky * kernel.width + kx;
                        sum += input[input_idx] * kernel.data[kernel_idx];
                    }
                }
                
                output[y * (width - kernel.width + 1) + x] = sum;
            }
        }
        
        return output;
    }

    // Gaussian blur implementation
    std::vector<double> gaussianBlur(const std::vector<double>& input,
                                   int width, int height,
                                   double sigma) {
        int kernel_size = static_cast<int>(ceil(sigma * 6));
        if (kernel_size % 2 == 0) kernel_size++;
        
        std::vector<double> kernel_data(kernel_size * kernel_size);
        double sum = 0.0;
        int radius = kernel_size / 2;
        
        // Generate 2D Gaussian kernel
        for (int y = -radius; y <= radius; y++) {
            for (int x = -radius; x <= radius; x++) {
                double value = exp(-(x*x + y*y)/(2*sigma*sigma));
                kernel_data[(y + radius) * kernel_size + (x + radius)] = value;
                sum += value;
            }
        }
        
        // Normalize kernel
        for (double& value : kernel_data) {
            value /= sum;
        }
        
        Kernel gaussian_kernel{kernel_data, kernel_size, kernel_size};
        return convolve2D(input, width, height, gaussian_kernel);
    }
}

class FocusAnalyzer {
private:
    // Normalize scores to 0-100 scale, similar to MTF values
    double normalizeScore(double rawScore, double expectedMax) {
        // Apply logarithmic scaling to handle large ranges
        double normalized = 100.0 * (log1p(rawScore) / log1p(expectedMax));
        return std::max(0.0, std::min(100.0, normalized));
    }

    // Convert RGBA to grayscale
    std::vector<double> toGrayscale(const std::vector<uint8_t>& data, int width, int height) {
        std::vector<double> gray(width * height);
        for (int i = 0; i < width * height; i++) {
            int idx = i * 4;  // RGBA format
            gray[i] = 0.299 * data[idx] + 
                     0.587 * data[idx + 1] + 
                     0.114 * data[idx + 2];
        }
        return gray;
    }

public:
    double measureModifiedLaplacian(const std::vector<uint8_t>& data, int width, int height) {
        std::vector<double> gray = this->toGrayscale(data, width, height);
        
        Kernel kernel_x{{0, 0, 0, -1, 2, -1, 0, 0, 0}, 3, 3};
        Kernel kernel_y{{0, -1, 0, 0, 2, 0, 0, -1, 0}, 3, 3};
        
        auto lap_x = convolve2D(gray, width, height, kernel_x);
        auto lap_y = convolve2D(gray, width, height, kernel_y);
        
        double sum = 0.0;
        int count = lap_x.size();
        for (int i = 0; i < count; i++) {
            sum += std::abs(lap_x[i]) + std::abs(lap_y[i]);
        }
        
        double rawScore = sum / count;
        return normalizeScore(rawScore, 1000.0); // Normalized to 0-100 scale
    }

    double measureTenengrad(const std::vector<uint8_t>& data, int width, int height) {
        std::vector<double> gray = this->toGrayscale(data, width, height);
        
        Kernel sobel_x{{-1, 0, 1, -2, 0, 2, -1, 0, 1}, 3, 3};
        Kernel sobel_y{{-1, -2, -1, 0, 0, 0, 1, 2, 1}, 3, 3};
        
        auto gx = convolve2D(gray, width, height, sobel_x);
        auto gy = convolve2D(gray, width, height, sobel_y);
        
        double sum = 0.0;
        int count = gx.size();
        for (int i = 0; i < count; i++) {
            sum += gx[i] * gx[i] + gy[i] * gy[i];
        }
        
        double rawScore = std::sqrt(sum / count);
        return normalizeScore(rawScore, 800.0); // Normalized to 0-100 scale
    }

    double measureEnergyGradient(const std::vector<uint8_t>& data, int width, int height) {
        std::vector<double> gray = this->toGrayscale(data, width, height);
        
        auto gradient1 = gaussianBlur(gray, width, height, 1.0);
        auto gradient2 = gaussianBlur(gray, width, height, 2.0);
        
        double energy1 = 0.0, energy2 = 0.0;
        int count = gradient1.size();
        
        for (int i = 0; i < count; i++) {
            energy1 += gradient1[i] * gradient1[i];
            energy2 += gradient2[i] * gradient2[i];
        }
        
        double rawScore = (0.7 * energy1 + 0.3 * energy2) / count;
        return normalizeScore(rawScore, 2400.0); // Normalized to 0-100 scale
    }

    double measureCombinedFocus(const std::vector<uint8_t>& data, int width, int height) {
        double ml_score = measureModifiedLaplacian(data, width, height);
        double tenengrad_score = measureTenengrad(data, width, height);
        double energy_score = measureEnergyGradient(data, width, height);
        
        // Use weighted average for combined score
        return (0.4 * ml_score + 0.4 * tenengrad_score + 0.2 * energy_score);
    }
};

// Wrapper function that handles the data conversion
val analyzeImage(const val& uint8Array, int width, int height, const std::string& metric) {
    std::vector<uint8_t> data;
    const auto length = uint8Array["length"].as<unsigned>();
    data.resize(length);
    val memoryView{typed_memory_view(length, data.data())};
    memoryView.call<void>("set", uint8Array);

    FocusAnalyzer analyzer;
    val result = val::object();
    
    // Calculate all metrics
    double ml_score = analyzer.measureModifiedLaplacian(data, width, height);
    double tenengrad_score = analyzer.measureTenengrad(data, width, height);
    double energy_score = analyzer.measureEnergyGradient(data, width, height);
    
    // Select primary score based on chosen metric
    double primaryScore;
    if (metric == "Modified Laplacian") {
        primaryScore = ml_score;
    } else if (metric == "Tenengrad") {
        primaryScore = tenengrad_score;
    } else if (metric == "Energy Gradient") {
        primaryScore = energy_score;
    } else {
        primaryScore = (0.4 * ml_score + 0.4 * tenengrad_score + 0.2 * energy_score);
    }
    
    result.set("quality_score", primaryScore);
    result.set("details", val::object());
    result["details"].set("modifiedLaplacian", ml_score);
    result["details"].set("tenengrad", tenengrad_score);
    result["details"].set("energyGradient", energy_score);
    
    return result;
}

EMSCRIPTEN_BINDINGS(focus_metrics) {
    function("analyzeImage", &analyzeImage);
}