#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <emscripten/emscripten.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

using namespace emscripten;

namespace focus_metrics {

struct ConvolutionKernel {
    std::vector<double> data;
    int width;
    int height;
};

class FocusMetric {
protected:
    static std::vector<double> convolve2D(const std::vector<double>& input, 
                                        int width, int height,
                                        const ConvolutionKernel& kernel) {
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

    static std::vector<double> gaussianBlur(const std::vector<double>& input,
                                          int width, int height,
                                          double sigma) {
        int kernel_size = static_cast<int>(ceil(sigma * 6));
        if (kernel_size % 2 == 0) kernel_size++;
        
        std::vector<double> kernel_data(kernel_size * kernel_size);
        double sum = 0.0;
        int radius = kernel_size / 2;
        
        for (int y = -radius; y <= radius; y++) {
            for (int x = -radius; x <= radius; x++) {
                double value = exp(-(x*x + y*y)/(2*sigma*sigma));
                kernel_data[(y + radius) * kernel_size + (x + radius)] = value;
                sum += value;
            }
        }
        
        for (double& value : kernel_data) {
            value /= sum;
        }
        
        ConvolutionKernel gaussian_kernel{kernel_data, kernel_size, kernel_size};
        return convolve2D(input, width, height, gaussian_kernel);
    }

    static std::vector<double> toGrayscale(const std::vector<uint8_t>& data, 
                                         int width, int height) {
        std::vector<double> gray(width * height);
        for (int i = 0; i < width * height; i++) {
            int idx = i * 4;
            gray[i] = 0.299 * data[idx] + 
                     0.587 * data[idx + 1] + 
                     0.114 * data[idx + 2];
        }
        return gray;
    }

    static double normalizeScore(double rawScore, double expectedMax) {
        double normalized = 100.0 * (log1p(rawScore) / log1p(expectedMax));
        return std::max(0.0, std::min(100.0, normalized));
    }
};

class ModifiedLaplacian : public FocusMetric {
public:
    static double measure(const std::vector<uint8_t>& data, int width, int height) {
        std::vector<double> gray = toGrayscale(data, width, height);
        
        ConvolutionKernel kernel_x{{0, 0, 0, -1, 2, -1, 0, 0, 0}, 3, 3};
        ConvolutionKernel kernel_y{{0, -1, 0, 0, 2, 0, 0, -1, 0}, 3, 3};
        
        auto lap_x = convolve2D(gray, width, height, kernel_x);
        auto lap_y = convolve2D(gray, width, height, kernel_y);
        
        double sum = 0.0;
        for (size_t i = 0; i < lap_x.size(); i++) {
            sum += std::abs(lap_x[i]) + std::abs(lap_y[i]);
        }
        
        return normalizeScore(sum / lap_x.size(), 1000.0);
    }
};

class Tenengrad : public FocusMetric {
public:
    static double measure(const std::vector<uint8_t>& data, int width, int height) {
        std::vector<double> gray = toGrayscale(data, width, height);
        
        ConvolutionKernel sobel_x{{-1, 0, 1, -2, 0, 2, -1, 0, 1}, 3, 3};
        ConvolutionKernel sobel_y{{-1, -2, -1, 0, 0, 0, 1, 2, 1}, 3, 3};
        
        auto gx = convolve2D(gray, width, height, sobel_x);
        auto gy = convolve2D(gray, width, height, sobel_y);
        
        double sum = 0.0;
        for (size_t i = 0; i < gx.size(); i++) {
            sum += gx[i] * gx[i] + gy[i] * gy[i];
        }
        
        return normalizeScore(std::sqrt(sum / gx.size()), 800.0);
    }
};

class BrennerGradient : public FocusMetric {
public:
    static double measure(const std::vector<uint8_t>& data, int width, int height) {
        std::vector<double> gray = toGrayscale(data, width, height);
        double sum = 0.0;
        
        // Horizontal gradient
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width - 2; x++) {
                double diff = gray[y * width + x] - gray[y * width + (x + 2)];
                sum += diff * diff;
            }
        }
        
        // Vertical gradient
        for (int y = 0; y < height - 2; y++) {
            for (int x = 0; x < width; x++) {
                double diff = gray[y * width + x] - gray[(y + 2) * width + x];
                sum += diff * diff;
            }
        }
        
        // More aggressive normalization to prevent saturation
        return normalizeScore(sum / (width * height), 5000.0);
    }
};

struct AnalysisResult {
    double quality_score;
    double ml_score;
    double tenengrad_score;
    double brenner_score;
};

class FocusAnalyzer {
public:
    static AnalysisResult analyze(const std::vector<uint8_t>& data, 
                                int width, int height, 
                                const std::string& metric) {
        AnalysisResult result;
        
        result.ml_score = ModifiedLaplacian::measure(data, width, height);
        result.tenengrad_score = Tenengrad::measure(data, width, height);
        result.brenner_score = BrennerGradient::measure(data, width, height);
        
        if (metric == "Modified Laplacian") {
            result.quality_score = result.ml_score;
        } else if (metric == "Tenengrad") {
            result.quality_score = result.tenengrad_score;
        } else if (metric == "Brenner Gradient") {
            result.quality_score = result.brenner_score;
        } else {
            result.quality_score = (0.4 * result.ml_score + 
                                  0.4 * result.tenengrad_score + 
                                  0.2 * result.brenner_score);
        }
        
        return result;
    }
};

} // namespace focus_metrics

val analyzeImage(const val& uint8Array, int width, int height, const std::string& metric) {
    std::vector<uint8_t> data;
    const auto length = uint8Array["length"].as<unsigned>();
    data.resize(length);
    val memoryView{typed_memory_view(length, data.data())};
    memoryView.call<void>("set", uint8Array);

    auto result = focus_metrics::FocusAnalyzer::analyze(data, width, height, metric);
    
    val output = val::object();
    output.set("quality_score", result.quality_score);
    output.set("details", val::object());
    output["details"].set("modifiedLaplacian", result.ml_score);
    output["details"].set("tenengrad", result.tenengrad_score);
    output["details"].set("brennerGradient", result.brenner_score);
    
    return output;
}

EMSCRIPTEN_BINDINGS(focus_metrics) {
    function("analyzeImage", &analyzeImage);
}