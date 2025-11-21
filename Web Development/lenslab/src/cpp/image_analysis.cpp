#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <emscripten/emscripten.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace emscripten;

// Helper functions for image processing
namespace {
    // Convert RGBA to grayscale
    std::vector<uint8_t> toGrayscale(const uint8_t* data, int width, int height) {
        std::vector<uint8_t> gray(width * height);
        for (int i = 0; i < width * height; i++) {
            // Convert RGBA to grayscale using luminance formula
            gray[i] = static_cast<uint8_t>(
                0.299 * data[i * 4] +     // R
                0.587 * data[i * 4 + 1] + // G
                0.114 * data[i * 4 + 2]   // B
            );
        }
        return gray;
    }

    // Calculate Sobel gradients
    void calculateSobel(const std::vector<uint8_t>& gray, int width, int height,
                       std::vector<double>& gx, std::vector<double>& gy) {
        gx.resize((width - 2) * (height - 2));
        gy.resize((width - 2) * (height - 2));

        // Sobel kernels
        const int kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        const int ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                double sx = 0, sy = 0;
                
                for (int ky_idx = -1; ky_idx <= 1; ky_idx++) {
                    for (int kx_idx = -1; kx_idx <= 1; kx_idx++) {
                        int pixel = gray[(y + ky_idx) * width + (x + kx_idx)];
                        sx += pixel * kx[ky_idx + 1][kx_idx + 1];
                        sy += pixel * ky[ky_idx + 1][kx_idx + 1];
                    }
                }

                int idx = (y - 1) * (width - 2) + (x - 1);
                gx[idx] = sx;
                gy[idx] = sy;
            }
        }
    }
}

// Use a wrapper class to handle the image data
class ImageAnalyzer {
public:
    double measureSharpness(uintptr_t ptr, int width, int height) {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(ptr);
        std::vector<uint8_t> gray = toGrayscale(data, width, height);
        std::vector<double> gx, gy;
        calculateSobel(gray, width, height, gx, gy);

        // Calculate magnitude of gradients
        double sum = 0.0;
        double sum_squared = 0.0;
        int count = gx.size();

        for (int i = 0; i < count; i++) {
            double magnitude = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
            sum += magnitude;
            sum_squared += magnitude * magnitude;
        }

        // Calculate variance
        double mean = sum / count;
        double variance = (sum_squared / count) - (mean * mean);
        
        return std::max(1.0, variance / 100.0);
    }

    double measureContrast(uintptr_t ptr, int width, int height) {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(ptr);
        std::vector<uint8_t> gray = toGrayscale(data, width, height);
        
        // Calculate mean
        double sum = 0.0;
        for (uint8_t value : gray) {
            sum += value;
        }
        double mean = sum / gray.size();

        // Calculate standard deviation
        double sum_squared_diff = 0.0;
        for (uint8_t value : gray) {
            double diff = value - mean;
            sum_squared_diff += diff * diff;
        }
        double std_dev = std::sqrt(sum_squared_diff / gray.size());

        return std::max(1.0, std_dev);
    }

    double measureBrightness(uintptr_t ptr, int width, int height) {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(ptr);
        double sum = 0.0;
        int count = width * height;

        // Calculate average RGB values
        for (int i = 0; i < count; i++) {
            int idx = i * 4;
            sum += (data[idx] + data[idx + 1] + data[idx + 2]) / 3.0;
        }

        return std::max(1.0, sum / count);
    }
};

// Binding code
EMSCRIPTEN_BINDINGS(image_analysis) {
    class_<ImageAnalyzer>("ImageAnalyzer")
        .constructor<>()
        .function("measureSharpness", &ImageAnalyzer::measureSharpness)
        .function("measureContrast", &ImageAnalyzer::measureContrast)
        .function("measureBrightness", &ImageAnalyzer::measureBrightness)
        ;
}