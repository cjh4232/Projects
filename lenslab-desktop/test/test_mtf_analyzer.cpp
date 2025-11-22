/**
 * MTF Analyzer unit tests
 */

#include "analysis/MTFAnalyzer.h"
#include "analysis/FocusMetrics.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace lenslab;

/**
 * Generate a synthetic slant edge test pattern
 */
cv::Mat generateSlantEdge(int width, int height, double angle_deg, double blur_sigma)
{
    cv::Mat image(height, width, CV_8UC1);

    double angle_rad = angle_deg * CV_PI / 180.0;
    double slope = std::tan(angle_rad);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Edge position at this y
            double edgeX = width / 2.0 + slope * (y - height / 2.0);

            // Distance from edge
            double dist = x - edgeX;

            // Smooth edge using error function approximation
            double value;
            if (blur_sigma > 0) {
                value = 0.5 * (1.0 + std::tanh(dist / (blur_sigma * std::sqrt(2.0))));
            } else {
                value = (dist > 0) ? 1.0 : 0.0;
            }

            image.at<uint8_t>(y, x) = static_cast<uint8_t>(value * 200 + 28);
        }
    }

    return image;
}

bool testEdgeDetection()
{
    std::cout << "Test: Edge Detection... ";

    MTFAnalyzer analyzer;
    cv::Mat image = generateSlantEdge(200, 200, 10.0, 2.0);

    ROI roi{50, 50, 100, 100, "test"};
    MTFResult result = analyzer.analyze(image, roi);

    if (!result.valid) {
        std::cout << "FAILED - " << result.errorMessage << std::endl;
        return false;
    }

    std::cout << "PASSED (MTF50=" << result.mtf50 << ")" << std::endl;
    return true;
}

bool testMTFRange()
{
    std::cout << "Test: MTF Range Validation... ";

    MTFAnalyzer analyzer;

    // Test with different blur levels - MTF should decrease with more blur
    double prevMTF50 = 1.0;

    for (double sigma : {1.0, 2.0, 3.0, 4.0}) {
        cv::Mat image = generateSlantEdge(200, 200, 10.0, sigma);
        ROI roi{50, 50, 100, 100, "test"};
        MTFResult result = analyzer.analyze(image, roi);

        if (!result.valid) {
            std::cout << "FAILED at sigma=" << sigma << std::endl;
            return false;
        }

        if (result.mtf50 >= prevMTF50) {
            std::cout << "FAILED - MTF should decrease with blur" << std::endl;
            return false;
        }

        prevMTF50 = result.mtf50;
    }

    std::cout << "PASSED" << std::endl;
    return true;
}

bool testFocusMetrics()
{
    std::cout << "Test: Focus Metrics... ";

    // Sharp image should have higher focus score than blurred
    cv::Mat sharp = generateSlantEdge(200, 200, 10.0, 1.0);
    cv::Mat blurred = generateSlantEdge(200, 200, 10.0, 5.0);

    FocusResult sharpResult = FocusMetrics::calculate(sharp);
    FocusResult blurredResult = FocusMetrics::calculate(blurred);

    if (sharpResult.combined <= blurredResult.combined) {
        std::cout << "FAILED - Sharp image should score higher" << std::endl;
        return false;
    }

    std::cout << "PASSED (sharp=" << sharpResult.combined
              << ", blurred=" << blurredResult.combined << ")" << std::endl;
    return true;
}

int main()
{
    std::cout << "\n=== LensLab MTF Analyzer Tests ===\n" << std::endl;

    int passed = 0;
    int failed = 0;

    if (testEdgeDetection()) passed++; else failed++;
    if (testMTFRange()) passed++; else failed++;
    if (testFocusMetrics()) passed++; else failed++;

    std::cout << "\n=== Results: " << passed << " passed, "
              << failed << " failed ===" << std::endl;

    return (failed > 0) ? 1 : 0;
}
