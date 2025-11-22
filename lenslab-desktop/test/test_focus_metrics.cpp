/**
 * Focus Metrics unit tests
 */

#include "analysis/FocusMetrics.h"
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace lenslab;

/**
 * Generate a test pattern with variable blur
 */
cv::Mat generateTestPattern(int size, double blurSigma)
{
    cv::Mat pattern(size, size, CV_8UC1);

    // Create checkerboard pattern
    int blockSize = size / 8;
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int bx = x / blockSize;
            int by = y / blockSize;
            pattern.at<uint8_t>(y, x) = ((bx + by) % 2 == 0) ? 200 : 50;
        }
    }

    // Apply Gaussian blur
    if (blurSigma > 0) {
        cv::GaussianBlur(pattern, pattern, cv::Size(0, 0), blurSigma);
    }

    return pattern;
}

bool testBrennerGradient()
{
    std::cout << "Test: Brenner Gradient... ";

    cv::Mat sharp = generateTestPattern(256, 0);
    cv::Mat blurred = generateTestPattern(256, 5);

    double sharpScore = FocusMetrics::brennerGradient(sharp);
    double blurredScore = FocusMetrics::brennerGradient(blurred);

    if (sharpScore <= blurredScore) {
        std::cout << "FAILED" << std::endl;
        return false;
    }

    std::cout << "PASSED (sharp=" << sharpScore << ", blurred=" << blurredScore << ")" << std::endl;
    return true;
}

bool testTenengrad()
{
    std::cout << "Test: Tenengrad... ";

    cv::Mat sharp = generateTestPattern(256, 0);
    cv::Mat blurred = generateTestPattern(256, 5);

    double sharpScore = FocusMetrics::tenengrad(sharp);
    double blurredScore = FocusMetrics::tenengrad(blurred);

    if (sharpScore <= blurredScore) {
        std::cout << "FAILED" << std::endl;
        return false;
    }

    std::cout << "PASSED (sharp=" << sharpScore << ", blurred=" << blurredScore << ")" << std::endl;
    return true;
}

bool testModifiedLaplacian()
{
    std::cout << "Test: Modified Laplacian... ";

    cv::Mat sharp = generateTestPattern(256, 0);
    cv::Mat blurred = generateTestPattern(256, 5);

    double sharpScore = FocusMetrics::modifiedLaplacian(sharp);
    double blurredScore = FocusMetrics::modifiedLaplacian(blurred);

    if (sharpScore <= blurredScore) {
        std::cout << "FAILED" << std::endl;
        return false;
    }

    std::cout << "PASSED (sharp=" << sharpScore << ", blurred=" << blurredScore << ")" << std::endl;
    return true;
}

bool testCombinedScore()
{
    std::cout << "Test: Combined Score... ";

    cv::Mat sharp = generateTestPattern(256, 0);
    cv::Mat blurred = generateTestPattern(256, 5);

    FocusResult sharpResult = FocusMetrics::calculate(sharp);
    FocusResult blurredResult = FocusMetrics::calculate(blurred);

    // Combined score should be in range 0-100
    if (sharpResult.combined < 0 || sharpResult.combined > 100 ||
        blurredResult.combined < 0 || blurredResult.combined > 100) {
        std::cout << "FAILED - Score out of range" << std::endl;
        return false;
    }

    // Sharp should score higher
    if (sharpResult.combined <= blurredResult.combined) {
        std::cout << "FAILED - Sharp should score higher" << std::endl;
        return false;
    }

    std::cout << "PASSED (sharp=" << sharpResult.combined
              << ", blurred=" << blurredResult.combined << ")" << std::endl;
    return true;
}

int main()
{
    std::cout << "\n=== LensLab Focus Metrics Tests ===\n" << std::endl;

    int passed = 0;
    int failed = 0;

    if (testBrennerGradient()) passed++; else failed++;
    if (testTenengrad()) passed++; else failed++;
    if (testModifiedLaplacian()) passed++; else failed++;
    if (testCombinedScore()) passed++; else failed++;

    std::cout << "\n=== Results: " << passed << " passed, "
              << failed << " failed ===" << std::endl;

    return (failed > 0) ? 1 : 0;
}
