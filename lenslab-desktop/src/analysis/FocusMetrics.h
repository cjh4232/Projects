#pragma once

#include <opencv2/core.hpp>

namespace lenslab {

/**
 * Focus quality metrics for real-time focus assessment
 */
struct FocusResult
{
    double brenner = 0.0;          // Brenner gradient
    double tenengrad = 0.0;        // Tenengrad (Sobel magnitude)
    double modifiedLaplacian = 0.0; // Modified Laplacian
    double combined = 0.0;          // Weighted combination (0-100)
};

class FocusMetrics
{
public:
    /**
     * Calculate all focus metrics for an image region
     * @param image Grayscale image (CV_8UC1)
     * @return FocusResult with all metrics
     */
    static FocusResult calculate(const cv::Mat& image);

    /**
     * Calculate Brenner gradient
     * Sum of squared differences between pixels separated by 2
     */
    static double brennerGradient(const cv::Mat& image);

    /**
     * Calculate Tenengrad metric
     * Sum of squared Sobel gradient magnitudes
     */
    static double tenengrad(const cv::Mat& image);

    /**
     * Calculate Modified Laplacian
     * Sum of absolute Laplacian responses in x and y
     */
    static double modifiedLaplacian(const cv::Mat& image);

    /**
     * Normalize a raw metric to 0-100 scale
     */
    static double normalize(double rawScore, double expectedMax);

private:
    // Normalization constants (empirically determined)
    static constexpr double BRENNER_MAX = 5000.0;
    static constexpr double TENENGRAD_MAX = 800.0;
    static constexpr double LAPLACIAN_MAX = 1000.0;

    // Weights for combined score
    static constexpr double BRENNER_WEIGHT = 0.2;
    static constexpr double TENENGRAD_WEIGHT = 0.4;
    static constexpr double LAPLACIAN_WEIGHT = 0.4;
};

} // namespace lenslab
