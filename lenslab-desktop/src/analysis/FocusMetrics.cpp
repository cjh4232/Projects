#include "FocusMetrics.h"
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace lenslab {

FocusResult FocusMetrics::calculate(const cv::Mat& image)
{
    FocusResult result;

    if (image.empty() || image.type() != CV_8UC1) {
        return result;
    }

    // Calculate raw metrics
    double brennerRaw = brennerGradient(image);
    double tenengradRaw = tenengrad(image);
    double laplacianRaw = modifiedLaplacian(image);

    // Normalize to 0-100
    result.brenner = normalize(brennerRaw, BRENNER_MAX);
    result.tenengrad = normalize(tenengradRaw, TENENGRAD_MAX);
    result.modifiedLaplacian = normalize(laplacianRaw, LAPLACIAN_MAX);

    // Weighted combination
    result.combined = BRENNER_WEIGHT * result.brenner +
                      TENENGRAD_WEIGHT * result.tenengrad +
                      LAPLACIAN_WEIGHT * result.modifiedLaplacian;

    return result;
}

double FocusMetrics::brennerGradient(const cv::Mat& image)
{
    double sum = 0.0;
    int width = image.cols;
    int height = image.rows;

    // Horizontal gradient
    for (int y = 0; y < height; y++) {
        const uint8_t* row = image.ptr<uint8_t>(y);
        for (int x = 0; x < width - 2; x++) {
            double diff = static_cast<double>(row[x]) - static_cast<double>(row[x + 2]);
            sum += diff * diff;
        }
    }

    // Vertical gradient
    for (int y = 0; y < height - 2; y++) {
        const uint8_t* row0 = image.ptr<uint8_t>(y);
        const uint8_t* row2 = image.ptr<uint8_t>(y + 2);
        for (int x = 0; x < width; x++) {
            double diff = static_cast<double>(row0[x]) - static_cast<double>(row2[x]);
            sum += diff * diff;
        }
    }

    // Normalize by image size
    return sum / (width * height);
}

double FocusMetrics::tenengrad(const cv::Mat& image)
{
    cv::Mat gx, gy;

    // Sobel gradients
    cv::Sobel(image, gx, CV_64F, 1, 0, 3);
    cv::Sobel(image, gy, CV_64F, 0, 1, 3);

    // Sum of squared magnitudes
    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);

    double sum = cv::sum(magnitude.mul(magnitude))[0];

    // Normalize by image size
    return std::sqrt(sum / (image.cols * image.rows));
}

double FocusMetrics::modifiedLaplacian(const cv::Mat& image)
{
    // Laplacian kernels for x and y
    cv::Mat kernelX = (cv::Mat_<double>(3, 3) <<
        0, 0, 0,
        -1, 2, -1,
        0, 0, 0);

    cv::Mat kernelY = (cv::Mat_<double>(3, 3) <<
        0, -1, 0,
        0, 2, 0,
        0, -1, 0);

    cv::Mat lapX, lapY;
    cv::filter2D(image, lapX, CV_64F, kernelX);
    cv::filter2D(image, lapY, CV_64F, kernelY);

    // Sum of absolute values
    cv::Mat absLapX = cv::abs(lapX);
    cv::Mat absLapY = cv::abs(lapY);

    double sum = cv::sum(absLapX + absLapY)[0];

    // Normalize by image size
    return sum / (image.cols * image.rows);
}

double FocusMetrics::normalize(double rawScore, double expectedMax)
{
    // Logarithmic normalization to handle wide range
    double normalized = 100.0 * (std::log1p(rawScore) / std::log1p(expectedMax));
    return std::clamp(normalized, 0.0, 100.0);
}

} // namespace lenslab
