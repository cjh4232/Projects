#include "MTFAnalyzer.h"
#include "app/Logger.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace lenslab {

MTFAnalyzer::MTFAnalyzer() = default;

MTFResult MTFAnalyzer::analyze(const cv::Mat& image, const ROI& roi)
{
    MTFResult result;

    // Extract ROI
    if (roi.x < 0 || roi.y < 0 ||
        roi.x + roi.width > image.cols ||
        roi.y + roi.height > image.rows) {
        result.errorMessage = "ROI out of bounds";
        return result;
    }

    cv::Mat roiImage = image(cv::Rect(roi.x, roi.y, roi.width, roi.height));

    // Ensure grayscale
    cv::Mat gray;
    if (roiImage.channels() == 3) {
        cv::cvtColor(roiImage, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = roiImage;
    }

    // Step 1: Detect edge
    cv::Vec4i line;
    double angle;
    if (!detectEdge(gray, line, angle)) {
        result.errorMessage = "No suitable edge detected";
        return result;
    }

    // Step 2: Assess quality
    result.quality = assessQuality(gray, line);
    if (!result.quality.isAcceptable) {
        result.errorMessage = result.quality.reason;
        return result;
    }

    // Step 3: Sample ESF (Edge Spread Function)
    std::vector<double> esf = sampleESF(gray, line, angle);
    if (esf.empty()) {
        result.errorMessage = "Failed to sample ESF";
        return result;
    }

    // Step 4: Compute LSF (Line Spread Function)
    std::vector<double> lsf = computeLSF(esf);
    if (lsf.empty()) {
        result.errorMessage = "Failed to compute LSF";
        return result;
    }

    // Step 5: Calculate FWHM
    result.fwhm = calculateFWHM(lsf);

    // Step 6: Compute MTF via FFT
    std::vector<double> mtf = computeMTF(lsf);
    if (mtf.empty()) {
        result.errorMessage = "Failed to compute MTF";
        return result;
    }

    // Generate frequency axis
    result.frequencies.resize(mtf.size());
    for (size_t i = 0; i < mtf.size(); i++) {
        result.frequencies[i] = static_cast<double>(i) / (2.0 * mtf.size());
    }
    result.mtfValues = mtf;

    // Find key metrics
    result.mtf50 = findMTFFrequency(result.frequencies, result.mtfValues, 0.5);
    result.mtf20 = findMTFFrequency(result.frequencies, result.mtfValues, 0.2);
    result.mtf10 = findMTFFrequency(result.frequencies, result.mtfValues, 0.1);

    // Convert to lp/mm if pixel size is set
    if (m_pixelSizeMm > 0) {
        result.mtf50LpMm = result.mtf50 / (2.0 * m_pixelSizeMm);
        result.mtf20LpMm = result.mtf20 / (2.0 * m_pixelSizeMm);
        result.mtf10LpMm = result.mtf10 / (2.0 * m_pixelSizeMm);
    }

    result.valid = true;
    return result;
}

bool MTFAnalyzer::detectEdge(const cv::Mat& roi, cv::Vec4i& line, double& angle)
{
    // Edge detection using Canny
    cv::Mat edges;
    cv::Canny(roi, edges, 50, 150);

    // Hough line detection
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 30, 10);

    if (lines.empty()) {
        return false;
    }

    // Find the longest line within acceptable angle range
    double bestLength = 0;
    bool found = false;

    for (const auto& l : lines) {
        double dx = l[2] - l[0];
        double dy = l[3] - l[1];
        double length = std::sqrt(dx * dx + dy * dy);
        double lineAngle = std::abs(std::atan2(dy, dx) * 180.0 / CV_PI);

        // Normalize to 0-90 range
        if (lineAngle > 90) lineAngle = 180 - lineAngle;

        // Check if angle is suitable for slant-edge analysis
        if (lineAngle >= MIN_EDGE_ANGLE && lineAngle <= MAX_EDGE_ANGLE) {
            if (length > bestLength) {
                bestLength = length;
                line = l;
                angle = lineAngle;
                found = true;
            }
        }
    }

    return found;
}

std::vector<double> MTFAnalyzer::sampleESF(const cv::Mat& roi, const cv::Vec4i& line, double angle)
{
    // TODO: Implement proper super-resolution ESF sampling
    // This is a simplified version - full implementation would match mtf_analyzer_6.cpp

    const int NUM_BINS = 256;
    std::vector<double> esf(NUM_BINS, 0);
    std::vector<int> counts(NUM_BINS, 0);

    // Calculate line parameters
    double dx = line[2] - line[0];
    double dy = line[3] - line[1];
    double length = std::sqrt(dx * dx + dy * dy);

    // Normal direction (perpendicular to edge)
    double nx = -dy / length;
    double ny = dx / length;

    // Line midpoint
    double cx = (line[0] + line[2]) / 2.0;
    double cy = (line[1] + line[3]) / 2.0;

    // Sample across the edge
    for (int y = 0; y < roi.rows; y++) {
        for (int x = 0; x < roi.cols; x++) {
            // Distance from line (perpendicular)
            double dist = (x - cx) * nx + (y - cy) * ny;

            // Map to bin index
            int bin = static_cast<int>((dist + NUM_BINS / 2.0) * 2);
            if (bin >= 0 && bin < NUM_BINS) {
                esf[bin] += roi.at<uint8_t>(y, x);
                counts[bin]++;
            }
        }
    }

    // Average bins
    for (int i = 0; i < NUM_BINS; i++) {
        if (counts[i] > 0) {
            esf[i] /= counts[i];
        }
    }

    return esf;
}

std::vector<double> MTFAnalyzer::computeLSF(const std::vector<double>& esf)
{
    if (esf.size() < 3) return {};

    std::vector<double> lsf(esf.size() - 1);

    // Derivative of ESF
    for (size_t i = 0; i < lsf.size(); i++) {
        lsf[i] = esf[i + 1] - esf[i];
    }

    return lsf;
}

std::vector<double> MTFAnalyzer::computeMTF(const std::vector<double>& lsf)
{
    if (lsf.empty()) return {};

    // Apply Hanning window
    std::vector<double> windowed(lsf.size());
    for (size_t i = 0; i < lsf.size(); i++) {
        double window = 0.5 * (1.0 - std::cos(2.0 * CV_PI * i / (lsf.size() - 1)));
        windowed[i] = lsf[i] * window;
    }

    // DFT using OpenCV
    cv::Mat input(1, static_cast<int>(windowed.size()), CV_64F, windowed.data());
    cv::Mat output;
    cv::dft(input, output, cv::DFT_COMPLEX_OUTPUT);

    // Compute magnitude (MTF)
    std::vector<double> mtf(output.cols / 2);
    for (int i = 0; i < static_cast<int>(mtf.size()); i++) {
        double re = output.at<double>(0, 2 * i);
        double im = output.at<double>(0, 2 * i + 1);
        mtf[i] = std::sqrt(re * re + im * im);
    }

    // Normalize to DC = 1.0
    if (mtf[0] > 0) {
        double dc = mtf[0];
        for (auto& v : mtf) {
            v /= dc;
        }
    }

    return mtf;
}

ROIQuality MTFAnalyzer::assessQuality(const cv::Mat& roi, const cv::Vec4i& line)
{
    ROIQuality quality;

    // Calculate edge strength (contrast across edge)
    cv::Mat sobelX, sobelY;
    cv::Sobel(roi, sobelX, CV_64F, 1, 0);
    cv::Sobel(roi, sobelY, CV_64F, 0, 1);

    cv::Mat magnitude;
    cv::magnitude(sobelX, sobelY, magnitude);

    double maxMag;
    cv::minMaxLoc(magnitude, nullptr, &maxMag);
    quality.edgeStrength = std::min(100.0, maxMag / 2.55);  // Normalize to 0-100

    // Calculate noise level (std dev in flat regions)
    cv::Mat meanImg, stdImg;
    cv::meanStdDev(roi, meanImg, stdImg);
    quality.noiseLevel = stdImg.at<double>(0);

    // Linearity score (based on Hough line detection confidence)
    quality.linearityScore = 80.0;  // TODO: Implement proper linearity check

    // Overall score
    quality.overallScore = 0.4 * quality.edgeStrength +
                          0.4 * quality.linearityScore +
                          0.2 * (100.0 - quality.noiseLevel);

    quality.isAcceptable = quality.overallScore >= QUALITY_THRESHOLD;
    if (!quality.isAcceptable) {
        quality.reason = "Quality score too low: " + std::to_string(quality.overallScore);
    }

    return quality;
}

double MTFAnalyzer::findMTFFrequency(const std::vector<double>& frequencies,
                                    const std::vector<double>& mtf,
                                    double targetValue)
{
    for (size_t i = 1; i < mtf.size(); i++) {
        if (mtf[i] <= targetValue && mtf[i - 1] > targetValue) {
            // Linear interpolation
            double t = (targetValue - mtf[i]) / (mtf[i - 1] - mtf[i]);
            return frequencies[i] - t * (frequencies[i] - frequencies[i - 1]);
        }
    }
    return 0.0;  // Not found
}

double MTFAnalyzer::calculateFWHM(const std::vector<double>& lsf)
{
    if (lsf.empty()) return 0.0;

    double maxVal = *std::max_element(lsf.begin(), lsf.end());
    double halfMax = maxVal / 2.0;

    // Find left crossing
    double leftX = 0;
    for (size_t i = 1; i < lsf.size(); i++) {
        if (lsf[i] >= halfMax && lsf[i - 1] < halfMax) {
            double t = (halfMax - lsf[i - 1]) / (lsf[i] - lsf[i - 1]);
            leftX = (i - 1) + t;
            break;
        }
    }

    // Find right crossing
    double rightX = lsf.size() - 1;
    for (size_t i = lsf.size() - 1; i > 0; i--) {
        if (lsf[i - 1] >= halfMax && lsf[i] < halfMax) {
            double t = (halfMax - lsf[i]) / (lsf[i - 1] - lsf[i]);
            rightX = i - t;
            break;
        }
    }

    return rightX - leftX;
}

} // namespace lenslab
