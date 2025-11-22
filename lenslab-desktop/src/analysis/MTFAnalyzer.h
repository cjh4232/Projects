#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>

namespace lenslab {

/**
 * Region of Interest for analysis
 */
struct ROI
{
    int x = 0;
    int y = 0;
    int width = 100;
    int height = 100;
    std::string label;
};

/**
 * Quality assessment for an ROI
 */
struct ROIQuality
{
    double edgeStrength = 0.0;     // Edge contrast (0-100)
    double linearityScore = 0.0;   // Edge straightness (0-100)
    double noiseLevel = 0.0;       // Noise assessment (lower is better)
    double overallScore = 0.0;     // Combined score (0-100)
    bool isAcceptable = false;     // Meets quality threshold
    std::string reason;            // Quality explanation
};

/**
 * Results from MTF analysis
 */
struct MTFResult
{
    // MTF curve data
    std::vector<double> frequencies;   // cycles/pixel
    std::vector<double> mtfValues;     // MTF at each frequency

    // Key metrics (cycles/pixel)
    double mtf50 = 0.0;    // Frequency at 50% contrast
    double mtf20 = 0.0;    // Frequency at 20% contrast
    double mtf10 = 0.0;    // Frequency at 10% contrast

    // Key metrics (lp/mm) - requires pixel size
    double mtf50LpMm = 0.0;
    double mtf20LpMm = 0.0;
    double mtf10LpMm = 0.0;

    // FWHM
    double fwhm = 0.0;     // Full Width at Half Maximum (pixels)

    // Quality
    ROIQuality quality;

    // Status
    bool valid = false;
    std::string errorMessage;
};

/**
 * MTF Analyzer - ISO 12233 compliant slant-edge analysis
 *
 * Ported from LensLab mtf_analyzer_6.cpp
 */
class MTFAnalyzer
{
public:
    MTFAnalyzer();

    /**
     * Analyze an image region for MTF
     * @param image Full image (grayscale)
     * @param roi Region to analyze
     * @return MTF analysis results
     */
    MTFResult analyze(const cv::Mat& image, const ROI& roi);

    /**
     * Set pixel size for lp/mm calculations
     * @param pixelSizeMm Pixel size in millimeters
     */
    void setPixelSize(double pixelSizeMm) { m_pixelSizeMm = pixelSizeMm; }

    /**
     * Enable/disable debug output
     */
    void setDebugMode(bool enabled) { m_debugMode = enabled; }

private:
    // Analysis pipeline stages
    bool detectEdge(const cv::Mat& roi, cv::Vec4i& line, double& angle);
    std::vector<double> sampleESF(const cv::Mat& roi, const cv::Vec4i& line, double angle);
    std::vector<double> computeLSF(const std::vector<double>& esf);
    std::vector<double> computeMTF(const std::vector<double>& lsf);
    ROIQuality assessQuality(const cv::Mat& roi, const cv::Vec4i& line);

    // Utility functions
    double findMTFFrequency(const std::vector<double>& frequencies,
                           const std::vector<double>& mtf,
                           double targetValue);
    double calculateFWHM(const std::vector<double>& lsf);

    // Configuration
    double m_pixelSizeMm = 0.0;  // 0 = not set
    bool m_debugMode = false;

    // Analysis parameters
    static constexpr double MIN_EDGE_ANGLE = 5.0;   // degrees
    static constexpr double MAX_EDGE_ANGLE = 20.0;  // degrees
    static constexpr double OPTIMAL_ANGLE_MIN = 8.0;
    static constexpr double OPTIMAL_ANGLE_MAX = 15.0;
    static constexpr double QUALITY_THRESHOLD = 50.0;
};

} // namespace lenslab
