// mtf_cursor.cpp

#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

class MTFCursor {
private:
    static constexpr double ANGLE_TOLERANCE = 4.0;
    static constexpr double ROI_WIDTH = 100;
    static constexpr double DEFAULT_GAUSS_SIGMA = 1.5;
    static constexpr double VALIDATION_THRESHOLD = 0.05;

public:
    // Main processing steps for MTF calculation
    static std::vector<double> calculateEdgeProfile(const cv::Mat& roi, const cv::Vec4i& edge_line) {
        std::vector<double> profile;
        double angle = atan2(edge_line[3] - edge_line[1], edge_line[2] - edge_line[0]) * 180 / CV_PI;
        
        // Sample perpendicular to the edge
        for (int y = 0; y < roi.rows; y++) {
            for (int x = 0; x < roi.cols; x++) {
                double dist = pointToLineDistance(cv::Point(x, y), edge_line);
                profile.push_back(roi.at<uchar>(y, x));
            }
        }
        
        return profile;
    }

    static std::vector<double> calculateESF(const std::vector<double>& profile, int num_bins = 100) {
        std::vector<double> esf;
        // Bin the edge profile data
        std::vector<double> binned_values(num_bins, 0.0);
        std::vector<int> bin_counts(num_bins, 0);
        
        // Calculate the bin width
        double bin_width = profile.size() / static_cast<double>(num_bins);
        
        // Bin the values
        for (size_t i = 0; i < profile.size(); ++i) {
            int bin_index = static_cast<int>(i / bin_width);
            if (bin_index >= 0 && bin_index < num_bins) {
                binned_values[bin_index] += profile[i];
                bin_counts[bin_index]++;
            }
        }
        
        // Calculate averages for each bin
        for (int i = 0; i < num_bins; i++) {
            if (bin_counts[i] > 0) {
                esf.push_back(binned_values[i] / bin_counts[i]);
            }
        }
        
        return esf;
    }

    static std::vector<double> calculateLSF(const std::vector<double>& esf) {
        std::vector<double> lsf;
        lsf.reserve(esf.size() - 1);  // Pre-allocate memory
        
        // Calculate first derivative of ESF using central difference
        for (size_t i = 1; i < esf.size() - 1; i++) {
            // Using central difference formula for better accuracy
            lsf.push_back((esf[i+1] - esf[i-1]) / 2.0);
        }
        
        // Apply Gaussian smoothing to reduce noise
        const int window = 3;
        const double sigma = 1.0;
        std::vector<double> kernel(window);
        double kernel_sum = 0.0;
        
        // Create Gaussian kernel
        for (int i = 0; i < window; i++) {
            double x = i - window / 2;
            kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
            kernel_sum += kernel[i];
        }
        
        // Normalize kernel
        for (double& k : kernel) {
            k /= kernel_sum;
        }
        
        // Apply smoothing
        std::vector<double> smoothed_lsf = lsf;
        for (size_t i = window/2; i < lsf.size() - window/2; i++) {
            double sum = 0.0;
            for (int j = 0; j < window; j++) {
                sum += lsf[i + j - window/2] * kernel[j];
            }
            smoothed_lsf[i] = sum;
        }
        
        return smoothed_lsf;
    }

    static std::vector<double> calculateMTF(const std::vector<double>& lsf) {
        std::vector<double> mtf;
        int n = lsf.size();
        
        // Perform FFT on LSF
        cv::Mat lsf_mat(lsf);
        cv::Mat fft_result;
        cv::dft(lsf_mat, fft_result, cv::DFT_COMPLEX_OUTPUT);
        
        // Calculate magnitude spectrum
        std::vector<double> magnitudes;
        for (int i = 0; i < fft_result.rows; i++) {
            double real = fft_result.at<cv::Vec2f>(i)[0];
            double imag = fft_result.at<cv::Vec2f>(i)[1];
            magnitudes.push_back(sqrt(real*real + imag*imag));
        }
        
        // Normalize MTF
        double max_magnitude = *std::max_element(magnitudes.begin(), magnitudes.end());
        for (double mag : magnitudes) {
            mtf.push_back(mag / max_magnitude);
        }
        
        return mtf;
    }

private:
    static double pointToLineDistance(const cv::Point& point, const cv::Vec4i& line) {
        double x0 = point.x;
        double y0 = point.y;
        double x1 = line[0];
        double y1 = line[1];
        double x2 = line[2];
        double y2 = line[3];
        
        return abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / 
               sqrt(pow(y2-y1, 2) + pow(x2-x1, 2));
    }
};
