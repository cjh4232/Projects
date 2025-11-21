/**
 * @namespace focus_metrics
 * @brief Contains implementations of various focus measurement algorithms
 * 
 * This namespace encapsulates multiple focus measurement techniques including
 * Modified Laplacian, Tenengrad, and Brenner Gradient methods. Each algorithm
 * is implemented as a separate class inheriting from the FocusMetric base class.
 */

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <emscripten/emscripten.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <complex>



using namespace emscripten;

namespace focus_metrics {

struct EdgeInfo {
    std::vector<double> magnitude;
    double angle;
    bool valid;
    std::vector<size_t> pixels;  // Store pixels for visualization
    double mtf_score;            // Store MTF score when calculated
    
    EdgeInfo() : angle(0.0), valid(false), mtf_score(0.0) {}  // Default constructor
};

struct EdgeCandidate {
    double angle;
    double magnitude;
    double length;  // Length as percentage of ROI dimension
    std::vector<size_t> pixels;
};

struct AnalysisResult {
    double quality_score;
    double ml_score;
    double tenengrad_score;
    double brenner_score;
    bool valid_mtf;
    std::string mtf_error;
};

struct ConvolutionKernel {
    std::vector<double> data;
    int width;
    int height;
};  

/**
 * @class FocusMetric
 * @brief Base class providing common utilities for focus measurement
 * 
 * Contains protected static methods for image processing operations including
 * convolution, Gaussian blur, grayscale conversion, and score normalization.
 */

class FocusMetric {
protected:
    static std::vector<double> convolve2D(const std::vector<double>& input, 
                                        int width, int height,
                                        const ConvolutionKernel& kernel) {
        // Create input Mat
        cv::Mat input_mat(height, width, CV_64F, const_cast<double*>(input.data()));
        
        // Create kernel Mat
        cv::Mat kernel_mat(kernel.height, kernel.width, CV_64F, 
                          const_cast<double*>(kernel.data.data()));
        
        // Output Mat
        cv::Mat output_mat;
        
        // Perform convolution
        cv::filter2D(input_mat, output_mat, CV_64F, kernel_mat);
        
        // Convert back to vector
        std::vector<double> output((width - kernel.width + 1) * 
                                 (height - kernel.height + 1));
        memcpy(output.data(), output_mat.data, output.size() * sizeof(double));
        
        return output;
    }

    static std::vector<double> gaussianBlur(const std::vector<double>& input,
                                          int width, int height,
                                          double sigma) {
        // Create input Mat
        cv::Mat input_mat(height, width, CV_64F, const_cast<double*>(input.data()));
        
        // Calculate kernel size (same as original)
        int kernel_size = static_cast<int>(ceil(sigma * 6));
        if (kernel_size % 2 == 0) kernel_size++;
        
        // Output Mat
        cv::Mat output_mat;
        
        // Perform Gaussian blur
        cv::GaussianBlur(input_mat, output_mat, cv::Size(kernel_size, kernel_size), 
                        sigma, sigma);
        
        // Convert back to vector
        std::vector<double> output(width * height);
        memcpy(output.data(), output_mat.data, output.size() * sizeof(double));
        
        return output;
    }

    static std::vector<double> toGrayscale(const std::vector<uint8_t>& data, 
                                         int width, int height) {
        // Create cv::Mat from input data (assuming RGBA format)
        cv::Mat color_mat(height, width, CV_8UC4, const_cast<uint8_t*>(data.data()));
        
        // Create grayscale Mat
        cv::Mat gray_mat;
        
        // Convert RGBA to grayscale
        cv::cvtColor(color_mat, gray_mat, cv::COLOR_RGBA2GRAY);
        
        // Convert to double precision
        cv::Mat double_mat;
        gray_mat.convertTo(double_mat, CV_64F);
        
        // Convert to vector
        std::vector<double> gray(width * height);
        memcpy(gray.data(), double_mat.data, width * height * sizeof(double));
        
        return gray;
    }

    static double normalizeScore(double rawScore, double expectedMax) {
        // This function is fine as is - it's simple scalar math
        static const double LOG_FACTOR = 100.0 / log1p(expectedMax);
        double normalized = log1p(rawScore) * LOG_FACTOR;
        return std::max(0.0, std::min(100.0, normalized));
    }
};


/**
 * @class ModifiedLaplacian
 * @brief Implements the Modified Laplacian focus measurement algorithm
 * 
 * Measures image focus by calculating second-order derivatives in both
 * horizontal and vertical directions using specialized kernels.
 */
class ModifiedLaplacian : public FocusMetric {
public:
    static double measure(const std::vector<uint8_t>& data, int width, int height) {
        // Convert input to grayscale using OpenCV
        cv::Mat gray = cv::Mat(height, width, CV_8UC4, const_cast<uint8_t*>(data.data()));
        cv::Mat gray_single;
        cv::cvtColor(gray, gray_single, cv::COLOR_RGBA2GRAY);
        
        // Convert to float for better precision
        cv::Mat float_img;
        gray_single.convertTo(float_img, CV_32F);
        
        // Apply Laplacian in x and y directions
        cv::Mat lap_x, lap_y;
        cv::Matx13f kernel_x(0, -1, 0);  // Modified Laplacian kernels
        cv::Matx31f kernel_y(0, -1, 0);
        
        cv::filter2D(float_img, lap_x, CV_32F, kernel_x);
        cv::filter2D(float_img, lap_y, CV_32F, kernel_y);
        
        // Calculate absolute values and sum
        cv::Mat abs_lap_x, abs_lap_y;
        cv::abs(lap_x, abs_lap_x);
        cv::abs(lap_y, abs_lap_y);
        
        // Sum of absolute values
        cv::Scalar sum = cv::sum(abs_lap_x + abs_lap_y);
        double total = sum[0] / (width * height);
        
        return normalizeScore(total, 1000.0);
    }
};


/**
 * @class Tenegrad
 * @brief Implements the Tenengrad focus measurement algorithm
 * 
 * Measures image focus by using Sobel operators to compute
 * image gradients, then calculating the gradient magnitude..
 */
class Tenengrad : public FocusMetric {
public:
    static double measure(const std::vector<uint8_t>& data, int width, int height) {
        // Convert input to grayscale using OpenCV
        cv::Mat gray = cv::Mat(height, width, CV_8UC4, const_cast<uint8_t*>(data.data()));
        cv::Mat gray_single;
        cv::cvtColor(gray, gray_single, cv::COLOR_RGBA2GRAY);
        
        // Convert to float
        cv::Mat float_img;
        gray_single.convertTo(float_img, CV_32F);
        
        // Calculate Sobel gradients
        cv::Mat grad_x, grad_y;
        cv::Sobel(float_img, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(float_img, grad_y, CV_32F, 0, 1, 3);
        
        // Calculate gradient magnitude squared
        cv::Mat grad_mag;
        cv::multiply(grad_x, grad_x, grad_x);
        cv::multiply(grad_y, grad_y, grad_y);
        grad_mag = grad_x + grad_y;
        
        // Calculate mean gradient magnitude
        cv::Scalar mean = cv::mean(grad_mag);
        double total = std::sqrt(mean[0]);
        
        return normalizeScore(total, 800.0);
    }
};


/**
 * @class BrennerGradient
 * @brief Implements the Brenner Gradient focus measurement algorithm
 * 
 * Measures image focus by measuring squared differences between
 * pixels separated by two positions.
 */
class BrennerGradient : public FocusMetric {
public:
    static double measure(const std::vector<uint8_t>& data, int width, int height) {
        // Convert input to grayscale using OpenCV
        cv::Mat gray = cv::Mat(height, width, CV_8UC4, const_cast<uint8_t*>(data.data()));
        cv::Mat gray_single;
        cv::cvtColor(gray, gray_single, cv::COLOR_RGBA2GRAY);
        
        // Convert to float
        cv::Mat float_img;
        gray_single.convertTo(float_img, CV_32F);
        
        // Create horizontal and vertical gradient images
        cv::Mat grad_x, grad_y;
        
        // Custom kernels for Brenner gradient (2-pixel difference)
        cv::Mat kernel_x = (cv::Mat_<float>(1, 3) << -1, 0, 1);
        cv::Mat kernel_y = (cv::Mat_<float>(3, 1) << -1, 0, 1);
        
        cv::filter2D(float_img, grad_x, CV_32F, kernel_x);
        cv::filter2D(float_img, grad_y, CV_32F, kernel_y);
        
        // Square the gradients
        cv::multiply(grad_x, grad_x, grad_x);
        cv::multiply(grad_y, grad_y, grad_y);
        
        // Sum squared gradients
        cv::Scalar sum_x = cv::sum(grad_x);
        cv::Scalar sum_y = cv::sum(grad_y);
        double total = (sum_x[0] + sum_y[0]) / (width * height);
        
        // More aggressive normalization as in original
        return normalizeScore(total, 5000.0);
    }
};


struct EdgeROI {
    cv::Vec4i line;
    cv::Rect roi;
    double angle;
    cv::Mat roi_image;
};


struct EdgeProfileData {
    std::vector<std::vector<double>> profiles;
    cv::Mat visualization;
};


struct ESFData {
    std::vector<double> distances;  // Perpendicular distances from edge
    std::vector<double> intensities;  // Corresponding intensity values
    std::vector<double> binned_distances;  // After binning
    std::vector<double> binned_intensities;  // After binning
};


struct LSFData {
    std::vector<double> distances;     // X-axis positions
    std::vector<double> lsf_values;    // LSF values
    std::vector<double> smoothed_lsf;  // Smoothed LSF values
    double fwhm;                       // Full Width at Half Maximum
};


struct MTFData {
    std::vector<double> frequencies;
    std::vector<double> mtf_values;
    double mtf50;
    double mtf20;
    double mtf10;
    bool is_converted = false;  // Flag to prevent double conversion

    // Convert frequencies to lp/mm
    void convertToLPMM(double pixel_size_mm) {
        if(pixel_size_mm <= 0 || is_converted) return;
        
        double conversion_factor = 1.0 / (2 * pixel_size_mm);
        
        // Convert all frequencies
        for(auto& freq : frequencies) {
            freq *= conversion_factor;
        }
        
        // Convert MTF metrics
        mtf50 *= conversion_factor;
        mtf20 *= conversion_factor;
        mtf10 *= conversion_factor;
        
        is_converted = true;
    }

    // Get MTF50 in appropriate units
    double getMTF50(bool use_lp_mm = false, double pixel_size_mm = 1.0) const {
        if(is_converted || !use_lp_mm) {
            return mtf50;
        }
        return mtf50 / (2 * pixel_size_mm);
    }
};


class MTFAnalyzer {
private:
    static constexpr double ANGLE1_TARGET = 11.0;
    static constexpr double ANGLE2_TARGET = 281.0;
    static constexpr double ANGLE_TOLERANCE = 4.0;
    static constexpr double ROI_LENGTH_FACTOR = 0.8;

public:
    static std::vector<EdgeROI> detectEdgesAndCreateROIs(const cv::Mat& input) {
        std::vector<EdgeROI> detected_edges;
        
        // Create mask for central region
        cv::Mat mask = cv::Mat::ones(input.size(), CV_8UC1);
        int radius = std::min(input.rows, input.cols) / 20; // 5% of min dimension
        cv::circle(mask, cv::Point(input.cols/2, input.rows/2), radius, 
                  cv::Scalar(0), -1);
        
        // Apply mask to input
        cv::Mat masked;
        input.copyTo(masked, mask);
        
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(masked, gray, cv::COLOR_BGR2GRAY);
        
        // Adaptive Canny edge detection
        cv::Mat edges;
        double mean_intensity = cv::mean(gray)[0];
        int canny_low = std::max(20, static_cast<int>(mean_intensity * 0.2));
        int canny_high = std::min(200, static_cast<int>(mean_intensity * 0.6));
        cv::Canny(gray, edges, canny_low, canny_high);
        
        // Probabilistic Hough Line detection with adaptive parameters
        std::vector<cv::Vec4i> lines;
        int min_line_length = std::min(input.rows, input.cols) / 8;
        int max_line_gap = std::min(input.rows, input.cols) / 16;
        cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, min_line_length, max_line_gap);
        
        // Group lines by quadrant
        cv::Point2f image_center(input.cols/2.0f, input.rows/2.0f);
        std::map<int, std::vector<cv::Vec4i>> quadrant_groups;

        for(const auto& l : lines) {
            double angle = std::atan2(-(l[3] - l[1]), l[2] - l[0]) * 180.0 / CV_PI;
            while(angle < 0) angle += 360.0;
            
            if(std::abs(angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE || 
               std::abs(angle - ANGLE2_TARGET) <= ANGLE_TOLERANCE) {
                
                cv::Point2f line_center((l[0] + l[2])/2.0f, (l[1] + l[3])/2.0f);
                
                int quadrant;
                if(line_center.x >= image_center.x && line_center.y < image_center.y) quadrant = 0;
                else if(line_center.x < image_center.x && line_center.y < image_center.y) quadrant = 1;
                else if(line_center.x < image_center.x && line_center.y >= image_center.y) quadrant = 2;
                else quadrant = 3;
                
                quadrant_groups[quadrant].push_back(l);
            }
        }

        // Process each quadrant
        for(const auto& group : quadrant_groups) {
            if(group.second.empty()) continue;
            
            // Find longest line in quadrant
            auto longest_line = *std::max_element(group.second.begin(), group.second.end(),
                [](const cv::Vec4i& a, const cv::Vec4i& b) {
                    return std::hypot(a[2]-a[0], a[3]-a[1]) < std::hypot(b[2]-b[0], b[3]-b[1]);
                });
            
            EdgeROI roi_data;
            roi_data.line = longest_line;
            
            double angle = std::atan2(-(longest_line[3] - longest_line[1]), 
                                    longest_line[2] - longest_line[0]) * 180.0 / CV_PI;
            while(angle < 0) angle += 360.0;
            roi_data.angle = angle;
            
            // Calculate ROI dimensions
            cv::Point2f line_center((longest_line[0] + longest_line[2])/2.0f, 
                                  (longest_line[1] + longest_line[3])/2.0f);
            double length = std::hypot(longest_line[2]-longest_line[0], 
                                     longest_line[3]-longest_line[1]);
            double roi_length = length * ROI_LENGTH_FACTOR;
            
            double narrow_dim = std::min(input.rows, input.cols) * 0.125;
            int roi_width = std::abs(angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE ? 
                           roi_length : narrow_dim;
            int roi_height = std::abs(angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE ? 
                           narrow_dim : roi_length;
            
            // Create ROI rectangle
            roi_data.roi = cv::Rect(
                static_cast<int>(line_center.x - roi_width/2),
                static_cast<int>(line_center.y - roi_height/2),
                roi_width,
                roi_height
            );
            
            // Ensure ROI is within image bounds
            roi_data.roi &= cv::Rect(0, 0, input.cols, input.rows);
            
            // Extract ROI image
            roi_data.roi_image = input(roi_data.roi).clone();
            
            detected_edges.push_back(roi_data);
        }
        
        return detected_edges;
    }

    static EdgeProfileData sampleEdgeProfiles(const cv::Mat& roi, const EdgeROI& edge_data, bool debug = false) {
        EdgeProfileData result;
        static int roi_image_count = 0;
        
        cv::Mat gray;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        
        // Calculate angle from line points
        double line_angle = std::atan2(-(edge_data.line[3] - edge_data.line[1]), 
                                    edge_data.line[2] - edge_data.line[0]) * 180.0 / CV_PI;
        while(line_angle < 0) line_angle += 360.0;

        double angle_rad = -(line_angle * CV_PI / 180.0);
        double perp_angle = angle_rad + CV_PI/2.0;
        
        // Direction vectors for edge and perpendicular
        double dx = std::cos(perp_angle);
        double dy = std::sin(perp_angle);
        double edge_dx = -std::sin(angle_rad);
        double edge_dy = std::cos(angle_rad);
        
        // Setup sampling parameters
        bool is_vertical = std::abs(line_angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE;
        int longer_dim = is_vertical ? gray.rows : gray.cols;
        const int NUM_SAMPLES = 40;  // Increased from 20
        const int SAMPLE_LENGTH = std::min(gray.cols, gray.rows) / 2;
        const double SAMPLE_STEP = 0.5;  // Sub-pixel sampling
        double center_x = gray.cols / 2.0;
        double center_y = gray.rows / 2.0;
        double sample_range = longer_dim * 0.4;

        // Debug visualization setup
        cv::Mat sampling_vis;
        std::vector<cv::Point2f> sampled_points;
        if(debug) {
            sampling_vis = roi.clone();
        }

        // Sample profiles
        for(int i = 0; i < NUM_SAMPLES; i++) {
            std::vector<double> profile;
            std::vector<double> distances;
            
            double t = (i - (NUM_SAMPLES-1)/2.0) / (NUM_SAMPLES-1);
            double offset = t * sample_range;
            
            double start_x = center_x + edge_dx * offset;
            double start_y = center_y + edge_dy * offset;
            
            for(double j = -SAMPLE_LENGTH; j <= SAMPLE_LENGTH; j += SAMPLE_STEP) {
                double x = start_x + dx * j;
                double y = start_y + dy * j;
                
                if(x >= 0 && x < gray.cols-1 && y >= 0 && y < gray.rows-1) {
                    // Bilinear interpolation
                    int x0 = static_cast<int>(x);
                    int y0 = static_cast<int>(y);
                    double fx = x - x0;
                    double fy = y - y0;
                    
                    double p00 = gray.at<uchar>(y0, x0);
                    double p10 = gray.at<uchar>(y0, x0+1);
                    double p01 = gray.at<uchar>(y0+1, x0);
                    double p11 = gray.at<uchar>(y0+1, x0+1);
                    
                    double value = (1-fx)*(1-fy)*p00 + fx*(1-fy)*p10 +
                                (1-fx)*fy*p01 + fx*fy*p11;
                    
                    profile.push_back(value);
                    distances.push_back(j);

                    if(debug) {
                        sampled_points.push_back(cv::Point2f(x, y));
                    }
                }
            }
            
            if(!profile.empty()) {
                result.profiles.push_back(profile);
            }
        }

        // Create debug visualizations
        if(debug) {
            // Draw sampling points
            for(const auto& pt : sampled_points) {
                cv::circle(sampling_vis, pt, 1, cv::Scalar(0,255,255), -1);
            }

            // Draw edge line
            cv::Point2f center(roi.cols/2.0f, roi.rows/2.0f);
            cv::Point2f edge_start = center + cv::Point2f(-50*std::cos(angle_rad), -50*std::sin(angle_rad));
            cv::Point2f edge_end = center + cv::Point2f(50*std::cos(angle_rad), 50*std::sin(angle_rad));
            cv::line(sampling_vis, edge_start, edge_end, cv::Scalar(0,0,255), 2);

            // Create and overlay density heatmap
            cv::Mat density = cv::Mat::zeros(roi.rows, roi.cols, CV_32F);
            for(const auto& pt : sampled_points) {
                cv::circle(density, pt, 2, cv::Scalar(1), -1, cv::LINE_AA);
            }
            cv::GaussianBlur(density, density, cv::Size(5,5), 0);
            cv::normalize(density, density, 0, 1, cv::NORM_MINMAX);

            cv::Mat density_vis;
            density.convertTo(density_vis, CV_8UC3, 255);
            cv::applyColorMap(density_vis, density_vis, cv::COLORMAP_JET);
            cv::addWeighted(sampling_vis, 0.7, density_vis, 0.3, 0, sampling_vis);

            // Add annotations
            cv::putText(sampling_vis, "Angle: " + std::to_string(line_angle),
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                    cv::Scalar(0,0,255), 1);
            
            // Save visualizations
            cv::imwrite("sampling_density_" + std::to_string(roi_image_count) + ".png", 
                    sampling_vis);
            
            result.visualization = sampling_vis;
            roi_image_count++;
        }
        
        return result;
    }
        
        

    static ESFData computeSuperResolutionESF(const EdgeProfileData& profiles, 
                                       double angle_degrees,
                                       bool debug = false) {
        ESFData result;
        const double BIN_WIDTH = 0.1;  // Finer binning
        
        // First pass - find global min/max for normalization
        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::lowest();
        
        for(const auto& profile : profiles.profiles) {
            if(profile.empty()) continue;
            auto [min_it, max_it] = std::minmax_element(profile.begin(), profile.end());
            global_min = std::min(global_min, *min_it);
            global_max = std::max(global_max, *max_it);
        }

        // Process each profile
        for(const auto& profile : profiles.profiles) {
            if(profile.empty()) continue;
            
            // Find edge location using maximum gradient
            std::vector<double> gradients(profile.size()-1);
            for(size_t i = 0; i < profile.size()-1; i++) {
                gradients[i] = std::abs(profile[i+1] - profile[i]);
            }
            auto max_grad_it = std::max_element(gradients.begin(), gradients.end());
            int edge_idx = std::distance(gradients.begin(), max_grad_it);
            
            // Store normalized samples centered on edge
            for(size_t i = 0; i < profile.size(); i++) {
                double normalized = (profile[i] - global_min) / (global_max - global_min);
                result.distances.push_back(static_cast<double>(i) - edge_idx);
                result.intensities.push_back(normalized);
            }
        }

        // Sort by distance
        std::vector<size_t> indices(result.distances.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                [&](size_t a, size_t b) {
                    return result.distances[a] < result.distances[b];
                });

        // Apply moving average to reduce noise while preserving edge shape
        const int WINDOW_SIZE = 5;
        std::vector<double> smoothed_intensities(result.intensities.size());
        
        for(size_t i = 0; i < result.intensities.size(); i++) {
            int start = std::max(0, static_cast<int>(i) - WINDOW_SIZE/2);
            int end = std::min(static_cast<int>(result.intensities.size()), 
                            static_cast<int>(i) + WINDOW_SIZE/2 + 1);
            
            double sum = 0.0;
            int count = 0;
            for(int j = start; j < end; j++) {
                sum += result.intensities[indices[j]];
                count++;
            }
            smoothed_intensities[i] = sum / count;
        }

        // Adaptive binning based on gradient
        std::map<int, std::vector<double>> bins;
        double min_dist = *std::min_element(result.distances.begin(), result.distances.end());
        
        for(size_t i = 0; i < result.distances.size(); i++) {
            int bin = static_cast<int>((result.distances[indices[i]] - min_dist) / BIN_WIDTH);
            bins[bin].push_back(smoothed_intensities[i]);
        }

        // Calculate bin statistics
        for(const auto& bin : bins) {
            if(bin.second.size() >= 3) {
                std::vector<double> bin_values = bin.second;
                std::sort(bin_values.begin(), bin_values.end());
                
                // Use median for robustness
                double median = bin_values[bin_values.size()/2];
                
                result.binned_distances.push_back(min_dist + bin.first * BIN_WIDTH);
                result.binned_intensities.push_back(median);
            }
        }

        if(debug) {
            int plot_height = 400;
            int plot_width = 800;
            result.visualization = cv::Mat(plot_height, plot_width, CV_8UC3, cv::Scalar(255,255,255));
            
            // Draw grid
            for(int i = 0; i < plot_height; i += 50) {
                cv::line(result.visualization, cv::Point(0, i), 
                        cv::Point(plot_width, i), cv::Scalar(200,200,200), 1);
            }

            // Plot raw samples
            for(size_t i = 0; i < result.distances.size(); i++) {
                int x = ((result.distances[i] - min_dist) * plot_width) / 
                        (result.binned_distances.back() - min_dist);
                int y = plot_height - (plot_height * result.intensities[i] / 255.0);
                
                if(x >= 0 && x < plot_width && y >= 0 && y < plot_height) {
                    cv::circle(result.visualization, cv::Point(x, y), 
                            1, cv::Scalar(200,200,200), -1);
                }
            }

            // Plot binned data
            std::vector<cv::Point> points;
            for(size_t i = 0; i < result.binned_distances.size(); i++) {
                int x = ((result.binned_distances[i] - min_dist) * plot_width) / 
                        (result.binned_distances.back() - min_dist);
                int y = plot_height - (plot_height * result.binned_intensities[i]);
                
                if(x >= 0 && x < plot_width && y >= 0 && y < plot_height) {
                    points.push_back(cv::Point(x, y));
                }
            }
            
            if(points.size() > 1) {
                cv::polylines(result.visualization, points, false, 
                            cv::Scalar(0,0,255), 2, cv::LINE_AA);
            }
        }

        return result;
    }

    static LSFData computeLSF(const ESFData& esf, bool debug = false) {
        LSFData result;
        static int roi_counter = 0;
        
        // Ensure we have enough points
        if(esf.binned_distances.size() < 5) {
            return result;
        }

        // Compute first derivative using 5-point stencil
        for(size_t i = 2; i < esf.binned_distances.size() - 2; i++) {
            double h = esf.binned_distances[i+1] - esf.binned_distances[i];
            if(h == 0) continue;

            // 5-point stencil coefficients for smooth derivative
            double derivative = (
                -esf.binned_intensities[i+2] + 
                8*esf.binned_intensities[i+1] - 
                8*esf.binned_intensities[i-1] + 
                esf.binned_intensities[i-2]
            ) / (12 * h);

            result.distances.push_back(esf.binned_distances[i]);
            result.lsf_values.push_back(std::abs(derivative));

            if(i < 5 && debug) {
                std::cout << "Derivative at " << i << ": " << derivative << std::endl;
            }
        }

        std::cout << "Raw LSF points: " << result.lsf_values.size() << std::endl;

        // Apply Gaussian smoothing
        const int window = 5;
        const double sigma = 1.0;
        std::vector<double> kernel(window);
        double sum = 0.0;
        
        // Create Gaussian kernel
        for(int i = 0; i < window; i++) {
            double x = i - window/2;
            kernel[i] = exp(-(x*x)/(2*sigma*sigma));
            sum += kernel[i];
        }
        for(double& k : kernel) k /= sum;

        if(debug) {
            std::cout << "Gaussian kernel values: ";
            for(double k : kernel) std::cout << k << " ";
            std::cout << std::endl;
        }

        // Apply smoothing
        result.smoothed_lsf = result.lsf_values;
        std::vector<double> temp(result.lsf_values.size());
        
        for(size_t i = window/2; i < result.lsf_values.size() - window/2; i++) {
            double sum = 0.0;
            for(int j = 0; j < window; j++) {
                sum += result.lsf_values[i + j - window/2] * kernel[j];
            }
            temp[i] = sum;

            if(i < 5 && debug) {
                std::cout << "Smoothed value at " << i << ": " << sum << std::endl;
            }
        }
        
        result.smoothed_lsf = temp;

        // Normalize LSF
        if(!result.smoothed_lsf.empty()) {
            double max_val = *std::max_element(result.smoothed_lsf.begin(), 
                                            result.smoothed_lsf.end());
            if(debug) {
                std::cout << "Max LSF value before normalization: " << max_val << std::endl;
            }
            
            if(max_val > 0) {
                for(auto& val : result.smoothed_lsf) {
                    val /= max_val;
                }
            }
        }

        // Calculate FWHM
        result.fwhm = calculateFWHM(result.distances, result.smoothed_lsf);
        if(debug) {
            std::cout << "Calculated FWHM: " << result.fwhm << std::endl;
        }

        // Create visualization if debugging
        if(debug) {
            int plot_height = 400;
            int plot_width = 800;
            result.visualization = cv::Mat(plot_height, plot_width, CV_8UC3, cv::Scalar(255,255,255));

            // Draw grid
            for(int i = 0; i < plot_height; i += 50) {
                cv::line(result.visualization, cv::Point(0, i), 
                        cv::Point(plot_width, i), cv::Scalar(200,200,200), 1);
            }

            // Draw half-max line
            int half_max_y = plot_height - (plot_height * 0.5);
            cv::line(result.visualization, 
                    cv::Point(0, half_max_y), 
                    cv::Point(plot_width, half_max_y), 
                    cv::Scalar(0,255,0), 1, cv::LINE_AA);

            // Plot smoothed LSF
            if(!result.smoothed_lsf.empty()) {
                std::vector<cv::Point> points;
                for(size_t i = 0; i < result.smoothed_lsf.size(); i++) {
                    int x = (i * plot_width) / result.smoothed_lsf.size();
                    int y = plot_height - (plot_height * result.smoothed_lsf[i]);
                    if(y >= 0 && y < plot_height) {
                        points.push_back(cv::Point(x, y));
                    }
                }
                
                if(points.size() > 1) {
                    cv::polylines(result.visualization, points, false, 
                                cv::Scalar(0,0,255), 2, cv::LINE_AA);
                }
            }

            // Add labels
            cv::putText(result.visualization, "LSF", 
                        cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 
                        0.5, cv::Scalar(0,0,0), 1);
            cv::putText(result.visualization, 
                        "FWHM: " + std::to_string(result.fwhm), 
                        cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 
                        0.5, cv::Scalar(0,0,0), 1);

            cv::imwrite("lsf_" + std::to_string(roi_counter++) + ".png", 
                    result.visualization);
        }

        return result;
    }

    static double calculateNoiseLevelLSF(const std::vector<double>& lsf) {
        // Calculate noise in the tail regions (first and last 10% of points)
        size_t region_size = lsf.size() / 10;
        std::vector<double> tail_values;
        
        // Collect tail values
        for(size_t i = 0; i < region_size; i++) {
            tail_values.push_back(lsf[i]);
            tail_values.push_back(lsf[lsf.size() - 1 - i]);
        }
        
        // Calculate standard deviation of tail values
        double mean = std::accumulate(tail_values.begin(), tail_values.end(), 0.0) / 
                    tail_values.size();
        double sqsum = std::inner_product(tail_values.begin(), tail_values.end(),
                                        tail_values.begin(), 0.0,
                                        std::plus<>(),
                                        [mean](double x, double y) {
                                            return (x - mean) * (y - mean);
                                        });
        return std::sqrt(sqsum / (tail_values.size() - 1));
    }

    static MTFData computeMTF(const LSFData& lsf, bool debug = false) {
        MTFData result;
        
        // Ensure we have valid LSF data
        if(lsf.smoothed_lsf.empty() || lsf.smoothed_lsf.size() < 4) {
            return result;
        }

        // Prepare data for FFT
        int padded_size = nextPowerOfTwo(lsf.smoothed_lsf.size());
        std::vector<std::complex<double>> fft_input(padded_size);
        
        // Center LSF in padded array
        int offset = (padded_size - lsf.smoothed_lsf.size()) / 2;
        for(size_t i = 0; i < lsf.smoothed_lsf.size(); i++) {
            fft_input[i + offset] = std::complex<double>(lsf.smoothed_lsf[i], 0.0);
        }

        // Apply Hann window
        applyHannWindow(fft_input);

        // Perform FFT
        std::vector<std::complex<double>> fft_result = computeFFT(fft_input);

        // Calculate MTF values
        result.frequencies.resize(fft_result.size() / 2);
        result.mtf_values.resize(fft_result.size() / 2);

        double max_magnitude = 0.0;
        for(size_t i = 0; i < fft_result.size() / 2; i++) {
            double magnitude = std::abs(fft_result[i]);
            max_magnitude = std::max(max_magnitude, magnitude);
        }

        // Normalize and store MTF values
        for(size_t i = 0; i < fft_result.size() / 2; i++) {
            result.frequencies[i] = static_cast<double>(i) / padded_size;
            result.mtf_values[i] = std::abs(fft_result[i]) / max_magnitude;
        }

        // Calculate MTF metrics
        result.mtf50 = findMTFFrequency(result.frequencies, result.mtf_values, 0.5);
        result.mtf20 = findMTFFrequency(result.frequencies, result.mtf_values, 0.2);
        result.mtf10 = findMTFFrequency(result.frequencies, result.mtf_values, 0.1);

        if(debug) {
            result.visualization = createMTFVisualization(result);
        }

        return result;
    }

    static cv::Mat analyzeImageMTF(const cv::Mat& input_image, double pixel_size_mm) {
        // Validate input parameters
        if (input_image.empty()) {
            throw std::runtime_error("Input image is empty");
        }
        if (pixel_size_mm <= 0) {
            throw std::runtime_error("Pixel size must be positive");
        }

        // Step 1: Detect edges and create ROIs
        std::vector<EdgeROI> edges = detectEdgesAndCreateROIs(input_image);
        if (edges.empty()) {
            throw std::runtime_error("No suitable edges found in image");
        }

        // Storage for MTF results from all valid edges
        std::vector<MTFData> mtf_results;

        // Process each detected edge
        for (const auto& edge : edges) {
            try {
                // Step 2: Sample edge profiles
                EdgeProfileData profiles = sampleEdgeProfiles(edge.roi_image, edge);

                // Step 3: Compute super-resolution ESF
                ESFData esf = computeSuperResolutionESF(profiles, edge.angle);

                // Step 4: Compute LSF
                LSFData lsf = computeLSF(esf);

                // Step 5: Compute MTF
                MTFData mtf = computeMTF(lsf);

                // Only include results that meet quality criteria
                if (isValidMTF(mtf, lsf)) {
                    mtf_results.push_back(mtf);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error processing edge: " << e.what() << std::endl;
                continue;
            }
        }

        if (mtf_results.empty()) {
            throw std::runtime_error("No valid MTF results obtained");
        }

        // Average the MTF results
        MTFData averaged_mtf = averageMTFResults(mtf_results);

        // Create visualization with lp/mm units and pixel size
        return createMTFVisualization(averaged_mtf, true, pixel_size_mm);
    }

private:

    static std::vector<double> smoothLSF(const std::vector<double>& input, 
                                       int window_size) {
        std::vector<double> output(input.size());
        std::vector<double> kernel = createGaussianKernel(window_size);
        
        int half_window = window_size / 2;
        
        for(int i = 0; i < static_cast<int>(input.size()); i++) {
            double sum = 0.0;
            double weight_sum = 0.0;
            
            for(int j = -half_window; j <= half_window; j++) {
                int idx = i + j;
                if(idx >= 0 && idx < static_cast<int>(input.size())) {
                    double weight = kernel[j + half_window];
                    sum += input[idx] * weight;
                    weight_sum += weight;
                }
            }
            
            output[i] = sum / weight_sum;
        }
        
        return output;
    }

    static std::vector<double> createGaussianKernel(int size) {
        std::vector<double> kernel(size);
        double sigma = size / 6.0;  // Makes kernel effectively zero at edges
        int half_size = size / 2;
        
        for(int i = -half_size; i <= half_size; i++) {
            kernel[i + half_size] = exp(-(i*i)/(2*sigma*sigma));
        }
        
        return kernel;
    }

    static double calculateFWHM(const std::vector<double>& x, 
                              const std::vector<double>& y) {
        // Find peak position
        auto peak_it = std::max_element(y.begin(), y.end());
        int peak_idx = std::distance(y.begin(), peak_it);
        double half_max = *peak_it / 2.0;
        
        // Find left crossing
        double left_x = x[peak_idx];
        for(int i = peak_idx; i >= 0; i--) {
            if(y[i] <= half_max) {
                double t = (half_max - y[i]) / (y[i+1] - y[i]);
                left_x = x[i] + t * (x[i+1] - x[i]);
                break;
            }
        }
        
        // Find right crossing
        double right_x = x[peak_idx];
        for(int i = peak_idx; i < static_cast<int>(y.size()) - 1; i++) {
            if(y[i+1] <= half_max) {
                double t = (half_max - y[i+1]) / (y[i] - y[i+1]);
                right_x = x[i+1] + t * (x[i] - x[i+1]);
                break;
            }
        }
        
        return right_x - left_x;
    }

    static cv::Mat createLSFVisualization(const LSFData& lsf) {
        int plot_height = 400;
        int plot_width = 800;
        cv::Mat visualization(plot_height, plot_width, CV_8UC3, 
                            cv::Scalar(255,255,255));
        
        // Draw grid
        for(int i = 0; i < plot_height; i += 50) {
            cv::line(visualization, cv::Point(0, i), 
                    cv::Point(plot_width, i), 
                    cv::Scalar(200,200,200), 1);
        }

        // Plot raw LSF points
        std::vector<cv::Point> raw_points;
        for(size_t i = 0; i < lsf.lsf_values.size(); i++) {
            int x = (i * plot_width) / lsf.lsf_values.size();
            int y = plot_height - (plot_height * lsf.lsf_values[i]);
            if(y >= 0 && y < plot_height) {
                raw_points.push_back(cv::Point(x, y));
                cv::circle(visualization, cv::Point(x, y), 
                          1, cv::Scalar(200,200,200), -1);
            }
        }

        // Plot smoothed LSF
        std::vector<cv::Point> smooth_points;
        for(size_t i = 0; i < lsf.smoothed_lsf.size(); i++) {
            int x = (i * plot_width) / lsf.smoothed_lsf.size();
            int y = plot_height - (plot_height * lsf.smoothed_lsf[i]);
            if(y >= 0 && y < plot_height) {
                smooth_points.push_back(cv::Point(x, y));
            }
        }
        
        if(smooth_points.size() > 1) {
            cv::polylines(visualization, smooth_points, false, 
                         cv::Scalar(0,0,255), 2, cv::LINE_AA);
        }

        // Draw FWHM
        double peak_val = *std::max_element(lsf.smoothed_lsf.begin(), 
                                          lsf.smoothed_lsf.end());
        int half_max_y = plot_height - (plot_height * peak_val / 2);
        cv::line(visualization, 
                cv::Point(0, half_max_y), 
                cv::Point(plot_width, half_max_y), 
                cv::Scalar(0,255,0), 1, cv::LINE_AA);

        // Add labels
        cv::putText(visualization, "LSF", 
                   cv::Point(10, 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 
                   0.5, cv::Scalar(0,0,0), 1);
        cv::putText(visualization, 
                   "FWHM: " + std::to_string(lsf.fwhm), 
                   cv::Point(10, 40), 
                   cv::FONT_HERSHEY_SIMPLEX, 
                   0.5, cv::Scalar(0,0,0), 1);

        return visualization;
    }

    static int nextPowerOfTwo(int n) {
        int power = 1;
        while(power < n) {
            power *= 2;
        }
        return power;
    }

    static void applyHannWindow(std::vector<std::complex<double>>& data) {
        int size = data.size();
        for(int i = 0; i < size; i++) {
            double multiplier = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (size - 1)));
            data[i] *= multiplier;
        }
    }

    static std::vector<std::complex<double>> computeFFT(std::vector<std::complex<double>>& data) {
        int n = data.size();
        
        if(n <= 1) {
            return data;
        }

        // Split into even and odd
        std::vector<std::complex<double>> even(n/2), odd(n/2);
        for(int i = 0; i < n/2; i++) {
            even[i] = data[2*i];
            odd[i] = data[2*i + 1];
        }

        // Recursive FFT
        even = computeFFT(even);
        odd = computeFFT(odd);

        // Combine
        std::vector<std::complex<double>> result(n);
        for(int i = 0; i < n/2; i++) {
            double angle = -2.0 * M_PI * i / n;
            std::complex<double> twiddle(std::cos(angle), std::sin(angle));
            result[i] = even[i] + twiddle * odd[i];
            result[i + n/2] = even[i] - twiddle * odd[i];
        }

        return result;
    }

    static double findMTFFrequency(const std::vector<double>& frequencies,
                                 const std::vector<double>& mtf_values,
                                 double target_value) {
        for(size_t i = 0; i < mtf_values.size() - 1; i++) {
            if(mtf_values[i] >= target_value && mtf_values[i+1] < target_value) {
                // Linear interpolation
                double t = (target_value - mtf_values[i+1]) / 
                          (mtf_values[i] - mtf_values[i+1]);
                return frequencies[i+1] + t * (frequencies[i] - frequencies[i+1]);
            }
        }
        return 0.0;
    }

    static bool isValidMTF(const MTFData& mtf, const LSFData& lsf) {
        // Check LSF quality
        double noise_level = calculateNoiseLevelLSF(lsf.smoothed_lsf);
        double peak_value = *std::max_element(lsf.smoothed_lsf.begin(), 
                                            lsf.smoothed_lsf.end());
        double snr = peak_value / noise_level;

        // Quality criteria
        bool good_snr = snr >= 10.0;
        bool reasonable_fwhm = lsf.fwhm >= 0.5 && lsf.fwhm <= 5.0;
        bool reasonable_mtf50 = mtf.mtf50 > 0.01 && mtf.mtf50 < 0.5;

        return good_snr && reasonable_fwhm && reasonable_mtf50;
    }

    static MTFData averageMTFResults(const std::vector<MTFData>& results) {
        if (results.empty()) {
            throw std::runtime_error("No MTF results to average");
        }

        MTFData averaged;
        averaged.frequencies = results[0].frequencies;
        averaged.mtf_values.resize(averaged.frequencies.size(), 0.0);

        // Average MTF values
        for (const auto& result : results) {
            for (size_t i = 0; i < averaged.mtf_values.size(); i++) {
                averaged.mtf_values[i] += result.mtf_values[i] / results.size();
            }
        }

        // Recalculate metrics
        averaged.mtf50 = findMTFFrequency(averaged.frequencies, averaged.mtf_values, 0.5);
        averaged.mtf20 = findMTFFrequency(averaged.frequencies, averaged.mtf_values, 0.2);
        averaged.mtf10 = findMTFFrequency(averaged.frequencies, averaged.mtf_values, 0.1);

        return averaged;
    }
};


class FocusAnalyzer {
public:
    static AnalysisResult analyze(const std::vector<uint8_t>& data, 
                                int width, int height, 
                                const std::string& metric) {
        AnalysisResult result;
        
        // Calculate traditional metrics
        result.ml_score = ModifiedLaplacian::measure(data, width, height);
        result.tenengrad_score = Tenengrad::measure(data, width, height);
        result.brenner_score = BrennerGradient::measure(data, width, height);
        
        // Initialize MTF-specific fields
        result.valid_mtf = false;
        result.mtf_error = "";
        
        // Calculate selected metric
        if (metric == "Modified Laplacian") {
            result.quality_score = result.ml_score;
        } else if (metric == "Tenengrad") {
            result.quality_score = result.tenengrad_score;
        } else if (metric == "Brenner Gradient") {
            result.quality_score = result.brenner_score;
        } else {
            // Combined metric
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
    output.set("valid_mtf", result.valid_mtf);
    output.set("mtf_error", result.mtf_error);
    output.set("details", val::object());
    output["details"].set("modifiedLaplacian", result.ml_score);
    output["details"].set("tenengrad", result.tenengrad_score);
    output["details"].set("brennerGradient", result.brenner_score);

    val edges = val::array();
    for (const auto& edge : result.detected_edges) {
        val edge_data = val::object();
        
        val x_coords = val::array();
        val y_coords = val::array();
        for (size_t i = 0; i < edge.x_coords.size(); i++) {
            x_coords.set(i, edge.x_coords[i]);
            y_coords.set(i, edge.y_coords[i]);
        }
        
        edge_data.set("x_coords", x_coords);
        edge_data.set("y_coords", y_coords);
        edge_data.set("angle", edge.angle);
        edge_data.set("strength", edge.strength);
        
        edges.call<void>("push", edge_data);
    }
    output.set("detected_edges", edges);
    
    return output;
}

EMSCRIPTEN_BINDINGS(focus_metrics) {
    function("analyzeImage", &analyzeImage)
        .function("analyze", &focus_metrics::FocusAnalyzer::analyze);

    // Explicitly register the ImageAnalyzer class
    class_<focus_metrics::FocusAnalyzer>("ImageAnalyzer")
        .constructor<>();
}
