// Modified mtf_analyzer_2.cpp to implement
// more OpenCV functions for optimization

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>   // for std::iota, std::accumulate
#include <algorithm> // for std::sort
#include <map>       // for std::map
#include <cmath>     // for mathematical operations

class MTFAnalyzer
{
private:
    static constexpr double ANGLE1_TARGET = 11.0;
    static constexpr double ANGLE2_TARGET = 281.0;
    static constexpr double ANGLE_TOLERANCE = 4.0;
    static constexpr double ROI_LENGTH_FACTOR = 0.8;
    static constexpr double ROI_WIDTH = 100;
    static int roi_counter;

public:
    struct EdgeROI
    {
        cv::Vec4i line;
        cv::Rect roi;
        double angle;
        cv::Mat roi_image;
    };

    struct EdgeProfileData
    {
        std::vector<std::vector<double>> profiles;
        cv::Mat visualization;
    };

    struct ESFData
    {
        std::vector<double> distances;          // Perpendicular distances from edge
        std::vector<double> intensities;        // Corresponding intensity values
        std::vector<double> binned_distances;   // After binning
        std::vector<double> binned_intensities; // After binning
        cv::Mat visualization;                  // For debugging
    };

    struct LSFData
    {
        std::vector<double> distances;    // X-axis positions
        std::vector<double> lsf_values;   // LSF values
        std::vector<double> smoothed_lsf; // Smoothed LSF values
        double fwhm;                      // Full Width at Half Maximum
        cv::Mat visualization;            // Debug visualization
    };

    struct MTFData
    {
        std::vector<double> frequencies;
        std::vector<double> mtf_values;
        double mtf50;
        double mtf20;
        double mtf10;
        cv::Mat visualization;
        bool is_converted = false; // Flag to prevent double conversion

        // Convert frequencies to lp/mm
        void convertToLPMM(double pixel_size_mm)
        {
            if (pixel_size_mm <= 0 || is_converted)
                return;

            double conversion_factor = 1.0 / (2 * pixel_size_mm);

            // Convert all frequencies
            for (auto &freq : frequencies)
            {
                freq *= conversion_factor;
            }

            // Convert MTF metrics
            mtf50 *= conversion_factor;
            mtf20 *= conversion_factor;
            mtf10 *= conversion_factor;

            is_converted = true;
        }

        // Get MTF50 in appropriate units
        double getMTF50(bool use_lp_mm = false, double pixel_size_mm = 1.0) const
        {
            if (is_converted || !use_lp_mm)
            {
                return mtf50;
            }
            return mtf50 / (2 * pixel_size_mm);
        }
    };

    class BenchmarkTimer
    {
        using Clock = std::chrono::high_resolution_clock;
        Clock::time_point start;
        std::string name;
    public:
        BenchmarkTimer(std::string n): name(n), start(Clock::now()) {
            std::cout << "\nStarting " << name << "..." << std::endl;  // Add start message
        }
        ~BenchmarkTimer() {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>
                (Clock::now() - start);
            std::cout << "\n" << name << " completed in: " << duration.count() << "μs" << std::endl;
            std::cout.flush();  // Force output
        }
    };
    
    static std::vector<EdgeROI> detectEdgesAndCreateROIs(const cv::Mat& input, bool debug = false)
    {
        std::vector<EdgeROI> detected_edges;

        // Create copy and mask central region using more efficient OpenCV operations
        cv::Mat masked;
        input.copyTo(masked);
        int radius = std::min(input.rows, input.cols) / 100 * 5;
        cv::circle(masked, cv::Point(input.cols/2, input.rows/2), radius, cv::Scalar(0,0,0), -1, cv::LINE_AA);

        if (debug) {
            cv::imwrite("open-cv-1_masked_input.png", masked);
        }

        // Convert to grayscale and compute mean using OpenCV's optimized functions
        cv::Mat gray;
        cv::cvtColor(masked, gray, cv::COLOR_BGR2GRAY);
        cv::Scalar mean_scalar = cv::mean(gray);
        double mean_intensity = mean_scalar[0];

        // Use OpenCV's built-in automatic thresholding for Canny
        cv::Mat edges;
        int canny_low = std::max(20, static_cast<int>(mean_intensity * 0.2));
        int canny_high = std::min(200, static_cast<int>(mean_intensity * 0.6));
        cv::Canny(gray, edges, canny_low, canny_high);

        if (debug) {
            cv::imwrite("open-cv-2_detected_edges.png", edges);
        }

        // Optimize line detection parameters
        std::vector<cv::Vec4i> lines;
        int min_line_length = std::min(input.rows, input.cols) / 8;
        int max_line_gap = std::min(input.rows, input.cols) / 16;
        cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, min_line_length, max_line_gap);

        if (debug) {
            cv::Mat lines_image = masked.clone();
            cv::Point2f image_center(input.cols/2.0f, input.rows/2.0f);
            for (const auto& l : lines) {
                // Use OpenCV's line drawing with anti-aliasing
                cv::line(lines_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), 
                        cv::Scalar(0,255,0), 2, cv::LINE_AA);
            }
            cv::imwrite("open-cv-3_detected_lines.png", lines_image);
            std::cout << "Found " << lines.size() << " total lines" << std::endl;
        }

        // Use OpenCV's Point2f for better precision in angle calculations
        cv::Point2f image_center(input.cols/2.0f, input.rows/2.0f);
        
        // Create quadrant groups using OpenCV's Point2f for calculations
        std::map<int, std::vector<cv::Vec4i>> quadrant_groups;
        
        for (const auto& l : lines) {
            cv::Point2f pt1(l[0], l[1]), pt2(l[2], l[3]);
            cv::Point2f line_vector = pt2 - pt1;
            
            // Use OpenCV's fastAtan2 for better performance
            double angle = -cv::fastAtan2(line_vector.y, line_vector.x);
            while (angle < 0) angle += 360.0;

            if (std::abs(angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE || 
                std::abs(angle - ANGLE2_TARGET) <= ANGLE_TOLERANCE) {
                
                cv::Point2f line_center = (pt1 + pt2) * 0.5f;
                cv::Point2f relative_pos = line_center - image_center;

                // Determine quadrant using OpenCV's point arithmetic
                int quadrant = (relative_pos.x >= 0 ? 0 : 1) + (relative_pos.y >= 0 ? 2 : 0);
                quadrant_groups[quadrant].push_back(l);
            }
        }

        cv::Mat rois_image;
        if (debug) {
            rois_image = input.clone();
        }

        // Process each quadrant using OpenCV's optimized functions
        for (const auto& group : quadrant_groups) {
            if (group.second.empty()) continue;

            // Find longest line using OpenCV's norm function
            auto longest_line = *std::max_element(
                group.second.begin(), 
                group.second.end(),
                [](const cv::Vec4i& a, const cv::Vec4i& b) {
                    cv::Point2f vec_a(a[2] - a[0], a[3] - a[1]);
                    cv::Point2f vec_b(b[2] - b[0], b[3] - b[1]);
                    return cv::norm(vec_a) < cv::norm(vec_b);
                }
            );

            EdgeROI roi_data;
            roi_data.line = longest_line;
            
            // Use OpenCV's point arithmetic and fastAtan2
            cv::Point2f line_vector(longest_line[2] - longest_line[0], 
                                longest_line[3] - longest_line[1]);
            double angle = -cv::fastAtan2(line_vector.y, line_vector.x);
            while (angle < 0) angle += 360.0;
            roi_data.angle = angle;

            // Calculate ROI dimensions using OpenCV's point arithmetic
            cv::Point2f line_start(longest_line[0], longest_line[1]);
            cv::Point2f line_end(longest_line[2], longest_line[3]);
            cv::Point2f line_center = (line_start + line_end) * 0.5f;
            
            float length = cv::norm(line_end - line_start);
            float roi_length = length * ROI_LENGTH_FACTOR;
            
            float narrow_dim = std::min(input.rows, input.cols) * 0.125f;
            int roi_width = std::abs(angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE ? 
                        roi_length : narrow_dim;
            int roi_height = std::abs(angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE ? 
                            narrow_dim : roi_length;

            // Create ROI using OpenCV's rectangle constructor
            roi_data.roi = cv::Rect(
                cv::Point(line_center.x - roi_width/2, line_center.y - roi_height/2),
                cv::Size(roi_width, roi_height)
            );

            // Use OpenCV's rectangle intersection
            roi_data.roi &= cv::Rect(0, 0, input.cols, input.rows);
            
            // Use OpenCV's optimized ROI extraction
            roi_data.roi_image = input(roi_data.roi).clone();
            detected_edges.push_back(roi_data);

            if (debug) {
                // Use OpenCV's anti-aliased line drawing
                cv::line(rois_image, cv::Point(longest_line[0], longest_line[1]),
                        cv::Point(longest_line[2], longest_line[3]),
                        cv::Scalar(0,255,0), 2, cv::LINE_AA);
                cv::rectangle(rois_image, roi_data.roi, cv::Scalar(0,0,255), 2, cv::LINE_AA);
            }
        }

        if (debug) {
            cv::imwrite("open-cv-4_rois.png", rois_image);
            std::cout << "Found " << detected_edges.size() << " edges at target angles" << std::endl;
        }

        return detected_edges;
    }

    static EdgeProfileData sampleEdgeProfiles(const cv::Mat& roi, const EdgeROI& edge_data, bool debug = false)
    {
        EdgeProfileData result;
        static int roi_image_count = 0;

        std::cout << "Starting profile sampling..." << std::endl;
        std::cout << "ROI size: " << roi.size() << std::endl;

        // Convert to grayscale using OpenCV
        cv::Mat gray;
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        std::cout << "Converted to grayscale. Size: " << gray.size() << std::endl;

        // Calculate angle using OpenCV's point arithmetic
        cv::Point2f line_vector(edge_data.line[2] - edge_data.line[0], 
                            edge_data.line[3] - edge_data.line[1]);
        double line_angle = -cv::fastAtan2(line_vector.y, line_vector.x);
        while (line_angle < 0) line_angle += 360.0;
        std::cout << "Calculated line angle: " << line_angle << std::endl;

        // Convert angles to radians using OpenCV
        double angle_rad = -line_angle * CV_PI / 180.0;
        double perp_angle = angle_rad + CV_PI / 2.0;

        // Create direction vectors using OpenCV points
        cv::Point2f dir_vector(std::cos(perp_angle), std::sin(perp_angle));
        cv::Point2f edge_dir(-std::sin(angle_rad), std::cos(angle_rad));
        std::cout << "Direction vectors calculated." << std::endl;

        // Setup sampling parameters
        bool is_vertical = std::abs(line_angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE;
        int longer_dim = is_vertical ? gray.rows : gray.cols;
        const int NUM_SAMPLES = 40;
        const int SAMPLE_LENGTH = std::min(gray.cols, gray.rows) / 2;
        const double SAMPLE_STEP = 0.5;
        cv::Point2f center(gray.cols / 2.0f, gray.rows / 2.0f);
        double sample_range = longer_dim * 0.4;

        std::cout << "Sampling parameters:" << std::endl;
        std::cout << "Longer dimension: " << longer_dim << std::endl;
        std::cout << "Sample length: " << SAMPLE_LENGTH << std::endl;
        std::cout << "Sample range: " << sample_range << std::endl;

        // Sample profiles
        for (int i = 0; i < NUM_SAMPLES; i++) {
            std::vector<double> profile;
            double t = (i - (NUM_SAMPLES - 1) / 2.0) / (NUM_SAMPLES - 1);
            double offset = t * sample_range;

            // Calculate start point
            cv::Point2f start_point = center + edge_dir * offset;
            
            if (i == 0 || i == NUM_SAMPLES-1) {
                std::cout << "Profile " << i << " start point: " << start_point << std::endl;
            }

            // Sample points along profile
            for (double j = -SAMPLE_LENGTH; j <= SAMPLE_LENGTH; j += SAMPLE_STEP) {
                cv::Point2f sample_point = start_point + dir_vector * j;
                
                if (sample_point.x >= 0 && sample_point.x < gray.cols - 1 && 
                    sample_point.y >= 0 && sample_point.y < gray.rows - 1) {
                    // Bilinear interpolation
                    int x0 = static_cast<int>(sample_point.x);
                    int y0 = static_cast<int>(sample_point.y);
                    double fx = sample_point.x - x0;
                    double fy = sample_point.y - y0;

                    double p00 = gray.at<uchar>(y0, x0);
                    double p10 = gray.at<uchar>(y0, x0 + 1);
                    double p01 = gray.at<uchar>(y0 + 1, x0);
                    double p11 = gray.at<uchar>(y0 + 1, x0 + 1);

                    double value = (1 - fx) * (1 - fy) * p00 + 
                                fx * (1 - fy) * p10 +
                                (1 - fx) * fy * p01 + 
                                fx * fy * p11;

                    profile.push_back(value);
                }
            }

            if (!profile.empty()) {
                result.profiles.push_back(profile);
                if (i == 0) {
                    std::cout << "First profile size: " << profile.size() << std::endl;
                    std::cout << "First few values: ";
                    for (int j = 0; j < std::min(5, (int)profile.size()); j++) {
                        std::cout << profile[j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }

        std::cout << "Total profiles collected: " << result.profiles.size() << std::endl;
        if (!result.profiles.empty()) {
            std::cout << "Profile lengths: " << result.profiles[0].size() << std::endl;
        }

        return result;
    }

    static ESFData computeSuperResolutionESF(const EdgeProfileData& profiles, double angle_degrees, bool debug = false)
    {
        ESFData result;
        const double BIN_WIDTH = 0.1;

        std::cout << "\nStarting ESF computation..." << std::endl;
        std::cout << "Number of input profiles: " << profiles.profiles.size() << std::endl;
        if (!profiles.profiles.empty()) {
            std::cout << "First profile length: " << profiles.profiles[0].size() << std::endl;
        }

        // Find global min/max
        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::lowest();
        
        for (const auto& profile : profiles.profiles) {
            if (!profile.empty()) {
                auto [min_it, max_it] = std::minmax_element(profile.begin(), profile.end());
                global_min = std::min(global_min, *min_it);
                global_max = std::max(global_max, *max_it);
            }
        }

        std::cout << "Global min: " << global_min << ", Global max: " << global_max << std::endl;

        // Process each profile
        for (size_t profile_idx = 0; profile_idx < profiles.profiles.size(); profile_idx++) {
            const auto& profile = profiles.profiles[profile_idx];
            if (profile.empty()) continue;

            // Calculate gradients for edge detection
            std::vector<double> gradients(profile.size() - 1);
            for (size_t i = 0; i < profile.size() - 1; i++) {
                gradients[i] = std::abs(profile[i + 1] - profile[i]);
            }

            // Find edge location using maximum gradient
            auto max_grad_it = std::max_element(gradients.begin(), gradients.end());
            int edge_idx = std::distance(gradients.begin(), max_grad_it);

            if (profile_idx == 0) {
                std::cout << "First profile edge index: " << edge_idx << std::endl;
                std::cout << "Max gradient at edge: " << *max_grad_it << std::endl;
            }

            // Store normalized samples centered on edge
            for (size_t i = 0; i < profile.size(); i++) {
                double normalized = (profile[i] - global_min) / (global_max - global_min);
                result.distances.push_back(static_cast<double>(i) - edge_idx);
                result.intensities.push_back(normalized);
            }
        }

        std::cout << "Collected " << result.distances.size() << " total points" << std::endl;

        // Sort by distance
        std::vector<size_t> indices(result.distances.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                [&](size_t a, size_t b) {
                    return result.distances[a] < result.distances[b];
                });

        std::cout << "Sorted points by distance" << std::endl;

        // Apply moving average
        const int WINDOW_SIZE = 5;
        std::vector<double> smoothed_intensities(result.intensities.size());

        for (size_t i = 0; i < result.intensities.size(); i++) {
            int start = std::max(0, static_cast<int>(i) - WINDOW_SIZE / 2);
            int end = std::min(static_cast<int>(result.intensities.size()),
                            static_cast<int>(i) + WINDOW_SIZE / 2 + 1);

            double sum = 0.0;
            int count = 0;
            for (int j = start; j < end; j++) {
                sum += result.intensities[indices[j]];
                count++;
            }
            smoothed_intensities[i] = sum / count;
        }

        std::cout << "Applied moving average smoothing" << std::endl;

        // Adaptive binning
        std::map<int, std::vector<double>> bins;
        double min_dist = *std::min_element(result.distances.begin(), result.distances.end());

        for (size_t i = 0; i < result.distances.size(); i++) {
            int bin = static_cast<int>((result.distances[indices[i]] - min_dist) / BIN_WIDTH);
            bins[bin].push_back(smoothed_intensities[i]);
        }

        // Calculate bin statistics
        for (const auto& bin : bins) {
            if (bin.second.size() >= 3) {
                std::vector<double> bin_values = bin.second;
                std::sort(bin_values.begin(), bin_values.end());
                double median = bin_values[bin_values.size() / 2];

                result.binned_distances.push_back(min_dist + bin.first * BIN_WIDTH);
                result.binned_intensities.push_back(median);
            }
        }

        std::cout << "Final binned points: " << result.binned_distances.size() << std::endl;

        return result;
    }

    static LSFData computeLSF(const ESFData& esf, bool debug = false)
    {
        LSFData result;
        static int roi_counter = 0;

        std::cout << "\nStarting LSF computation..." << std::endl;
        std::cout << "Input ESF points: " << esf.binned_distances.size() << std::endl;

        // Ensure we have enough points
        if (esf.binned_distances.size() < 5) {
            std::cout << "Not enough points for LSF computation" << std::endl;
            return result;
        }

        // Create vectors for derivative calculation
        result.distances.resize(esf.binned_distances.size() - 4);
        result.lsf_values.resize(esf.binned_distances.size() - 4);

        // Compute derivative using 5-point stencil
        for (size_t i = 2; i < esf.binned_distances.size() - 2; i++) {
            double h = esf.binned_distances[i + 1] - esf.binned_distances[i];
            if (h != 0) {
                double derivative = (-esf.binned_intensities[i + 2] +
                                8 * esf.binned_intensities[i + 1] -
                                8 * esf.binned_intensities[i - 1] +
                                esf.binned_intensities[i - 2]) / (12 * h);
                
                result.distances[i-2] = esf.binned_distances[i];
                result.lsf_values[i-2] = std::abs(derivative);
            }
        }

        if (debug) {
            std::cout << "Computed LSF points: " << result.lsf_values.size() << std::endl;
        }

        // Apply Gaussian smoothing
        const int window = 5;
        const double sigma = 1.0;
        std::vector<double> kernel(window);
        double kernel_sum = 0.0;

        // Create Gaussian kernel
        for (int i = 0; i < window; i++) {
            double x = i - window/2;
            kernel[i] = std::exp(-(x*x)/(2*sigma*sigma));
            kernel_sum += kernel[i];
        }
        // Normalize kernel
        for (double& k : kernel) {
            k /= kernel_sum;
        }

        if (debug) {
            std::cout << "Created Gaussian kernel of size " << window << std::endl;
        }

        // Apply smoothing
        result.smoothed_lsf = result.lsf_values;
        std::vector<double> temp(result.lsf_values.size());

        for (size_t i = window/2; i < result.lsf_values.size() - window/2; i++) {
            double sum = 0.0;
            for (int j = 0; j < window; j++) {
                sum += result.lsf_values[i + j - window/2] * kernel[j];
            }
            temp[i] = sum;
        }

        result.smoothed_lsf = temp;

        // Normalize LSF
        if (!result.smoothed_lsf.empty()) {
            double max_val = *std::max_element(result.smoothed_lsf.begin(), result.smoothed_lsf.end());
            
            if (debug) {
                std::cout << "Max LSF value before normalization: " << max_val << std::endl;
            }

            if (max_val > 0) {
                for (auto& val : result.smoothed_lsf) {
                    val /= max_val;
                }
            }
        }

        // Calculate FWHM
        result.fwhm = calculateFWHM(result.distances, result.smoothed_lsf);
        
        if (debug) {
            std::cout << "Calculated FWHM: " << result.fwhm << std::endl;
        }

        // Create visualization if debugging
        if (debug) {
            const int plot_height = 400;
            const int plot_width = 800;
            result.visualization = cv::Mat(plot_height, plot_width, CV_8UC3, 
                                        cv::Scalar(255, 255, 255));

            // Draw grid with anti-aliasing
            for (int i = 0; i < plot_height; i += 50) {
                cv::line(result.visualization, cv::Point(0, i),
                        cv::Point(plot_width, i), cv::Scalar(200, 200, 200), 
                        1, cv::LINE_AA);
            }

            // Draw half-max line with anti-aliasing
            int half_max_y = plot_height - (plot_height * 0.5);
            cv::line(result.visualization,
                    cv::Point(0, half_max_y),
                    cv::Point(plot_width, half_max_y),
                    cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

            // Plot smoothed LSF using OpenCV's drawing functions
            if (!result.smoothed_lsf.empty()) {
                std::vector<cv::Point> points;
                for (size_t i = 0; i < result.smoothed_lsf.size(); i++) {
                    int x = (i * plot_width) / result.smoothed_lsf.size();
                    int y = plot_height - (plot_height * result.smoothed_lsf[i]);
                    if (y >= 0 && y < plot_height) {
                        points.push_back(cv::Point(x, y));
                    }
                }

                if (points.size() > 1) {
                    cv::polylines(result.visualization, points, false,
                                cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                }
            }

            // Add labels with anti-aliased text
            cv::putText(result.visualization, "LSF",
                    cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
            cv::putText(result.visualization,
                    "FWHM: " + std::to_string(result.fwhm),
                    cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

            cv::imwrite("opencv-lsf_" + std::to_string(roi_counter++) + ".png",
                    result.visualization);
        }

        return result;
    }

    // Helper function updated to use OpenCV
    static double calculateFWHM(const std::vector<double>& x, const std::vector<double>& y) 
    {
        if (x.empty() || y.empty() || x.size() != y.size()) {
            return 0.0;
        }

        // Find peak position
        auto peak_it = std::max_element(y.begin(), y.end());
        int peak_idx = std::distance(y.begin(), peak_it);
        double half_max = *peak_it / 2.0;

        // Find left crossing
        double left_x = x[peak_idx];
        for (int i = peak_idx; i >= 0; i--) {
            if (y[i] <= half_max) {
                double t = (half_max - y[i]) / (y[i + 1] - y[i]);
                left_x = x[i] + t * (x[i + 1] - x[i]);
                break;
            }
        }

        // Find right crossing
        double right_x = x[peak_idx];
        for (int i = peak_idx; i < static_cast<int>(y.size()) - 1; i++) {
            if (y[i + 1] <= half_max) {
                double t = (half_max - y[i + 1]) / (y[i] - y[i + 1]);
                right_x = x[i + 1] + t * (x[i] - x[i + 1]);
                break;
            }
        }

        return right_x - left_x;
    }
    
    static double calculateNoiseLevelLSF(const std::vector<double> &lsf)
    {
        // Calculate noise in the tail regions (first and last 10% of points)
        size_t region_size = lsf.size() / 10;
        std::vector<double> tail_values;

        // Collect tail values
        for (size_t i = 0; i < region_size; i++)
        {
            tail_values.push_back(lsf[i]);
            tail_values.push_back(lsf[lsf.size() - 1 - i]);
        }

        // Calculate standard deviation of tail values
        double mean = std::accumulate(tail_values.begin(), tail_values.end(), 0.0) /
                      tail_values.size();
        double sqsum = std::inner_product(tail_values.begin(), tail_values.end(),
                                          tail_values.begin(), 0.0,
                                          std::plus<>(),
                                          [mean](double x, double y)
                                          {
                                              return (x - mean) * (y - mean);
                                          });
        return std::sqrt(sqsum / (tail_values.size() - 1));
    }

    static MTFData computeMTF(const LSFData& lsf, bool debug = false)
    {
        MTFData result;

        // Ensure we have valid LSF data
        if (lsf.smoothed_lsf.empty() || lsf.smoothed_lsf.size() < 4) {
            std::cout << "Invalid LSF data for MTF calculation" << std::endl;
            return result;
        }

        // Convert vector to Mat
        cv::Mat lsf_data(1, lsf.smoothed_lsf.size(), CV_64F);
        for (size_t i = 0; i < lsf.smoothed_lsf.size(); i++) {
            lsf_data.at<double>(0, i) = lsf.smoothed_lsf[i];
        }

        // Get optimal DFT size and pad data
        int dft_size = cv::getOptimalDFTSize(lsf_data.cols);
        cv::Mat padded;
        cv::copyMakeBorder(lsf_data, padded, 0, 0, 0, dft_size - lsf_data.cols, 
                        cv::BORDER_CONSTANT, cv::Scalar(0));

        // Apply Hann window
        cv::Mat hann(1, dft_size, CV_64F);
        for (int i = 0; i < dft_size; i++) {
            hann.at<double>(0, i) = 0.5 * (1.0 - cos(2.0 * CV_PI * i / (dft_size - 1)));
        }
        padded = padded.mul(hann);

        // Prepare matrices for DFT
        cv::Mat planes[] = {padded, cv::Mat::zeros(padded.size(), CV_64F)};
        cv::Mat complexMat;
        cv::merge(planes, 2, complexMat);

        // Perform DFT
        cv::dft(complexMat, complexMat);

        // Split into real and imaginary parts
        cv::split(complexMat, planes);

        // Compute magnitude
        cv::Mat magnitude;
        cv::magnitude(planes[0], planes[1], magnitude);

        // Extract first half (up to Nyquist frequency)
        magnitude = magnitude(cv::Rect(0, 0, magnitude.cols/2, 1));

        // Normalize
        double maxVal;
        cv::minMaxLoc(magnitude, nullptr, &maxVal);
        magnitude /= maxVal;

        // Store results
        result.frequencies.resize(magnitude.cols);
        result.mtf_values.resize(magnitude.cols);
        
        for (int i = 0; i < magnitude.cols; i++) {
            result.frequencies[i] = static_cast<double>(i) / dft_size;
            result.mtf_values[i] = magnitude.at<double>(0, i);
        }

        // Calculate MTF metrics
        result.mtf50 = findMTFFrequency(result.frequencies, result.mtf_values, 0.5);
        result.mtf20 = findMTFFrequency(result.frequencies, result.mtf_values, 0.2);
        result.mtf10 = findMTFFrequency(result.frequencies, result.mtf_values, 0.1);

        if (debug) {
            result.visualization = createMTFVisualization(result);
        }

        return result;
    }

    static cv::Mat createMTFVisualization(const MTFData &mtf, bool use_lp_mm = true, double pixel_size_mm = 1.0)
    {
        // Increased margins and plot dimensions
        int plot_height = 400;
        int plot_width = 800;
        int left_margin = 80;
        int right_margin = 40;
        int top_margin = 40;
        int bottom_margin = 60;

        int total_width = left_margin + plot_width + right_margin;
        int total_height = top_margin + plot_height + bottom_margin;

        cv::Mat visualization(total_height, total_width, CV_8UC3, cv::Scalar(255, 255, 255));

        // Calculate maximum frequency in lp/mm for x-axis scaling
        double max_freq;
        if (use_lp_mm && !mtf.is_converted)
        {
            max_freq = mtf.frequencies.back() / (2 * pixel_size_mm);
        }
        else
        {
            max_freq = mtf.frequencies.back();
        }

        // Round max_freq up to nearest nice number for axis labeling
        double nice_max_freq = std::ceil(max_freq * 10) / 10.0;

        // Draw y-axis grid lines and labels
        for (int i = 0; i <= 10; i++)
        {
            double value = 1.0 - (i / 10.0);
            int y = top_margin + static_cast<int>(i * plot_height / 10.0);

            cv::line(visualization,
                     cv::Point(left_margin, y),
                     cv::Point(left_margin + plot_width, y),
                     cv::Scalar(220, 220, 220), 1);

            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << value;
            cv::putText(visualization, ss.str(),
                        cv::Point(5, y + 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }

        // Draw x-axis grid lines and labels
        int num_x_divisions = 10;
        for (int i = 0; i <= num_x_divisions; i++)
        {
            double freq = (static_cast<double>(i) / num_x_divisions) * nice_max_freq;
            int x = left_margin + (i * plot_width) / num_x_divisions;

            cv::line(visualization,
                     cv::Point(x, top_margin),
                     cv::Point(x, top_margin + plot_height),
                     cv::Scalar(220, 220, 220), 1);

            std::stringstream ss;
            ss << std::fixed << std::setprecision(use_lp_mm ? 1 : 2) << freq;
            cv::putText(visualization, ss.str(),
                        cv::Point(x - 20, total_height - 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        }

        // Draw axes
        cv::line(visualization,
                 cv::Point(left_margin, top_margin),
                 cv::Point(left_margin, top_margin + plot_height),
                 cv::Scalar(0, 0, 0), 2);
        cv::line(visualization,
                 cv::Point(left_margin, top_margin + plot_height),
                 cv::Point(left_margin + plot_width, top_margin + plot_height),
                 cv::Scalar(0, 0, 0), 2);

        // Plot MTF curve
        std::vector<cv::Point> points;
        for (size_t i = 0; i < mtf.mtf_values.size(); i++)
        {
            double freq;
            if (use_lp_mm && !mtf.is_converted)
            {
                freq = mtf.frequencies[i] / (2 * pixel_size_mm);
            }
            else
            {
                freq = mtf.frequencies[i];
            }

            if (freq > nice_max_freq)
                break;

            int x = left_margin + static_cast<int>((freq / nice_max_freq) * plot_width);
            int y = top_margin + static_cast<int>((1.0 - mtf.mtf_values[i]) * plot_height);

            if (y >= top_margin && y <= top_margin + plot_height)
            {
                points.push_back(cv::Point(x, y));
            }
        }

        if (points.size() > 1)
        {
            cv::polylines(visualization, points, false,
                          cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }

        // Draw MTF50 line
        int mtf50_y = top_margin + static_cast<int>(0.5 * plot_height);
        cv::line(visualization,
                 cv::Point(left_margin, mtf50_y),
                 cv::Point(left_margin + plot_width, mtf50_y),
                 cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

        // Add axis labels
        cv::putText(visualization, "MTF",
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        std::string x_label = use_lp_mm ? "Spatial Frequency (line pairs/mm)" : "Spatial Frequency (cycles/pixel)";

        cv::putText(visualization, x_label,
                    cv::Point(left_margin + plot_width / 2 - 100, total_height - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        // Add MTF50 value in top-right corner
        std::stringstream mtf50_ss;
        mtf50_ss << "MTF50: " << std::fixed << std::setprecision(use_lp_mm ? 1 : 4)
                 << mtf.getMTF50(use_lp_mm, pixel_size_mm)
                 << (use_lp_mm ? " lp/mm" : " cycles/pixel");
        cv::putText(visualization, mtf50_ss.str(),
                    cv::Point(left_margin + plot_width - 150, top_margin - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        return visualization;
    }

    static cv::Mat analyzeImageMTF(const cv::Mat &input_image, double pixel_size_mm)
    {
        // Validate input parameters
        if (input_image.empty())
        {
            throw std::runtime_error("Input image is empty");
        }
        if (pixel_size_mm <= 0)
        {
            throw std::runtime_error("Pixel size must be positive");
        }

        // Step 1: Detect edges and create ROIs
        std::vector<EdgeROI> edges = detectEdgesAndCreateROIs(input_image);
        if (edges.empty())
        {
            throw std::runtime_error("No suitable edges found in image");
        }

        // Storage for MTF results from all valid edges
        std::vector<MTFData> mtf_results;

        // Process each detected edge
        for (const auto &edge : edges)
        {
            try
            {
                // Step 2: Sample edge profiles
                EdgeProfileData profiles = sampleEdgeProfiles(edge.roi_image, edge);

                // Step 3: Compute super-resolution ESF
                ESFData esf = computeSuperResolutionESF(profiles, edge.angle);

                // Step 4: Compute LSF
                LSFData lsf = computeLSF(esf);

                // Step 5: Compute MTF
                MTFData mtf = computeMTF(lsf);

                // Only include results that meet quality criteria
                if (isValidMTF(mtf, lsf))
                {
                    mtf_results.push_back(mtf);
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error processing edge: " << e.what() << std::endl;
                continue;
            }
        }

        if (mtf_results.empty())
        {
            throw std::runtime_error("No valid MTF results obtained");
        }

        // Average the MTF results
        MTFData averaged_mtf = averageMTFResults(mtf_results);

        // Create visualization with lp/mm units and pixel size
        return createMTFVisualization(averaged_mtf, true, pixel_size_mm);
    }

private:
    static std::vector<double> smoothLSF(const std::vector<double> &input,
                                         int window_size)
    {
        std::vector<double> output(input.size());
        std::vector<double> kernel = createGaussianKernel(window_size);

        int half_window = window_size / 2;

        for (int i = 0; i < static_cast<int>(input.size()); i++)
        {
            double sum = 0.0;
            double weight_sum = 0.0;

            for (int j = -half_window; j <= half_window; j++)
            {
                int idx = i + j;
                if (idx >= 0 && idx < static_cast<int>(input.size()))
                {
                    double weight = kernel[j + half_window];
                    sum += input[idx] * weight;
                    weight_sum += weight;
                }
            }

            output[i] = sum / weight_sum;
        }

        return output;
    }

    static std::vector<double> createGaussianKernel(int size)
    {
        std::vector<double> kernel(size);
        double sigma = size / 6.0; // Makes kernel effectively zero at edges
        int half_size = size / 2;

        for (int i = -half_size; i <= half_size; i++)
        {
            kernel[i + half_size] = exp(-(i * i) / (2 * sigma * sigma));
        }

        return kernel;
    }

    static cv::Mat createLSFVisualization(const LSFData &lsf)
    {
        int plot_height = 400;
        int plot_width = 800;
        cv::Mat visualization(plot_height, plot_width, CV_8UC3,
                              cv::Scalar(255, 255, 255));

        // Draw grid
        for (int i = 0; i < plot_height; i += 50)
        {
            cv::line(visualization, cv::Point(0, i),
                     cv::Point(plot_width, i),
                     cv::Scalar(200, 200, 200), 1);
        }

        // Plot raw LSF points
        std::vector<cv::Point> raw_points;
        for (size_t i = 0; i < lsf.lsf_values.size(); i++)
        {
            int x = (i * plot_width) / lsf.lsf_values.size();
            int y = plot_height - (plot_height * lsf.lsf_values[i]);
            if (y >= 0 && y < plot_height)
            {
                raw_points.push_back(cv::Point(x, y));
                cv::circle(visualization, cv::Point(x, y),
                           1, cv::Scalar(200, 200, 200), -1);
            }
        }

        // Plot smoothed LSF
        std::vector<cv::Point> smooth_points;
        for (size_t i = 0; i < lsf.smoothed_lsf.size(); i++)
        {
            int x = (i * plot_width) / lsf.smoothed_lsf.size();
            int y = plot_height - (plot_height * lsf.smoothed_lsf[i]);
            if (y >= 0 && y < plot_height)
            {
                smooth_points.push_back(cv::Point(x, y));
            }
        }

        if (smooth_points.size() > 1)
        {
            cv::polylines(visualization, smooth_points, false,
                          cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }

        // Draw FWHM
        double peak_val = *std::max_element(lsf.smoothed_lsf.begin(),
                                            lsf.smoothed_lsf.end());
        int half_max_y = plot_height - (plot_height * peak_val / 2);
        cv::line(visualization,
                 cv::Point(0, half_max_y),
                 cv::Point(plot_width, half_max_y),
                 cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

        // Add labels
        cv::putText(visualization, "LSF",
                    cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 1);
        cv::putText(visualization,
                    "FWHM: " + std::to_string(lsf.fwhm),
                    cv::Point(10, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 0), 1);

        return visualization;
    }

    static int nextPowerOfTwo(int n)
    {
        int power = 1;
        while (power < n)
        {
            power *= 2;
        }
        return power;
    }

    static void applyHannWindow(std::vector<std::complex<double>> &data)
    {
        int size = data.size();
        for (int i = 0; i < size; i++)
        {
            double multiplier = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (size - 1)));
            data[i] *= multiplier;
        }
    }

    static std::vector<std::complex<double>> computeFFT(std::vector<std::complex<double>>& data)
    {
        int n = data.size();
        
        // Bit-reverse permutation
        for(int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            while(j >= bit) {
                j -= bit;
                bit >>= 1;
            }
            j += bit;
            if(i < j) std::swap(data[i], data[j]);
        }
        
        // Iterative FFT computation
        for(int len = 2; len <= n; len <<= 1) {
            double angle = -2 * M_PI / len;
            std::complex<double> wlen(std::cos(angle), std::sin(angle));
            
            for(int i = 0; i < n; i += len) {
                std::complex<double> w(1);
                for(int j = 0; j < len/2; j++) {
                    std::complex<double> u = data[i + j];
                    std::complex<double> v = data[i + j + len/2] * w;
                    data[i + j] = u + v;
                    data[i + j + len/2] = u - v;
                    w *= wlen;
                }
            }
        }
        
        return data;
    }

    // Updated helper function for finding MTF frequencies using OpenCV
    static double findMTFFrequency(const std::vector<double>& frequencies,
                                const std::vector<double>& mtf_values,
                                double target_value)
    {
        cv::Mat freq_mat(frequencies);
        cv::Mat mtf_mat(mtf_values);
        
        for (size_t i = 0; i < mtf_values.size() - 1; i++) {
            if (mtf_values[i] >= target_value && mtf_values[i + 1] < target_value) {
                // Linear interpolation using OpenCV's operations
                double t = (target_value - mtf_values[i + 1]) /
                        (mtf_values[i] - mtf_values[i + 1]);
                return frequencies[i + 1] + t * (frequencies[i] - frequencies[i + 1]);
            }
        }
        return 0.0;
    }

    static bool isValidMTF(const MTFData &mtf, const LSFData &lsf)
    {
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

    static MTFData averageMTFResults(const std::vector<MTFData> &results)
    {
        if (results.empty())
        {
            throw std::runtime_error("No MTF results to average");
        }

        MTFData averaged;
        averaged.frequencies = results[0].frequencies;
        averaged.mtf_values.resize(averaged.frequencies.size(), 0.0);

        // Average MTF values
        for (const auto &result : results)
        {
            for (size_t i = 0; i < averaged.mtf_values.size(); i++)
            {
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

int MTFAnalyzer::roi_counter = 0;

int main(int argc, char **argv)
{
    MTFAnalyzer::BenchmarkTimer totalTimer("Total Processing Time");
    
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_image> [pixel_size_mm]\n";
        std::cerr << "Example: " << argv[0] << " image.png 0.00345\n";
        return -1;
    }

    cv::Mat image = cv::imread(argv[1]);
    if (image.empty())
    {
        std::cerr << "Could not open image: " << argv[1] << "\n";
        return -1;
    }

    // Default pixel size if not provided
    double pixel_size_mm = 1.0; // Default will show cycles/pixel

    if (argc == 3)
    {
        try
        {
            pixel_size_mm = std::stod(argv[2]);
            if (pixel_size_mm <= 0)
            {
                throw std::invalid_argument("Pixel size must be positive");
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error parsing pixel size: " << e.what() << "\n";
            std::cerr << "Using default value (cycles/pixel)\n";
            pixel_size_mm = 1.0;
        }
    }

    // Step 1: Edge Detection and ROI Creation
    std::cout << "\n=== Step 1: Edge Detection and ROI Creation ===\n";
    MTFAnalyzer::BenchmarkTimer edgeTimer("Edge Detection");
    std::vector<MTFAnalyzer::EdgeROI> edges = MTFAnalyzer::detectEdgesAndCreateROIs(image, true);
    std::cout << "Found " << edges.size() << " edges at target angles\n";

    // Process each edge
    for (size_t i = 0; i < edges.size(); i++)
    {
        MTFAnalyzer::BenchmarkTimer edgeProcessTimer("Edge " + std::to_string(i) + " Processing");
        std::cout << "\n=== Processing Edge " << i + 1 << " ===\n";
        std::cout << "Edge angle: " << edges[i].angle << " degrees\n";

        // Edge Profile Sampling
        std::cout << "\n--- Step 2: Edge Profile Sampling ---\n";
        MTFAnalyzer::BenchmarkTimer profileTimer("Edge Profile Sampling");
        MTFAnalyzer::EdgeProfileData profiles = MTFAnalyzer::sampleEdgeProfiles(
            edges[i].roi_image,
            edges[i],
            true);

        // Super-Resolution ESF
        std::cout << "\n--- Step 3: Super-Resolution ESF ---\n";
        MTFAnalyzer::BenchmarkTimer esfTimer("ESF Computation");
        MTFAnalyzer::ESFData esf = MTFAnalyzer::computeSuperResolutionESF(
            profiles,
            edges[i].angle,
            true);

        // Line Spread Function
        std::cout << "\n--- Step 4: Line Spread Function ---\n";
        MTFAnalyzer::BenchmarkTimer lsfTimer("LSF Computation");
        MTFAnalyzer::LSFData lsf = MTFAnalyzer::computeLSF(esf, true);

        // MTF Calculation and Visualization
        std::cout << "\n--- Step 5: MTF Calculation ---\n";
        MTFAnalyzer::BenchmarkTimer mtfTimer("MTF Calculation");
        MTFAnalyzer::MTFData mtf = MTFAnalyzer::computeMTF(lsf, true);

        // Convert to lp/mm if pixel size is provided
        if (pixel_size_mm != 1.0)
        {
            mtf.convertToLPMM(pixel_size_mm);
        }

        std::cout << "MTF Results:\n";
        if (pixel_size_mm != 1.0)
        {
            std::cout << "Using pixel size: " << pixel_size_mm << " mm\n";
            std::cout << "MTF50: " << mtf.getMTF50(true, pixel_size_mm) << " lp/mm\n";
            std::cout << "MTF20: " << mtf.mtf20 << " lp/mm\n";
            std::cout << "MTF10: " << mtf.mtf10 << " lp/mm\n";
        }
        else
        {
            std::cout << "MTF50: " << mtf.getMTF50() << " cycles/pixel\n";
            std::cout << "MTF20: " << mtf.mtf20 << " cycles/pixel\n";
            std::cout << "MTF10: " << mtf.mtf10 << " cycles/pixel\n";
        }

        // Create and save visualization
        cv::Mat mtf_vis = MTFAnalyzer::createMTFVisualization(mtf, pixel_size_mm != 1.0, pixel_size_mm);
        std::string mtf_filename = "opencv-mtf_" + std::to_string(i) + ".png";
        cv::imwrite(mtf_filename, mtf_vis);
        std::cout << "Saved MTF visualization to: " << mtf_filename << "\n";

        // Save intermediate visualizations
        if (!profiles.visualization.empty()) {
            std::string profile_filename = "opencv-edge_profiles_" + std::to_string(i) + ".png";
            cv::imwrite(profile_filename, profiles.visualization);
            std::cout << "Saved edge profiles visualization to: " << profile_filename << "\n";
        }

        if (!esf.visualization.empty()) {
            std::string esf_filename = "opencv-super_res_esf_" + std::to_string(i) + ".png";
            cv::imwrite(esf_filename, esf.visualization);
            std::cout << "Saved super-resolution ESF visualization to: " << esf_filename << "\n";
        }

        if (!lsf.visualization.empty()) {
            std::string lsf_filename = "opencv-lsf_" + std::to_string(i) + ".png";
            cv::imwrite(lsf_filename, lsf.visualization);
            std::cout << "Saved LSF visualization to: " << lsf_filename << "\n";
        }
    }

    // Try unified analysis approach
    try
    {
        cv::Mat mtf_plot = MTFAnalyzer::analyzeImageMTF(image, pixel_size_mm);
        cv::imwrite("opencv-mtf_unified.png", mtf_plot);
        std::cout << "\nSaved unified MTF analysis to: mtf_unified.png\n";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error in unified analysis: " << e.what() << "\n";
    }

    return 0;
}