#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>   // for std::iota, std::accumulate
#include <algorithm> // for std::sort
#include <map>      // for std::map
#include <cmath>    // for mathematical operations

class MTFAnalyzer {
private:
    static constexpr double ANGLE1_TARGET = 11.0;
    static constexpr double ANGLE2_TARGET = 281.0;
    static constexpr double ANGLE_TOLERANCE = 4.0;
    static constexpr double ROI_LENGTH_FACTOR = 0.8;
    static constexpr double ROI_WIDTH = 100;
    static int roi_counter;

public:

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
        cv::Mat visualization;  // For debugging
    };

    struct LSFData {
        std::vector<double> distances;     // X-axis positions
        std::vector<double> lsf_values;    // LSF values
        std::vector<double> smoothed_lsf;  // Smoothed LSF values
        double fwhm;                       // Full Width at Half Maximum
        cv::Mat visualization;             // Debug visualization
    };

    struct MTFData {
        std::vector<double> frequencies;
        std::vector<double> mtf_values;
        double mtf50;
        double mtf20;
        double mtf10;
        cv::Mat visualization;
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
    
    static std::vector<EdgeROI> detectEdgesAndCreateROIs(const cv::Mat& input, bool debug = false) {
        std::vector<EdgeROI> detected_edges;
        cv::Mat masked = input.clone();
        
        // Mask central region
        int radius = std::min(input.rows, input.cols) / 100 * 5;
        cv::Point center(input.cols/2, input.rows/2);
        cv::circle(masked, center, radius, cv::Scalar(0,0,0), -1);
        
        if(debug) {
            cv::imwrite("_1_masked_input.png", masked);
        }
        
        // Edge detection with adaptive parameters
        cv::Mat gray;
        cv::cvtColor(masked, gray, cv::COLOR_BGR2GRAY);
        double mean_intensity = cv::mean(gray)[0];
        int canny_low = std::max(20, static_cast<int>(mean_intensity * 0.2));
        int canny_high = std::min(200, static_cast<int>(mean_intensity * 0.6));
        
        cv::Mat edges;
        cv::Canny(gray, edges, canny_low, canny_high);
        
        if(debug) {
            cv::imwrite("_2_detected_edges.png", edges);
        }
        
        // Line detection with adaptive parameters
        std::vector<cv::Vec4i> lines;
        int min_line_length = std::min(input.rows, input.cols) / 8;
        int max_line_gap = std::min(input.rows, input.cols) / 16;
        cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, min_line_length, max_line_gap);
        
        if(debug) {
            cv::Mat lines_image = masked.clone();
            for(const auto& l : lines) {
                cv::line(lines_image, cv::Point(l[0], l[1]),
                        cv::Point(l[2], l[3]), cv::Scalar(0,255,0), 2);
            }
            cv::imwrite("_3_detected_lines.png", lines_image);
            std::cout << "Found " << lines.size() << " total lines" << std::endl;
        }
        
        cv::Mat rois_image;
        if(debug) {
            rois_image = input.clone();
        }

        // Group lines by quadrant
        std::map<int, std::vector<cv::Vec4i>> quadrant_groups;
        cv::Point2f image_center(input.cols/2.0f, input.rows/2.0f);

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
            
            auto longest_line = *std::max_element(group.second.begin(), group.second.end(),
                [](const cv::Vec4i& a, const cv::Vec4i& b) {
                    double len_a = std::hypot(a[2]-a[0], a[3]-a[1]);
                    double len_b = std::hypot(b[2]-b[0], b[3]-b[1]);
                    return len_a < len_b;
                });
            
            EdgeROI roi_data;
            roi_data.line = longest_line;
            double angle = std::atan2(-(longest_line[3] - longest_line[1]), 
                                    longest_line[2] - longest_line[0]) * 180.0 / CV_PI;
            while(angle < 0) angle += 360.0;
            roi_data.angle = angle;
            
            cv::Point2f line_center((longest_line[0] + longest_line[2])/2.0f, 
                                (longest_line[1] + longest_line[3])/2.0f);
            double length = std::sqrt(std::pow(longest_line[2]-longest_line[0], 2) + 
                                    std::pow(longest_line[3]-longest_line[1], 2));
            double roi_length = length * ROI_LENGTH_FACTOR;
            
            double narrow_dim = std::min(input.rows, input.cols) * 0.125;
            int roi_width = std::abs(angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE ? 
                        roi_length : narrow_dim;
            int roi_height = std::abs(angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE ? 
                        narrow_dim : roi_length;
            
            roi_data.roi = cv::Rect(
                static_cast<int>(line_center.x - roi_width/2),
                static_cast<int>(line_center.y - roi_height/2),
                roi_width,
                roi_height
            );
            
            roi_data.roi &= cv::Rect(0, 0, input.cols, input.rows);
            roi_data.roi_image = input(roi_data.roi).clone();
            detected_edges.push_back(roi_data);
            
            if(debug) {
                cv::line(rois_image, cv::Point(longest_line[0], longest_line[1]),
                        cv::Point(longest_line[2], longest_line[3]), 
                        cv::Scalar(0,255,0), 2);
                cv::rectangle(rois_image, roi_data.roi, cv::Scalar(0,0,255), 2);
            }
        }
        
        if(debug) {
            cv::imwrite("_4_rois.png", rois_image);
            std::cout << "Found " << detected_edges.size() << " edges at target angles" << std::endl;
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
        
        

    static ESFData computeSuperResolutionESF(const EdgeProfileData& profiles, double angle_degrees, bool debug = false)
    {
        ESFData result;
        const double BIN_WIDTH = 0.1;
        
        std::cout << "\nStarting ESF computation..." << std::endl;
        std::cout << "Number of input profiles: " << profiles.profiles.size() << std::endl;
        
        if (!profiles.profiles.empty()) {
            std::cout << "First profile length: " << profiles.profiles[0].size() << std::endl;
            
            // Pre-allocate memory based on expected size
            size_t total_points = profiles.profiles.size() * profiles.profiles[0].size();
            result.distances.reserve(total_points);
            result.intensities.reserve(total_points);
        }

        // Find global min/max using OpenCV
        cv::Mat all_profiles;
        for (const auto& profile : profiles.profiles) {
            if (!profile.empty()) {
                cv::Mat profile_mat(profile);
                all_profiles.push_back(profile_mat);
            }
        }
        
        double global_min, global_max;
        cv::minMaxLoc(all_profiles, &global_min, &global_max);
        
        std::cout << "Global min: " << global_min << ", Global max: " << global_max << std::endl;

        // Process profiles in parallel using OpenCV
        cv::parallel_for_(cv::Range(0, profiles.profiles.size()), [&](const cv::Range& range) {
            for (int profile_idx = range.start; profile_idx < range.end; profile_idx++) {
                const auto& profile = profiles.profiles[profile_idx];
                if (profile.empty()) continue;

                // Convert profile to OpenCV Mat for efficient processing
                cv::Mat profile_mat(profile);
                cv::Mat gradients;
                
                // Calculate gradients using OpenCV
                cv::Sobel(profile_mat, gradients, CV_64F, 1, 0, 3);
                cv::abs(gradients);
                
                // Find edge location
                double max_grad;
                cv::Point max_loc;
                cv::minMaxLoc(gradients, nullptr, &max_grad, nullptr, &max_loc);
                int edge_idx = max_loc.x;

                // Create normalized samples centered on edge
                cv::Mat normalized;
                cv::subtract(profile_mat, global_min, normalized);
                cv::divide(normalized, (global_max - global_min), normalized);

                // Store results with mutex protection
                static std::mutex mtx;
                std::lock_guard<std::mutex> lock(mtx);
                
                for (size_t i = 0; i < profile.size(); i++) {
                    result.distances.push_back(static_cast<double>(i) - edge_idx);
                    result.intensities.push_back(normalized.at<double>(i));
                }
            }
        });

        // Sort by distance using parallel algorithms
        std::vector<size_t> indices(result.distances.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(std::execution::par_unseq, indices.begin(), indices.end(),
            [&](size_t a, size_t b) {
                return result.distances[a] < result.distances[b];
            });

        // Pre-allocate memory for binning
        result.binned_distances.reserve(result.distances.size() / 10);  // Estimate bin count
        result.binned_intensities.reserve(result.distances.size() / 10);

        // Apply adaptive binning using OpenCV
        cv::Mat distances_mat(result.distances);
        cv::Mat intensities_mat(result.intensities);
        double min_dist = *std::min_element(result.distances.begin(), result.distances.end());

        std::map<int, std::vector<double>> bins;
        for (size_t i = 0; i < result.distances.size(); i++) {
            int bin = static_cast<int>((result.distances[indices[i]] - min_dist) / BIN_WIDTH);
            bins[bin].push_back(result.intensities[indices[i]]);
        }

        // Calculate bin statistics using OpenCV
        for (const auto& bin : bins) {
            if (bin.second.size() >= 3) {
                cv::Mat bin_values(bin.second);
                cv::sort(bin_values, bin_values, cv::SORT_ASCENDING);
                double median = bin_values.at<double>(bin_values.rows / 2);

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

        // Pre-allocate memory
        const size_t output_size = esf.binned_distances.size() - 4;
        result.distances.reserve(output_size);
        result.lsf_values.reserve(output_size);
        result.smoothed_lsf.reserve(output_size);

        // Convert ESF data to OpenCV Mat for efficient processing
        cv::Mat esf_distances(esf.binned_distances);
        cv::Mat esf_intensities(esf.binned_intensities);
        cv::Mat_<double> derivative(1, output_size);

        // Compute derivative using 5-point stencil with OpenCV operations
        cv::parallel_for_(cv::Range(2, esf.binned_distances.size() - 2), [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; i++) {
                double h = esf_distances.at<double>(i + 1) - esf_distances.at<double>(i);
                if (h != 0) {
                    derivative.at<double>(i-2) = (-esf_intensities.at<double>(i + 2) +
                                                8 * esf_intensities.at<double>(i + 1) -
                                                8 * esf_intensities.at<double>(i - 1) +
                                                esf_intensities.at<double>(i - 2)) / (12 * h);
                }
            }
        });

        // Store distances and take absolute values of derivatives
        cv::Mat abs_derivative;
        cv::abs(derivative, abs_derivative);

        // Copy results to output vectors
        cv::Mat distances_roi = esf_distances(cv::Range(2, esf_distances.rows - 2));
        distances_roi.copyTo(cv::Mat(result.distances));
        abs_derivative.copyTo(cv::Mat(result.lsf_values));

        if (debug) {
            std::cout << "Computed LSF points: " << result.lsf_values.size() << std::endl;
        }

        // Create and normalize Gaussian kernel using OpenCV
        const int window = 5;
        const double sigma = 1.0;
        cv::Mat kernel = cv::getGaussianKernel(window, sigma, CV_64F);
        cv::normalize(kernel, kernel, 1.0, 0.0, cv::NORM_L1);

        if (debug) {
            std::cout << "Created Gaussian kernel of size " << window << std::endl;
        }

        // Apply smoothing using OpenCV's filter2D
        cv::Mat lsf_mat(result.lsf_values);
        cv::Mat smoothed_mat;
        cv::filter2D(lsf_mat, smoothed_mat, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_REFLECT);
        
        // Copy smoothed results
        smoothed_mat.copyTo(cv::Mat(result.smoothed_lsf));

        // Normalize LSF using OpenCV operations
        if (!result.smoothed_lsf.empty()) {
            double max_val;
            cv::minMaxLoc(smoothed_mat, nullptr, &max_val);
            
            if (debug) {
                std::cout << "Max LSF value before normalization: " << max_val << std::endl;
            }

            if (max_val > 0) {
                cv::divide(smoothed_mat, max_val, smoothed_mat);
                smoothed_mat.copyTo(cv::Mat(result.smoothed_lsf));
            }
        }

        // Calculate FWHM using optimized method
        result.fwhm = calculateFWHM(result.distances, result.smoothed_lsf);
        
        if (debug) {
            std::cout << "Calculated FWHM: " << result.fwhm << std::endl;
            
            // Create visualization using OpenCV
            const int plot_height = 400;
            const int plot_width = 800;
            result.visualization = cv::Mat(plot_height, plot_width, CV_8UC3, cv::Scalar(255, 255, 255));

            // Draw grid with anti-aliasing
            for (int i = 0; i < plot_height; i += 50) {
                cv::line(result.visualization, cv::Point(0, i),
                        cv::Point(plot_width, i), cv::Scalar(200, 200, 200), 
                        1, cv::LINE_AA);
            }

            // Draw half-max line
            int half_max_y = plot_height - (plot_height * 0.5);
            cv::line(result.visualization,
                    cv::Point(0, half_max_y),
                    cv::Point(plot_width, half_max_y),
                    cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

            // Plot smoothed LSF
            std::vector<cv::Point> points;
            points.reserve(result.smoothed_lsf.size());
            
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

            // Add labels
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

    static MTFData computeMTF(const LSFData& lsf, bool debug = false)
    {
        MTFData result;

        // Ensure we have valid LSF data
        if (lsf.smoothed_lsf.empty() || lsf.smoothed_lsf.size() < 4) {
            std::cout << "Invalid LSF data for MTF calculation" << std::endl;
            return result;
        }

        // Convert LSF data to OpenCV Mat
        cv::Mat lsf_data(1, lsf.smoothed_lsf.size(), CV_64F);
        cv::Mat(lsf.smoothed_lsf).copyTo(lsf_data.row(0));

        // Get optimal DFT size and pad data
        int dft_size = cv::getOptimalDFTSize(lsf_data.cols);
        cv::Mat padded;
        cv::copyMakeBorder(lsf_data, padded, 0, 0, 0, dft_size - lsf_data.cols, 
                        cv::BORDER_CONSTANT, cv::Scalar(0));

        // Create and apply Hann window using OpenCV operations
        cv::Mat hann(1, dft_size, CV_64F);
        for (int i = 0; i < dft_size; i++) {
            hann.at<double>(0, i) = 0.5 * (1.0 - cos(2.0 * CV_PI * i / (dft_size - 1)));
        }
        padded = padded.mul(hann);

        // Prepare matrices for DFT
        cv::Mat planes[] = {padded, cv::Mat::zeros(padded.size(), CV_64F)};
        cv::Mat complexMat;
        cv::merge(planes, 2, complexMat);

        // Perform DFT with OpenCV's optimized implementation
        cv::dft(complexMat, complexMat, cv::DFT_SCALE);

        // Split into real and imaginary parts
        cv::split(complexMat, planes);

        // Compute magnitude using OpenCV
        cv::Mat magnitude;
        cv::magnitude(planes[0], planes[1], magnitude);

        // Extract first half (up to Nyquist frequency)
        magnitude = magnitude(cv::Rect(0, 0, magnitude.cols/2, 1));

        // Normalize using OpenCV
        double maxVal;
        cv::minMaxLoc(magnitude, nullptr, &maxVal);
        magnitude /= maxVal;

        // Pre-allocate vectors
        result.frequencies.resize(magnitude.cols);
        result.mtf_values.resize(magnitude.cols);

        // Copy results to output vectors
        cv::Mat frequencies(1, magnitude.cols, CV_64F);
        for (int i = 0; i < magnitude.cols; i++) {
            frequencies.at<double>(0, i) = static_cast<double>(i) / dft_size;
        }
        
        frequencies.copyTo(cv::Mat(result.frequencies));
        magnitude.copyTo(cv::Mat(result.mtf_values));

        // Calculate MTF metrics using optimized findMTFFrequency
        result.mtf50 = findMTFFrequency(result.frequencies, result.mtf_values, 0.5);
        result.mtf20 = findMTFFrequency(result.frequencies, result.mtf_values, 0.2);
        result.mtf10 = findMTFFrequency(result.frequencies, result.mtf_values, 0.1);

        if (debug) {
            result.visualization = createMTFVisualization(result);
        }

        return result;
    }

    static cv::Mat createMTFVisualization(const MTFData& mtf, bool use_lp_mm = true, double pixel_size_mm = 1.0) {
        // Increased margins and plot dimensions
        int plot_height = 400;
        int plot_width = 800;
        int left_margin = 80;   
        int right_margin = 40;  
        int top_margin = 40;    
        int bottom_margin = 60; 
        
        int total_width = left_margin + plot_width + right_margin;
        int total_height = top_margin + plot_height + bottom_margin;
        
        cv::Mat visualization(total_height, total_width, CV_8UC3, cv::Scalar(255,255,255));

        // Calculate maximum frequency in lp/mm for x-axis scaling
        double max_freq;
        if (use_lp_mm && !mtf.is_converted) {
            max_freq = mtf.frequencies.back() / (2 * pixel_size_mm);
        } else {
            max_freq = mtf.frequencies.back();
        }
        
        // Round max_freq up to nearest nice number for axis labeling
        double nice_max_freq = std::ceil(max_freq * 10) / 10.0;

        // Draw y-axis grid lines and labels
        for(int i = 0; i <= 10; i++) {
            double value = 1.0 - (i / 10.0);
            int y = top_margin + static_cast<int>(i * plot_height / 10.0);
            
            cv::line(visualization, 
                    cv::Point(left_margin, y), 
                    cv::Point(left_margin + plot_width, y), 
                    cv::Scalar(220,220,220), 1);
            
            std::stringstream ss;
            ss << std::fixed << std::setprecision(1) << value;
            cv::putText(visualization, ss.str(),
                    cv::Point(5, y + 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(0,0,0), 1, cv::LINE_AA);
        }
        
        // Draw x-axis grid lines and labels
        int num_x_divisions = 10;
        for(int i = 0; i <= num_x_divisions; i++) {
            double freq = (static_cast<double>(i) / num_x_divisions) * nice_max_freq;
            int x = left_margin + (i * plot_width) / num_x_divisions;
            
            cv::line(visualization,
                    cv::Point(x, top_margin),
                    cv::Point(x, top_margin + plot_height),
                    cv::Scalar(220,220,220), 1);
            
            std::stringstream ss;
            ss << std::fixed << std::setprecision(use_lp_mm ? 1 : 2) << freq;
            cv::putText(visualization, ss.str(),
                    cv::Point(x - 20, total_height - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4,
                    cv::Scalar(0,0,0), 1, cv::LINE_AA);
        }

        // Draw axes
        cv::line(visualization,
                cv::Point(left_margin, top_margin),
                cv::Point(left_margin, top_margin + plot_height),
                cv::Scalar(0,0,0), 2);
        cv::line(visualization,
                cv::Point(left_margin, top_margin + plot_height),
                cv::Point(left_margin + plot_width, top_margin + plot_height),
                cv::Scalar(0,0,0), 2);

        // Plot MTF curve
        std::vector<cv::Point> points;
        for(size_t i = 0; i < mtf.mtf_values.size(); i++) {
            double freq;
            if (use_lp_mm && !mtf.is_converted) {
                freq = mtf.frequencies[i] / (2 * pixel_size_mm);
            } else {
                freq = mtf.frequencies[i];
            }
                
            if(freq > nice_max_freq) break;
            
            int x = left_margin + static_cast<int>((freq / nice_max_freq) * plot_width);
            int y = top_margin + static_cast<int>((1.0 - mtf.mtf_values[i]) * plot_height);
            
            if(y >= top_margin && y <= top_margin + plot_height) {
                points.push_back(cv::Point(x, y));
            }
        }

        if(points.size() > 1) {
            cv::polylines(visualization, points, false, 
                        cv::Scalar(0,0,255), 2, cv::LINE_AA);
        }

        // Draw MTF50 line
        int mtf50_y = top_margin + static_cast<int>(0.5 * plot_height);
        cv::line(visualization, 
                cv::Point(left_margin, mtf50_y), 
                cv::Point(left_margin + plot_width, mtf50_y), 
                cv::Scalar(0,255,0), 1, cv::LINE_AA);

        // Add axis labels
        cv::putText(visualization, "MTF",
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, cv::Scalar(0,0,0), 1, cv::LINE_AA);

        std::string x_label = use_lp_mm ? 
            "Spatial Frequency (line pairs/mm)" : 
            "Spatial Frequency (cycles/pixel)";
                
        cv::putText(visualization, x_label,
                cv::Point(left_margin + plot_width/2 - 100, total_height - 5),
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, cv::Scalar(0,0,0), 1, cv::LINE_AA);

        // Add MTF50 value in top-right corner
        std::stringstream mtf50_ss;
        mtf50_ss << "MTF50: " << std::fixed << std::setprecision(use_lp_mm ? 1 : 4) 
                << mtf.getMTF50(use_lp_mm, pixel_size_mm)
                << (use_lp_mm ? " lp/mm" : " cycles/pixel");
        cv::putText(visualization, mtf50_ss.str(),
                cv::Point(left_margin + plot_width - 150, top_margin - 10),
                cv::FONT_HERSHEY_SIMPLEX, 
                0.5, cv::Scalar(0,0,0), 1, cv::LINE_AA);

        return visualization;
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

    // Optimized helper function for finding MTF frequencies
    static double findMTFFrequency(const std::vector<double>& frequencies,
                                const std::vector<double>& mtf_values,
                                double target_value)
    {
        cv::Mat freq_mat(frequencies);
        cv::Mat mtf_mat(mtf_values);
        
        // Use OpenCV's efficient array operations
        cv::Mat diff_mat;
        cv::subtract(mtf_mat, target_value, diff_mat);
        
        // Find zero crossing
        for (int i = 0; i < diff_mat.cols - 1; i++) {
            double curr = diff_mat.at<double>(0, i);
            double next = diff_mat.at<double>(0, i + 1);
            
            if (curr >= 0 && next < 0) {
                // Linear interpolation
                double t = curr / (curr - next);
                return frequencies[i] + t * (frequencies[i + 1] - frequencies[i]);
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

int MTFAnalyzer::roi_counter = 0;

int main(int argc, char** argv) {
    
    MTFAnalyzer::BenchmarkTimer totalTimer("Total Processing Time");

    if(argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> [pixel_size_mm]\n";
        std::cerr << "Example: " << argv[0] << " image.png 0.00345\n";
        return -1;
    }
    
    cv::Mat image = cv::imread(argv[1]);
    if(image.empty()) {
        std::cerr << "Could not open image: " << argv[1] << "\n";
        return -1;
    }

    // Default pixel size if not provided
    double pixel_size_mm = 1.0;  // Default will show cycles/pixel
    
    if(argc == 3) {
        try {
            pixel_size_mm = std::stod(argv[2]);
            if(pixel_size_mm <= 0) {
                throw std::invalid_argument("Pixel size must be positive");
            }
        } catch(const std::exception& e) {
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
    for(size_t i = 0; i < edges.size(); i++) {
        MTFAnalyzer::BenchmarkTimer edgeProcessTimer("Edge " + std::to_string(i) + " Processing");
        std::cout << "\n=== Processing Edge " << i + 1 << " ===\n";
        std::cout << "Edge angle: " << edges[i].angle << " degrees\n";
        
        // Edge Profile Sampling
        std::cout << "\n--- Step 2: Edge Profile Sampling ---\n";
        MTFAnalyzer::BenchmarkTimer profileTimer("Edge Profile Sampling");
        MTFAnalyzer::EdgeProfileData profiles = MTFAnalyzer::sampleEdgeProfiles(
            edges[i].roi_image,
            edges[i],
            true
        );
        
        // Super-Resolution ESF
        std::cout << "\n--- Step 3: Super-Resolution ESF ---\n";
        MTFAnalyzer::BenchmarkTimer esfTimer("ESF Computation");
        MTFAnalyzer::ESFData esf = MTFAnalyzer::computeSuperResolutionESF(
            profiles,
            edges[i].angle,
            true
        );
        
        // Line Spread Function
        std::cout << "\n--- Step 4: Line Spread Function ---\n";
        MTFAnalyzer::BenchmarkTimer lsfTimer("LSF Computation");
        MTFAnalyzer::LSFData lsf = MTFAnalyzer::computeLSF(esf, true);
        
        // MTF Calculation and Visualization
        std::cout << "\n--- Step 5: MTF Calculation ---\n";
        MTFAnalyzer::BenchmarkTimer mtfTimer("MTF Calculation");
        MTFAnalyzer::MTFData mtf = MTFAnalyzer::computeMTF(lsf, true);

        // Convert to lp/mm if pixel size is provided
        if(pixel_size_mm != 1.0) {
            mtf.convertToLPMM(pixel_size_mm);
        }

        std::cout << "MTF Results:\n";
        if(pixel_size_mm != 1.0) {
            std::cout << "Using pixel size: " << pixel_size_mm << " mm\n";
            std::cout << "MTF50: " << mtf.getMTF50(true, pixel_size_mm) << " lp/mm\n";
            std::cout << "MTF20: " << mtf.mtf20 << " lp/mm\n";
            std::cout << "MTF10: " << mtf.mtf10 << " lp/mm\n";
        } else {
            std::cout << "MTF50: " << mtf.getMTF50() << " cycles/pixel\n";
            std::cout << "MTF20: " << mtf.mtf20 << " cycles/pixel\n";
            std::cout << "MTF10: " << mtf.mtf10 << " cycles/pixel\n";
        }

        // Create and save visualization
        cv::Mat mtf_vis = MTFAnalyzer::createMTFVisualization(mtf, pixel_size_mm != 1.0, pixel_size_mm);
        std::string mtf_filename = "mtf_" + std::to_string(i) + ".png";
        cv::imwrite(mtf_filename, mtf_vis);
        std::cout << "Saved MTF visualization to: " << mtf_filename << "\n";

        // Save intermediate visualizations
        if(!profiles.visualization.empty()) {
            std::string profile_filename = "edge_profiles_" + std::to_string(i) + ".png";
            cv::imwrite(profile_filename, profiles.visualization);
            std::cout << "Saved edge profiles visualization to: " << profile_filename << "\n";
        }
        
        if(!esf.visualization.empty()) {
            std::string esf_filename = "super_res_esf_" + std::to_string(i) + ".png";
            cv::imwrite(esf_filename, esf.visualization);
            std::cout << "Saved super-resolution ESF visualization to: " << esf_filename << "\n";
        }
        
        if(!lsf.visualization.empty()) {
            std::string lsf_filename = "lsf_" + std::to_string(i) + ".png";
            cv::imwrite(lsf_filename, lsf.visualization);
            std::cout << "Saved LSF visualization to: " << lsf_filename << "\n";
        }
    }

    // Try unified analysis approach
    try {
        cv::Mat mtf_plot = MTFAnalyzer::analyzeImageMTF(image, pixel_size_mm);
        cv::imwrite("mtf_unified.png", mtf_plot);
        std::cout << "\nSaved unified MTF analysis to: mtf_unified.png\n";
    } catch(const std::exception& e) {
        std::cerr << "Error in unified analysis: " << e.what() << "\n";
    }
    
    return 0;
}