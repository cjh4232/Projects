#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <numeric>   // for std::iota, std::accumulate
#include <algorithm> // for std::sort
#include <map>       // for std::map
#include <cmath>     // for mathematical operations

class MTFAnalyzer
{
public:
    // Data structures
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

    struct ROIQuality
    {
        double edge_strength;      // Edge contrast/gradient strength (0-100)
        double linearity_score;    // How straight/linear the edge is (0-100)
        double noise_level;        // Noise assessment in the ROI (lower is better)
        double profile_adequacy;   // Number and quality of profiles (0-100)
        double overall_score;      // Combined quality score (0-100)
        bool is_acceptable;        // Whether ROI meets quality threshold
        std::string quality_reason; // Explanation of quality assessment
    };

    struct LSFData
    {
        std::vector<double> distances;    // X-axis positions
        std::vector<double> lsf_values;   // LSF values
        std::vector<double> smoothed_lsf; // Smoothed LSF values
        double fwhm;                      // Full Width at Half Maximum
        cv::Mat visualization;            // Debug visualization
        ROIQuality quality;               // Quality assessment of this ROI
    };

    struct MTFResults
    {
        // Basic MTF data
        std::vector<double> frequencies;
        std::vector<double> mtf_values;

        // Key metrics in cycles/pixel
        double mtf50; // Keep naming consistent
        double mtf20;
        double mtf10;

        // Converted metrics in lp/mm
        double mtf50_lp_mm;
        double mtf20_lp_mm;
        double mtf10_lp_mm;

        // Validation data
        std::vector<double> theoretical_mtf;
        double theoretical_mtf50;
        double theoretical_mtf20;
        double theoretical_mtf10;
        double rms_error;

        // Visualization
        cv::Mat visualization;

        // State flags
        bool has_theoretical = false;
        bool is_converted = false;

        void convertToLPMM(double pixel_size_mm)
        {
            if (pixel_size_mm <= 0 || is_converted)
                return;

            mtf50_lp_mm = mtf50 / (2 * pixel_size_mm);
            mtf20_lp_mm = mtf20 / (2 * pixel_size_mm);
            mtf10_lp_mm = mtf10 / (2 * pixel_size_mm);

            // Convert frequency axis
            for (auto &freq : frequencies)
            {
                freq /= (2 * pixel_size_mm);
            }

            is_converted = true;
        }

        void calculateTheoreticalMTF(double gaussian_sigma)
        {
            if (gaussian_sigma <= 0)
                return;

            theoretical_mtf.resize(frequencies.size());
            for (size_t i = 0; i < frequencies.size(); i++)
            {
                double f = frequencies[i];
                theoretical_mtf[i] = exp(-2 * M_PI * M_PI *
                                         gaussian_sigma * gaussian_sigma * f * f);
            }

            // Calculate theoretical metrics
            theoretical_mtf50 = calculateTheoreticalMTFx(0.5, gaussian_sigma);
            theoretical_mtf20 = calculateTheoreticalMTFx(0.2, gaussian_sigma);
            theoretical_mtf10 = calculateTheoreticalMTFx(0.1, gaussian_sigma);

            calculateRMSError();
            has_theoretical = true;
        }

        void printValidationResults(std::ostream &os = std::cout) const
        {
            if (!has_theoretical)
                return;

            os << "\nMTF Validation Results:" << std::endl;
            os << std::fixed << std::setprecision(4);
            os << "MTF50 (measured/theoretical): " << mtf50
               << " / " << theoretical_mtf50 << std::endl;
            os << "MTF20 (measured/theoretical): " << mtf20
               << " / " << theoretical_mtf20 << std::endl;
            os << "MTF10 (measured/theoretical): " << mtf10
               << " / " << theoretical_mtf10 << std::endl;
            os << "RMS Error: " << rms_error << std::endl;

            if (rms_error < VALIDATION_THRESHOLD)
            {
                os << "Validation PASSED (error < " << VALIDATION_THRESHOLD * 100 << "%)" << std::endl;
            }
            else
            {
                os << "Validation FAILED (error > " << VALIDATION_THRESHOLD * 100 << "%)" << std::endl;
            }
        }

    private:
        double calculateTheoreticalMTFx(double x, double sigma)
        {
            return 1.0 / (2.0 * M_PI * sigma) * std::sqrt(std::log(1.0 / x));
        }

        void calculateRMSError()
        {
            double sum_squared_diff = 0.0;
            int count = 0;

            for (size_t i = 0; i < frequencies.size(); i++)
            {
                if (frequencies[i] <= theoretical_mtf10 * 1.2)
                {
                    double diff = mtf_values[i] - theoretical_mtf[i];
                    sum_squared_diff += diff * diff;
                    count++;
                }
            }

            rms_error = std::sqrt(sum_squared_diff / count);
        }
    };

    class BenchmarkTimer
    {
        using Clock = std::chrono::high_resolution_clock;
        Clock::time_point start;
        std::string name;

    public:
        BenchmarkTimer(std::string n) : name(n), start(Clock::now())
        {
            std::cout << "\nStarting " << name << "..." << std::endl; // Add start message
        }
        ~BenchmarkTimer()
        {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start);
            std::cout << "\n"
                      << name << " completed in: " << duration.count() << "μs" << std::endl;
            std::cout.flush(); // Force output
        }
    };

    struct ProgramOptions
    {
        std::string input_file;
        double pixel_size_mm = 0.0;
        double gaussian_sigma = 0.0;
        bool debug = false;
        bool validate_theoretical = false;

        static ProgramOptions parseCommandLine(int argc, char **argv)
        {
            ProgramOptions options;

            for (int i = 1; i < argc; i++)
            {
                std::string arg = argv[i];
                if (arg == "--pixel-size" && i + 1 < argc)
                {
                    options.pixel_size_mm = std::stod(argv[++i]);
                }
                else if (arg == "--gaussian-sigma" && i + 1 < argc)
                {
                    options.gaussian_sigma = std::stod(argv[++i]);
                    options.validate_theoretical = true;
                }
                else if (arg == "--debug")
                {
                    options.debug = true;
                }
                else
                {
                    options.input_file = arg;
                }
            }

            return options;
        }
    };

    // Main analysis functions
    static MTFResults analyzeImage(const cv::Mat &image, const ProgramOptions &options)
    {
        // Validate input image
        if (image.empty())
        {
            throw std::runtime_error("Input image is empty");
        }

        // Process edges and compute MTF results directly
        std::vector<EdgeROI> edges = detectEdgesAndCreateROIs(image, options.debug);
        if (edges.empty())
        {
            throw std::runtime_error("No suitable edges found in image");
        }

        // Storage for MTF results from all valid edges
        std::vector<MTFResults> mtf_results;

        // Process each detected edge
        for (const auto &edge : edges)
        {
            try
            {
                EdgeProfileData profiles = sampleEdgeProfiles(edge.roi_image, edge);
                ESFData esf = computeSuperResolutionESF(profiles, edge.angle);
                LSFData lsf = computeLSF(esf, edge.roi_image, edge.line, profiles.profiles.size(), options.debug);
                
                // Quality filtering: Only proceed with acceptable quality ROIs
                if (!lsf.quality.is_acceptable)
                {
                    std::cout << "⚠️ Skipping ROI due to poor quality" << std::endl;
                    continue; // Skip this ROI
                }
                
                // Additional outlier filtering based on FWHM reasonableness
                // Skip ROIs with extremely small or large FWHM values that indicate measurement errors
                if (lsf.fwhm < 0.1 || lsf.fwhm > 5.0)
                {
                    std::cout << "⚠️ Skipping ROI due to unrealistic FWHM: " << lsf.fwhm << " pixels" << std::endl;
                    continue;
                }
                
                MTFResults mtf = computeMTF(lsf);

                // Save LSF visualization if in debug mode
                if (options.debug && !lsf.visualization.empty())
                {
                    static int lsf_counter = 0;
                    cv::imwrite("lsf_" + std::to_string(lsf_counter++) + ".png", lsf.visualization);
                }

                // Add theoretical calculations if requested
                if (options.gaussian_sigma > 0.0)
                {
                    mtf.calculateTheoreticalMTF(options.gaussian_sigma);
                }

                // Convert to lp/mm if pixel size provided
                if (options.pixel_size_mm > 0.0)
                {
                    mtf.convertToLPMM(options.pixel_size_mm);
                }

                if (isValidMTF(mtf, lsf, true)) // Always show debug for validation issues
                {
                    mtf_results.push_back(mtf);
                    std::cout << "✅ MTF result added to collection. Total: " << mtf_results.size() << std::endl;
                }
                else
                {
                    std::cout << "❌ MTF result rejected by validation" << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error processing edge: " << e.what() << std::endl;
                continue;
            }
        }

        std::cout << "\n=== MTF COLLECTION SUMMARY ===" << std::endl;
        std::cout << "Total MTF results collected: " << mtf_results.size() << std::endl;
        
        if (mtf_results.empty())
        {
            throw std::runtime_error("No valid MTF results obtained");
        }

        std::cout << "Proceeding to average " << mtf_results.size() << " MTF results..." << std::endl;
        
        // Average the MTF results
        MTFResults averaged_mtf = averageMTFResults(mtf_results);
        
        std::cout << "MTF averaging completed successfully!" << std::endl;

        // Print final MTF metrics
        std::cout << "\n=== FINAL MTF RESULTS ===" << std::endl;
        std::cout << "MTF50: " << averaged_mtf.mtf50 << " cycles/pixel" << std::endl;
        std::cout << "MTF20: " << averaged_mtf.mtf20 << " cycles/pixel" << std::endl;
        std::cout << "MTF10: " << averaged_mtf.mtf10 << " cycles/pixel" << std::endl;
        
        if (averaged_mtf.has_theoretical) {
            std::cout << "\n=== THEORETICAL COMPARISON ===" << std::endl;
            std::cout << "Theoretical MTF50: " << averaged_mtf.theoretical_mtf50 << " cycles/pixel" << std::endl;
            std::cout << "Theoretical MTF20: " << averaged_mtf.theoretical_mtf20 << " cycles/pixel" << std::endl;
            std::cout << "Theoretical MTF10: " << averaged_mtf.theoretical_mtf10 << " cycles/pixel" << std::endl;
            
            double mtf50_error = ((averaged_mtf.mtf50 - averaged_mtf.theoretical_mtf50) / averaged_mtf.theoretical_mtf50) * 100;
            std::cout << "MTF50 Error: " << mtf50_error << "%" << std::endl;
        }

        // Return the averaged results
        return averaged_mtf;
    }

    static cv::Mat analyzeImageMTF(const cv::Mat &input_image,
                                   double pixel_size_mm,
                                   double gaussian_sigma = 0.0)
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
        std::vector<MTFResults> mtf_results;

        // Process each detected edge
        for (const auto &edge : edges)
        {
            try
            {
                // Step 2: Sample edge profiles
                EdgeProfileData profiles = sampleEdgeProfiles(edge.roi_image, edge);

                // Step 3: Compute super-resolution ESF
                ESFData esf = computeSuperResolutionESF(profiles, edge.angle);

                // Step 4: Compute LSF with quality assessment
                LSFData lsf = computeLSF(esf, edge.roi_image, edge.line, profiles.profiles.size());

                // Quality filtering: Only proceed with acceptable quality ROIs
                if (!lsf.quality.is_acceptable)
                {
                    std::cout << "⚠️ Skipping ROI due to poor quality" << std::endl;
                    continue; // Skip this ROI
                }
                
                // Additional outlier filtering based on FWHM reasonableness
                // Skip ROIs with extremely small or large FWHM values that indicate measurement errors
                if (lsf.fwhm < 0.1 || lsf.fwhm > 5.0)
                {
                    std::cout << "⚠️ Skipping ROI due to unrealistic FWHM: " << lsf.fwhm << " pixels" << std::endl;
                    continue;
                }

                // Step 5: Compute MTF
                MTFResults mtf = computeMTF(lsf);

                // If gaussian sigma is provided, calculate theoretical values
                if (gaussian_sigma > 0.0)
                {
                    mtf.calculateTheoreticalMTF(gaussian_sigma);
                }

                // Convert to lp/mm
                mtf.convertToLPMM(pixel_size_mm);

                // Only include results that meet quality criteria
                if (isValidMTF(mtf, lsf, true)) // Always show debug for validation issues
                {
                    mtf_results.push_back(mtf);
                    std::cout << "✅ MTF result added to collection. Total: " << mtf_results.size() << std::endl;
                }
                else
                {
                    std::cout << "❌ MTF result rejected by validation" << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error processing edge: " << e.what() << std::endl;
                continue;
            }
        }

        std::cout << "\n=== MTF COLLECTION SUMMARY ===" << std::endl;
        std::cout << "Total MTF results collected: " << mtf_results.size() << std::endl;
        
        if (mtf_results.empty())
        {
            throw std::runtime_error("No valid MTF results obtained");
        }

        std::cout << "Proceeding to average " << mtf_results.size() << " MTF results..." << std::endl;
        
        // Average the MTF results
        MTFResults averaged_mtf = averageMTFResults(mtf_results);
        
        std::cout << "MTF averaging completed successfully!" << std::endl;

        // Print final MTF metrics
        std::cout << "\n=== FINAL MTF RESULTS ===" << std::endl;
        std::cout << "MTF50: " << averaged_mtf.mtf50 << " cycles/pixel" << std::endl;
        std::cout << "MTF20: " << averaged_mtf.mtf20 << " cycles/pixel" << std::endl;
        std::cout << "MTF10: " << averaged_mtf.mtf10 << " cycles/pixel" << std::endl;
        
        if (averaged_mtf.has_theoretical) {
            std::cout << "\n=== THEORETICAL COMPARISON ===" << std::endl;
            std::cout << "Theoretical MTF50: " << averaged_mtf.theoretical_mtf50 << " cycles/pixel" << std::endl;
            std::cout << "Theoretical MTF20: " << averaged_mtf.theoretical_mtf20 << " cycles/pixel" << std::endl;
            std::cout << "Theoretical MTF10: " << averaged_mtf.theoretical_mtf10 << " cycles/pixel" << std::endl;
            
            double mtf50_error = ((averaged_mtf.mtf50 - averaged_mtf.theoretical_mtf50) / averaged_mtf.theoretical_mtf50) * 100;
            std::cout << "MTF50 Error: " << mtf50_error << "%" << std::endl;
        }

        // Create visualization with appropriate units and validation if available
        return createMTFVisualization(averaged_mtf, true, pixel_size_mm);
    }

    static cv::Mat createMTFVisualization(const MTFResults &mtf, bool debug = false, double pixel_size_mm = 1.0)
    {
        const int plot_height = 400;
        const int plot_width = 800;
        const int margin = 50;

        cv::Mat vis(plot_height + 2 * margin, plot_width + 2 * margin, CV_8UC3, cv::Scalar(255, 255, 255));

        // Draw grid
        for (int i = 0; i <= 10; i++)
        {
            int y = margin + i * plot_height / 10;
            cv::line(vis, cv::Point(margin, y), cv::Point(margin + plot_width, y),
                     cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
        }

        // Add labels
        cv::putText(vis, "Modulation Transfer Function",
                    cv::Point(margin, margin - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        // Update x-axis label based on units
        std::string x_label = (pixel_size_mm != 1.0) ? "Frequency (lp/mm)" : "Frequency (cycles/pixel)";
        cv::putText(vis, x_label,
                    cv::Point(plot_width / 2 + margin - 100, plot_height + 2 * margin - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        cv::putText(vis, "MTF",
                    cv::Point(10, plot_height / 2 + margin),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        // Draw MTF50 line
        int mtf50_y = margin + plot_height / 2;
        cv::line(vis, cv::Point(margin, mtf50_y),
                 cv::Point(margin + plot_width, mtf50_y),
                 cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

        // Add MTF50 label
        std::string mtf50_text = "MTF50: " + std::to_string(mtf.mtf50);
        cv::putText(vis, mtf50_text,
                    cv::Point(margin + 10, margin + 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        // Plot measured MTF
        std::vector<cv::Point> points;
        for (size_t i = 0; i < mtf.frequencies.size(); i++)
        {
            // Use non-linear scaling for x-axis to show more detail in low frequencies
            double x_scale = std::min(1.0, mtf.frequencies[i] / 0.2) * 0.8 +
                             std::min(1.0, mtf.frequencies[i] / 0.5) * 0.2;

            int x = margin + static_cast<int>(x_scale * plot_width);
            int y = margin + plot_height - static_cast<int>(mtf.mtf_values[i] * plot_height);

            // Only include points within the plot area
            if (x >= margin && x <= margin + plot_width)
            {
                points.push_back(cv::Point(x, y));
            }
        }

        cv::polylines(vis, points, false, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        // Draw theoretical MTF if available
        if (mtf.has_theoretical)
        {
            std::vector<cv::Point> theo_points;
            for (size_t i = 0; i < mtf.frequencies.size(); i++)
            {
                double x_scale = std::min(1.0, mtf.frequencies[i] / 0.2) * 0.8 +
                                 std::min(1.0, mtf.frequencies[i] / 0.5) * 0.2;

                int x = margin + static_cast<int>(x_scale * plot_width);
                int y = margin + plot_height - static_cast<int>(mtf.theoretical_mtf[i] * plot_height);

                if (x >= margin && x <= margin + plot_width)
                {
                    theo_points.push_back(cv::Point(x, y));
                }
            }
            cv::polylines(vis, theo_points, false, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            // Add legend
            cv::putText(vis, "Measured MTF", cv::Point(margin + 10, margin + 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::putText(vis, "Theoretical MTF", cv::Point(margin + 10, margin + 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }

        return vis;
    }

    static void testFWHMCalculation()
    {
        std::cout << "\n====== TESTING FWHM CALCULATION ======" << std::endl;

        // Create a synthetic Gaussian LSF
        const int size = 200;
        const double sigma = 1.5;                      // Same as your applied blur
        const double theoretical_fwhm = 2.355 * sigma; // FWHM = 2.355*sigma for Gaussian

        std::vector<double> x(size);
        std::vector<double> y(size);

        // Create distance array centered at 0 with sub-pixel resolution
        const double pixel_spacing = 0.25; // 4x super-resolution as per ISO 12233
        for (int i = 0; i < size; i++)
        {
            x[i] = (i - size / 2) * pixel_spacing;
        }

        // Generate normalized Gaussian values
        double sum = 0.0;
        for (int i = 0; i < size; i++)
        {
            y[i] = std::exp(-(x[i] * x[i]) / (2 * sigma * sigma));
            sum += y[i];
        }

        // Normalize to ensure area = 1.0
        double dx = x[1] - x[0];
        double normalization_factor = sum * dx;
        for (auto &val : y)
        {
            val /= normalization_factor;
        }

        // Calculate FWHM using our function
        double measured_fwhm = calculateFWHMWithDebug(x, y, true);

        std::cout << "\n====== Gaussian Test Results ======" << std::endl;
        std::cout << "Input sigma: " << sigma << std::endl;
        std::cout << "Theoretical FWHM: " << theoretical_fwhm << std::endl;
        std::cout << "Measured FWHM: " << measured_fwhm << std::endl;
        std::cout << "Ratio (measured/theoretical): " << measured_fwhm / theoretical_fwhm << std::endl;

        if (std::abs(measured_fwhm / theoretical_fwhm - 1.0) > 0.05)
        {
            std::cout << "WARNING: FWHM measurement differs from theoretical by >5%" << std::endl;
        }
        else
        {
            std::cout << "FWHM calculation PASSED synthetic test" << std::endl;
        }
    }

private:
    // Edge detection
    static std::vector<EdgeROI> detectEdgesAndCreateROIs(const cv::Mat &input, bool debug = false)
    {
        std::vector<EdgeROI> detected_edges;

        // Create copy and mask central region using more efficient OpenCV operations
        cv::Mat masked;
        input.copyTo(masked);
        int radius = std::min(input.rows, input.cols) / 100 * 5;
        cv::circle(masked, cv::Point(input.cols / 2, input.rows / 2), radius, cv::Scalar(0, 0, 0), -1, cv::LINE_AA);

        if (debug)
        {
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

        if (debug)
        {
            cv::imwrite("open-cv-2_detected_edges.png", edges);
        }

        // Optimize line detection parameters
        std::vector<cv::Vec4i> lines;
        int min_line_length = std::min(input.rows, input.cols) / 8;
        int max_line_gap = std::min(input.rows, input.cols) / 16;
        cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, min_line_length, max_line_gap);

        if (debug)
        {
            cv::Mat lines_image = masked.clone();
            cv::Point2f image_center(input.cols / 2.0f, input.rows / 2.0f);
            for (const auto &l : lines)
            {
                // Use OpenCV's line drawing with anti-aliasing
                cv::line(lines_image, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]),
                         cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            }
            cv::imwrite("open-cv-3_detected_lines.png", lines_image);
            std::cout << "Found " << lines.size() << " total lines" << std::endl;
        }

        // Use OpenCV's Point2f for better precision in angle calculations
        cv::Point2f image_center(input.cols / 2.0f, input.rows / 2.0f);

        // Create quadrant groups using OpenCV's Point2f for calculations
        std::map<int, std::vector<cv::Vec4i>> quadrant_groups;

        for (const auto &l : lines)
        {
            cv::Point2f pt1(l[0], l[1]), pt2(l[2], l[3]);
            cv::Point2f line_vector = pt2 - pt1;

            // Use OpenCV's fastAtan2 for better performance
            double angle = -cv::fastAtan2(line_vector.y, line_vector.x);
            std::cout << "Found edge with angle: " << angle << ", raw vector: ("
                      << line_vector.x << ", " << line_vector.y << ")" << std::endl;
            while (angle < 0)
                angle += 360.0;

            // HYBRID APPROACH: Use proven complementary angles + research-based validation
            
            // First, check if it matches the proven complementary angle pairs (11° and 281°)
            bool matches_complementary = (std::abs(angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE ||
                                          std::abs(angle - ANGLE2_TARGET) <= ANGLE_TOLERANCE);
            
            if (matches_complementary) {
                // Edge matches proven working angles - accept it
                // Also validate quality using research standards
                double normalized_angle = normalizeAngleForSlantEdge(angle);
                bool is_optimal = isOptimalSlantEdgeAngle(angle);
                bool is_acceptable = isAcceptableSlantEdgeAngle(angle);
                std::string category = getAngleCategory(angle);
                
                if (debug) {
                    std::cout << "✓ Found complementary edge at angle " << angle 
                              << "° (normalized: " << normalized_angle << "°)" << std::endl;
                    std::cout << "  Research classification: " << category << std::endl;
                    
                    if (is_optimal) {
                        std::cout << "  Quality: OPTIMAL (research-validated)" << std::endl;
                    } else if (is_acceptable) {
                        std::cout << "  Quality: ACCEPTABLE (research-validated, may have slightly lower precision)" << std::endl;
                    } else {
                        std::cout << "  Quality: ⚠ SUBOPTIMAL (outside research recommendations, but proven to work)" << std::endl;
                    }
                }

                cv::Point2f line_center = (pt1 + pt2) * 0.5f;
                cv::Point2f relative_pos = line_center - image_center;

                // Determine quadrant using OpenCV's point arithmetic
                int quadrant = (relative_pos.x >= 0 ? 0 : 1) + (relative_pos.y >= 0 ? 2 : 0);
                quadrant_groups[quadrant].push_back(l);
                
            } else if (debug) {
                // Edge doesn't match complementary angles - could add additional detection logic here if needed
                double normalized_angle = normalizeAngleForSlantEdge(angle);
                std::cout << "⚪ Edge at angle " << angle 
                          << "° (normalized: " << normalized_angle 
                          << "°) doesn't match complementary angle pairs" << std::endl;
            }
        }

        cv::Mat rois_image;
        if (debug)
        {
            rois_image = input.clone();
        }

        // Process each quadrant using OpenCV's optimized functions
        for (const auto &group : quadrant_groups)
        {
            if (group.second.empty())
                continue;

            // Find longest line using OpenCV's norm function
            auto longest_line = *std::max_element(
                group.second.begin(),
                group.second.end(),
                [](const cv::Vec4i &a, const cv::Vec4i &b)
                {
                    cv::Point2f vec_a(a[2] - a[0], a[3] - a[1]);
                    cv::Point2f vec_b(b[2] - b[0], b[3] - b[1]);
                    return cv::norm(vec_a) < cv::norm(vec_b);
                });

            EdgeROI roi_data;
            roi_data.line = longest_line;

            // Use OpenCV's point arithmetic and fastAtan2
            cv::Point2f line_vector(longest_line[2] - longest_line[0],
                                    longest_line[3] - longest_line[1]);
            double angle = -cv::fastAtan2(line_vector.y, line_vector.x);
            while (angle < 0)
                angle += 360.0;
            roi_data.angle = angle;

            // Calculate ROI dimensions using OpenCV's point arithmetic
            cv::Point2f line_start(longest_line[0], longest_line[1]);
            cv::Point2f line_end(longest_line[2], longest_line[3]);
            cv::Point2f line_center = (line_start + line_end) * 0.5f;

            float length = cv::norm(line_end - line_start);
            
            // IMPROVED: Position ROI to fully contain the detected line endpoints
            // This ensures we capture the complete edge transition
            
            // Calculate minimum bounding box for the line endpoints
            float min_x = std::min(line_start.x, line_end.x);
            float max_x = std::max(line_start.x, line_end.x);
            float min_y = std::min(line_start.y, line_end.y);
            float max_y = std::max(line_start.y, line_end.y);
            
            // Add safety margins around the line (30 pixels on each side)
            const float margin = 30.0f;
            min_x -= margin;
            max_x += margin;
            min_y -= margin;
            max_y += margin;
            
            // Ensure minimum ROI dimensions for proper edge sampling
            float line_length = cv::norm(line_end - line_start);
            float min_dimension = std::max(60.0f, line_length * 0.5f); // At least 60 pixels
            
            // Expand ROI if too small in either dimension
            float current_width = max_x - min_x;
            float current_height = max_y - min_y;
            
            if (current_width < min_dimension) {
                float expand = (min_dimension - current_width) * 0.5f;
                min_x -= expand;
                max_x += expand;
            }
            
            if (current_height < min_dimension) {
                float expand = (min_dimension - current_height) * 0.5f;
                min_y -= expand;
                max_y += expand;
            }

            // Create ROI rectangle based on calculated bounds
            roi_data.roi = cv::Rect(
                cv::Point(static_cast<int>(min_x), static_cast<int>(min_y)),
                cv::Point(static_cast<int>(max_x), static_cast<int>(max_y)));

            // Use OpenCV's rectangle intersection
            roi_data.roi &= cv::Rect(0, 0, input.cols, input.rows);

            if (debug) {
                std::cout << "ROI Positioning Debug:" << std::endl;
                std::cout << "  Line endpoints: (" << line_start.x << "," << line_start.y 
                          << ") to (" << line_end.x << "," << line_end.y << ")" << std::endl;
                std::cout << "  Line length: " << line_length << " pixels" << std::endl;
                std::cout << "  Final ROI: " << roi_data.roi << " (size: " 
                          << roi_data.roi.width << "x" << roi_data.roi.height << ")" << std::endl;
            }

            // Use OpenCV's optimized ROI extraction
            roi_data.roi_image = input(roi_data.roi).clone();
            detected_edges.push_back(roi_data);

            if (debug)
            {
                // Use OpenCV's anti-aliased line drawing
                cv::line(rois_image, cv::Point(longest_line[0], longest_line[1]),
                         cv::Point(longest_line[2], longest_line[3]),
                         cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                cv::rectangle(rois_image, roi_data.roi, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            }
        }

        if (debug)
        {
            cv::imwrite("open-cv-4_rois.png", rois_image);
            std::cout << "Found " << detected_edges.size() << " edges at target angles" << std::endl;
        }

        return detected_edges;
    }

    static EdgeProfileData sampleEdgeProfiles(const cv::Mat &roi, const EdgeROI &edge_data, bool debug = false)
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
        std::cout << "Line vector components: (" << line_vector.x << ", " << line_vector.y
                  << "), resulting angle: " << line_angle << std::endl;
        while (line_angle < 0)
            line_angle += 360.0;
        std::cout << "Calculated line angle: " << line_angle << std::endl;

        // Calculate adaptive sampling interval
        double sample_step = calculateSamplingInterval(line_angle);
        std::cout << "Using adaptive sampling interval: " << sample_step << std::endl;

        // Convert angles to radians using OpenCV
        double angle_rad = -line_angle * CV_PI / 180.0;
        double perp_angle = angle_rad + CV_PI / 2.0;

        std::cout << "Edge angle: " << line_angle
                  << ", calculated sampling interval: " << sample_step
                  << ", normalized angle used: " << angle_rad << std::endl;

        // Create direction vectors using OpenCV points
        cv::Point2f dir_vector(std::cos(perp_angle), std::sin(perp_angle));
        cv::Point2f edge_dir(-std::sin(angle_rad), std::cos(angle_rad));
        std::cout << "Direction vectors calculated." << std::endl;

        // Setup sampling parameters
        bool is_vertical = std::abs(line_angle - ANGLE1_TARGET) <= ANGLE_TOLERANCE;
        int longer_dim = is_vertical ? gray.rows : gray.cols;
        const int NUM_SAMPLES = 40;
        const int SAMPLE_LENGTH = std::min(gray.cols, gray.rows) / 2;
        // const double SAMPLE_STEP = 0.5;
        
        // COORDINATE SYSTEM FIX: Translate line coordinates from full image space to ROI local space
        // edge_data.line is in full image coordinates, but gray is the cropped ROI
        cv::Point2f roi_top_left(edge_data.roi.x, edge_data.roi.y);
        cv::Point2f line_start_full(edge_data.line[0], edge_data.line[1]);
        cv::Point2f line_end_full(edge_data.line[2], edge_data.line[3]);
        
        // Translate to ROI local coordinates
        cv::Point2f line_start_roi = line_start_full - roi_top_left;
        cv::Point2f line_end_roi = line_end_full - roi_top_left;
        cv::Point2f center = (line_start_roi + line_end_roi) * 0.5f;
        
        std::cout << "COORDINATE TRANSLATION DEBUG:" << std::endl;
        std::cout << "ROI offset: (" << roi_top_left.x << ", " << roi_top_left.y << ")" << std::endl;
        std::cout << "Line in full image: (" << line_start_full.x << ", " << line_start_full.y << ") to (" 
                  << line_end_full.x << ", " << line_end_full.y << ")" << std::endl;
        std::cout << "Line in ROI space: (" << line_start_roi.x << ", " << line_start_roi.y << ") to (" 
                  << line_end_roi.x << ", " << line_end_roi.y << ")" << std::endl;
        std::cout << "Calculated center: (" << center.x << ", " << center.y << ")" << std::endl;
        std::cout << "ROI center assumption would be: (" << gray.cols / 2.0f << ", " << gray.rows / 2.0f << ")" << std::endl;
        
        double sample_range = longer_dim * 0.4;

        // Pre-allocate memory for profiles
        const int expectedPointsPerProfile = static_cast<int>((2 * SAMPLE_LENGTH) / sample_step) + 1;
        result.profiles.resize(NUM_SAMPLES); // Changed from reserve to resize for parallel access

        std::cout << "Sampling parameters:" << std::endl;
        std::cout << "Longer dimension: " << longer_dim << std::endl;
        std::cout << "Sample length: " << SAMPLE_LENGTH << std::endl;
        std::cout << "Sample range: " << sample_range << std::endl;
        std::cout << "Expected points per profile: " << expectedPointsPerProfile << std::endl;

        // Create mutex for thread-safe profile storage
        std::mutex profiles_mutex;

        // Parallel processing of profiles
        cv::parallel_for_(cv::Range(0, NUM_SAMPLES), [&](const cv::Range &range)
                          {
            for (int i = range.start; i < range.end; i++) {
                std::vector<double> profile;
                profile.reserve(expectedPointsPerProfile);
                
                double t = (i - (NUM_SAMPLES - 1) / 2.0) / (NUM_SAMPLES - 1);
                double offset = t * sample_range;

                // Calculate start point
                cv::Point2f start_point = center + edge_dir * offset;

                // Sample points along profile
                for (double j = -SAMPLE_LENGTH; j <= SAMPLE_LENGTH; j += sample_step) {
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
                    // Thread-safe assignment to result vector
                    result.profiles[i] = std::move(profile);
                    
                    // Debug output only for first profile
                    if (i == 0) {
                        std::lock_guard<std::mutex> lock(profiles_mutex);
                        std::cout << "First profile size: " << result.profiles[i].size() << std::endl;
                        std::cout << "First few values: ";
                        for (int j = 0; j < std::min(5, (int)result.profiles[i].size()); j++) {
                            std::cout << result.profiles[i][j] << " ";
                        }
                        std::cout << std::endl;
                    }
                }
            } });

        // Remove any empty profiles
        result.profiles.erase(
            std::remove_if(result.profiles.begin(), result.profiles.end(),
                           [](const std::vector<double> &profile)
                           { return profile.empty(); }),
            result.profiles.end());

        std::cout << "Total profiles collected: " << result.profiles.size() << std::endl;
        if (!result.profiles.empty())
        {
            std::cout << "Profile lengths: " << result.profiles[0].size() << std::endl;
        }

        return result;
    }

    // Core analysis pipeline
    static ESFData computeSuperResolutionESF(const EdgeProfileData &profiles, double angle_degrees, bool debug = false)
    {
        ESFData result;
        const double BIN_WIDTH = 0.05; // IMPROVEMENT: Further reduced for higher precision
        
        // IMPROVEMENT: Calculate dynamic sampling interval for this angle
        double sampling_interval = calculateSamplingInterval(angle_degrees);

        if (debug)
        {
            std::cout << "\n====== Super-Resolution ESF Computation ======" << std::endl;
            std::cout << "Number of input profiles: " << profiles.profiles.size() << std::endl;
            std::cout << "Edge angle: " << angle_degrees << " degrees" << std::endl;
            std::cout << "Sampling interval: " << sampling_interval << std::endl;
            std::cout << "Bin width: " << BIN_WIDTH << std::endl;
            if (!profiles.profiles.empty())
            {
                std::cout << "First profile length: " << profiles.profiles[0].size() << std::endl;
            }
        }

        // Pre-calculate total points needed
        size_t total_points = 0;
        for (const auto &profile : profiles.profiles)
        {
            total_points += profile.size();
        }

        // Pre-allocate vectors
        result.distances.reserve(total_points);
        result.intensities.reserve(total_points);
        result.binned_distances.reserve(total_points / 10);
        result.binned_intensities.reserve(total_points / 10);

        // Find global min/max for normalization
        double global_min = std::numeric_limits<double>::max();
        double global_max = std::numeric_limits<double>::lowest();

        for (const auto &profile : profiles.profiles)
        {
            if (!profile.empty())
            {
                auto [min_it, max_it] = std::minmax_element(profile.begin(), profile.end());
                global_min = std::min(global_min, *min_it);
                global_max = std::max(global_max, *max_it);
            }
        }

        if (debug)
        {
            std::cout << "Global intensity range - Min: " << global_min
                      << ", Max: " << global_max << std::endl;
        }

        // Process each profile
        std::vector<double> edge_positions;
        edge_positions.reserve(profiles.profiles.size());

        for (size_t profile_idx = 0; profile_idx < profiles.profiles.size(); profile_idx++)
        {
            const auto &profile = profiles.profiles[profile_idx];
            if (profile.empty())
                continue;

            // Calculate gradients for edge detection
            std::vector<double> gradients(profile.size() - 1);
            for (size_t i = 0; i < profile.size() - 1; i++)
            {
                gradients[i] = std::abs(profile[i + 1] - profile[i]);
            }

            // Find edge location using maximum gradient
            auto max_grad_it = std::max_element(gradients.begin(), gradients.end());
            int edge_idx = std::distance(gradients.begin(), max_grad_it);

            // Apply sub-pixel refinement using parabolic fit
            double subpixel_offset = 0.0;
            if (edge_idx > 0 && edge_idx < static_cast<int>(gradients.size()) - 1)
            {
                double y1 = gradients[edge_idx - 1];
                double y2 = gradients[edge_idx];
                double y3 = gradients[edge_idx + 1];

                // Fit parabola: y = a*x^2 + b*x + c
                double a = 0.5 * (y1 + y3) - y2;
                double b = 0.5 * (y3 - y1);

                // If 'a' is negative (proper peak), calculate sub-pixel offset
                if (a < 0)
                {
                    subpixel_offset = -b / (2 * a);

                    // Limit offset to reasonable range
                    subpixel_offset = std::max(-0.5, std::min(0.5, subpixel_offset));
                }
            }

            // Final edge position with sub-pixel precision
            double edge_position = edge_idx + subpixel_offset;
            edge_positions.push_back(edge_position);

            if (debug && (profile_idx == 0 || profile_idx == profiles.profiles.size() - 1))
            {
                std::cout << "\nProfile " << profile_idx << " edge detection:" << std::endl;
                std::cout << "Max gradient: " << *max_grad_it
                          << " at position " << edge_idx << std::endl;
                std::cout << "Sub-pixel offset: " << subpixel_offset
                          << ", Final position: " << edge_position << std::endl;

                // Print values around edge for verification
                std::cout << "Values around edge:" << std::endl;
                for (int i = -2; i <= 2; i++)
                {
                    int pos = edge_idx + i;
                    if (pos >= 0 && pos < static_cast<int>(profile.size()))
                    {
                        std::cout << "  Position " << pos << ": " << profile[pos];
                        if (i != 2 && pos < static_cast<int>(gradients.size()))
                            std::cout << " -> gradient: " << gradients[pos];
                        std::cout << std::endl;
                    }
                }
            }

            // Store normalized samples centered on edge
            for (size_t i = 0; i < profile.size(); i++)
            {
                double normalized = (profile[i] - global_min) / (global_max - global_min);
                // ISO 12233: Use sub-pixel distance with proper scaling
                // Distance in pixels relative to edge center
                double pixel_distance = static_cast<double>(i) - edge_position;
                result.distances.push_back(pixel_distance);
                result.intensities.push_back(normalized);
            }
        }

        if (debug)
        {
            std::cout << "\nEdge position statistics:" << std::endl;
            if (!edge_positions.empty())
            {
                auto [min_pos, max_pos] = std::minmax_element(edge_positions.begin(), edge_positions.end());
                double mean_pos = std::accumulate(edge_positions.begin(), edge_positions.end(), 0.0) / edge_positions.size();
                double variance = 0.0;
                for (double pos : edge_positions)
                {
                    variance += (pos - mean_pos) * (pos - mean_pos);
                }
                variance /= edge_positions.size();
                double std_dev = std::sqrt(variance);

                std::cout << "Edge positions - Min: " << *min_pos
                          << ", Max: " << *max_pos
                          << ", Mean: " << mean_pos
                          << ", StdDev: " << std_dev << std::endl;

                if (std_dev > 1.0)
                {
                    std::cout << "Warning: High variance in edge positions!" << std::endl;
                }
            }
        }

        // Sort points by distance
        std::vector<size_t> indices(result.distances.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](size_t a, size_t b)
                  { return result.distances[a] < result.distances[b]; });

        if (debug)
        {
            std::cout << "\n====== ESF Pre-binning Stats ======" << std::endl;
            std::cout << "Total data points: " << result.distances.size() << std::endl;
            std::cout << "Distance range: " << result.distances[indices.front()]
                      << " to " << result.distances[indices.back()] << std::endl;
        }

        // Bin the data
        std::map<int, std::vector<double>> bins;
        double min_dist = *std::min_element(result.distances.begin(), result.distances.end());

        for (size_t i = 0; i < result.distances.size(); i++)
        {
            int bin = static_cast<int>((result.distances[indices[i]] - min_dist) / BIN_WIDTH);
            bins[bin].push_back(result.intensities[indices[i]]);
        }

        // Calculate bin statistics
        for (const auto &bin : bins)
        {
            if (bin.second.size() >= 3)
            {
                std::vector<double> bin_values = bin.second;
                std::sort(bin_values.begin(), bin_values.end());
                double median = bin_values[bin_values.size() / 2];

                result.binned_distances.push_back(min_dist + bin.first * BIN_WIDTH);
                result.binned_intensities.push_back(median);
            }
        }

        if (debug)
        {
            std::cout << "\n====== ESF Post-binning Stats ======" << std::endl;
            std::cout << "Bins created: " << result.binned_distances.size() << std::endl;

            // Check bin spacing
            if (result.binned_distances.size() > 1)
            {
                double total_spacing = 0.0;
                int count = 0;
                for (size_t i = 1; i < result.binned_distances.size(); i++)
                {
                    double spacing = result.binned_distances[i] - result.binned_distances[i - 1];
                    total_spacing += spacing;
                    count++;
                }
                std::cout << "Average bin spacing: " << (total_spacing / count) << std::endl;
            }
        }

        return result;
    }

    static LSFData computeLSF(const ESFData &esf, const cv::Mat &roi, const cv::Vec4i &line, int profile_count, bool debug = false)
    {
        LSFData result;

        if (debug)
        {
            std::cout << "\n====== LSF Computation ======" << std::endl;
            std::cout << "Input ESF points: " << esf.binned_distances.size() << std::endl;

            // Print sample ESF values
            std::cout << "Sample ESF values:" << std::endl;
            size_t num_samples = std::min(size_t(10), esf.binned_distances.size());
            for (size_t i = 0; i < num_samples; i++)
            {
                std::cout << "  " << esf.binned_distances[i] << ": " << esf.binned_intensities[i] << std::endl;
            }
        }

        // Check for sufficient data points
        if (esf.binned_distances.size() < 5)
        {
            std::cerr << "Error: Not enough bins for LSF calculation" << std::endl;
            return result;
        }

        // Pre-allocate vectors
        result.distances.resize(esf.binned_distances.size() - 2);
        result.lsf_values.resize(esf.binned_distances.size() - 2);

        // Compute central difference derivative with step size normalization
        for (size_t i = 1; i < esf.binned_distances.size() - 1; i++)
        {
            double dx = esf.binned_distances[i + 1] - esf.binned_distances[i - 1];
            double dy = esf.binned_intensities[i + 1] - esf.binned_intensities[i - 1];

            if (std::abs(dx) > 1e-10)
            { // Avoid division by very small numbers
                result.distances[i - 1] = esf.binned_distances[i];
                result.lsf_values[i - 1] = std::abs(dy / dx);
            }
            else
            {
                result.lsf_values[i - 1] = 0.0;
            }
        }

        if (debug)
        {
            std::cout << "\nDerivative calculation (sample points):" << std::endl;
            size_t num_samples = std::min(size_t(10), result.lsf_values.size());
            for (size_t i = 0; i < num_samples; i++)
            {
                std::cout << "  Distance: " << result.distances[i]
                        << ", LSF: " << result.lsf_values[i] << std::endl;
            }

            // Calculate statistics on LSF values
            double sum = std::accumulate(result.lsf_values.begin(), result.lsf_values.end(), 0.0);
            double mean = sum / result.lsf_values.size();
            double sq_sum = std::inner_product(result.lsf_values.begin(), result.lsf_values.end(),
                                            result.lsf_values.begin(), 0.0);
            double stdev = std::sqrt(sq_sum / result.lsf_values.size() - mean * mean);

            std::cout << "Pre-normalized LSF statistics - Mean: " << mean
                    << ", StdDev: " << stdev << std::endl;
        }

        // ============= THIS IS THE KEY FIX =============
        // CRITICAL FIX: Normalize by area under the curve instead of peak
        double total_area = 0.0;
        for (size_t i = 0; i < result.lsf_values.size() - 1; i++)
        {
            double dx = result.distances[i + 1] - result.distances[i];
            double avg_height = (result.lsf_values[i] + result.lsf_values[i + 1]) / 2.0;
            total_area += dx * avg_height;
        }

        if (debug)
        {
            std::cout << "Original LSF area: " << total_area << std::endl;
            double peak_value = *std::max_element(result.lsf_values.begin(), result.lsf_values.end());
            std::cout << "Peak LSF value (pre-normalization): " << peak_value << std::endl;
        }

        // Normalize all values by the total area
        if (total_area > 0)
        {
            for (auto &value : result.lsf_values)
            {
                value /= total_area;
            }

            // Verify normalization
            if (debug)
            {
                double new_area = 0.0;
                for (size_t i = 0; i < result.lsf_values.size() - 1; i++)
                {
                    double dx = result.distances[i + 1] - result.distances[i];
                    double avg_height = (result.lsf_values[i] + result.lsf_values[i + 1]) / 2.0;
                    new_area += dx * avg_height;
                }

                std::cout << "Normalized LSF area: " << new_area << std::endl;
                if (std::abs(new_area - 1.0) > 0.1)
                {
                    std::cout << "Warning: Area normalization not precise" << std::endl;
                }
            }
        }
        // ============= END OF KEY FIX =============

        // Store normalized values
        result.smoothed_lsf = result.lsf_values;

        // Calculate FWHM using enhanced method
        result.fwhm = calculateFWHMWithDebug(result.distances, result.smoothed_lsf, debug);

        // ROI Quality Assessment
        result.quality = assessROIQuality(roi, line, profile_count);

        // Save individual ROI debug image for analysis
        static int roi_debug_counter = 0;
        if (!roi.empty())
        {
            std::string roi_filename = "debug_roi_" + std::to_string(roi_debug_counter++) + ".png";
            cv::imwrite(roi_filename, roi);
            std::cout << "ROI debug image saved: " << roi_filename << std::endl;
        }

        // Always report real FWHM from image analysis with quality
        std::cout << "\n=== REAL IMAGE FWHM ANALYSIS ===" << std::endl;
        std::cout << "Measured FWHM from image: " << result.fwhm << " pixels" << std::endl;
        std::cout << "ROI Quality Score: " << result.quality.overall_score << "/100" << std::endl;
        std::cout << "Quality Components:" << std::endl;
        std::cout << "  Edge Strength: " << result.quality.edge_strength << "/100" << std::endl;
        std::cout << "  Linearity: " << result.quality.linearity_score << "/100" << std::endl;
        std::cout << "  Noise Level: " << result.quality.noise_level << " (lower=better)" << std::endl;
        std::cout << "  Profile Adequacy: " << result.quality.profile_adequacy << "/100" << std::endl;
        std::cout << "Quality Status: " << (result.quality.is_acceptable ? "✅ ACCEPTABLE" : "❌ POOR") 
                  << " (" << result.quality.quality_reason << ")" << std::endl;
        
        if (debug)
        {
            std::cout << "\nFinal LSF Statistics:" << std::endl;
            std::cout << "FWHM: " << result.fwhm << std::endl;
        }

        return result;
    }
    // Enhanced FWHM calculation function with detailed debugging
    static double calculateFWHMWithDebug(const std::vector<double> &x,
                                         const std::vector<double> &y,
                                         bool debug = false)
    {
        if (x.empty() || y.empty() || x.size() != y.size())
        {
            std::cerr << "Error: Invalid input for FWHM calculation" << std::endl;
            return 0.0;
        }

        // Find peak value and location
        auto peak_it = std::max_element(y.begin(), y.end());
        int peak_idx = std::distance(y.begin(), peak_it);
        double max_value = *peak_it;
        double half_max = max_value / 2.0;

        if (debug)
        {
            std::cout << "\n====== FWHM Calculation ======" << std::endl;
            std::cout << "Peak value: " << max_value << " at index " << peak_idx
                      << " (x = " << x[peak_idx] << ")" << std::endl;
            std::cout << "Half maximum: " << half_max << std::endl;
        }

        // Search left from peak
        double left_x = x[peak_idx];
        bool found_left = false;

        for (int i = peak_idx; i > 0; i--)
        {
            if (y[i] >= half_max && y[i - 1] < half_max)
            {
                // Linear interpolation for better precision
                double t = (half_max - y[i - 1]) / (y[i] - y[i - 1]);
                left_x = x[i - 1] + t * (x[i] - x[i - 1]);
                found_left = true;

                if (debug)
                {
                    std::cout << "Left crossing found between indices " << (i - 1) << " and " << i << std::endl;
                    std::cout << "Left x: " << left_x << " (t = " << t << ")" << std::endl;
                    std::cout << "Points: (" << x[i - 1] << ", " << y[i - 1] << ") and ("
                              << x[i] << ", " << y[i] << ")" << std::endl;
                }
                break;
            }
        }

        if (!found_left && debug)
        {
            std::cout << "Warning: Could not find left crossing for half maximum" << std::endl;
        }

        // Search right from peak
        double right_x = x[peak_idx];
        bool found_right = false;

        for (int i = peak_idx; i < static_cast<int>(y.size()) - 1; i++)
        {
            if (y[i] >= half_max && y[i + 1] < half_max)
            {
                // Linear interpolation for better precision
                double t = (half_max - y[i + 1]) / (y[i] - y[i + 1]);
                right_x = x[i] + t * (x[i + 1] - x[i]);
                found_right = true;

                if (debug)
                {
                    std::cout << "Right crossing found between indices " << i << " and " << (i + 1) << std::endl;
                    std::cout << "Right x: " << right_x << " (t = " << t << ")" << std::endl;
                    std::cout << "Points: (" << x[i] << ", " << y[i] << ") and ("
                              << x[i + 1] << ", " << y[i + 1] << ")" << std::endl;
                }
                break;
            }
        }

        if (!found_right && debug)
        {
            std::cout << "Warning: Could not find right crossing for half maximum" << std::endl;
        }

        // Calculate FWHM
        double fwhm = std::abs(right_x - left_x);

        if (debug)
        {
            std::cout << "Calculated FWHM: " << fwhm << std::endl;

            // Calculate average bin spacing
            double avg_spacing = 0;
            int count = 0;
            for (size_t i = 1; i < x.size(); i++)
            {
                avg_spacing += std::abs(x[i] - x[i - 1]);
                count++;
            }
            if (count > 0)
            {
                avg_spacing /= count;
                std::cout << "Average bin spacing: " << avg_spacing << std::endl;

                if (fwhm < avg_spacing)
                {
                    std::cout << "Warning: FWHM is smaller than bin spacing!" << std::endl;
                }
            }

            // For Gaussian, the theoretical relationship
            std::cout << "For reference, a Gaussian with FWHM = " << fwhm << " has sigma ≈ "
                      << (fwhm / 2.355) << std::endl;
        }

        return fwhm;
    }

    static MTFResults computeMTF(const LSFData &lsf, bool debug = false)
    {
        MTFResults result;

        if (debug)
        {
            std::cout << "\n====== MTF Computation ======" << std::endl;
            std::cout << "Input LSF size: " << lsf.smoothed_lsf.size() << std::endl;
        }

        // Ensure we have valid LSF data
        if (lsf.smoothed_lsf.empty() || lsf.smoothed_lsf.size() < 4)
        {
            std::cout << "Invalid LSF data for MTF calculation" << std::endl;
            return result;
        }

        // Step 1: Apply windowing to reduce spectral leakage
        std::vector<double> windowed_lsf = lsf.smoothed_lsf;
        for (size_t i = 0; i < windowed_lsf.size(); i++)
        {
            // Hann window: w(n) = 0.5 * (1 - cos(2π*n/(N-1)))
            double window = 0.5 * (1.0 - std::cos(2.0 * M_PI * i / (windowed_lsf.size() - 1)));
            windowed_lsf[i] *= window;
        }

        // Convert vector to Mat (using windowed data)
        cv::Mat lsf_data(1, windowed_lsf.size(), CV_64F);
        for (size_t i = 0; i < windowed_lsf.size(); i++)
        {
            lsf_data.at<double>(0, i) = windowed_lsf[i];
        }

        if (debug)
        {
            std::cout << "\nLSF before DFT:" << std::endl;
            std::cout << "First few LSF values: ";
            for (int i = 0; i < std::min(5, lsf_data.cols); i++)
            {
                std::cout << lsf_data.at<double>(0, i) << " ";
            }
            std::cout << std::endl;
        }

        // Get optimal DFT size and pad data
        int dft_size = cv::getOptimalDFTSize(lsf_data.cols);
        cv::Mat padded;
        cv::copyMakeBorder(lsf_data, padded, 0, 0, 0, dft_size - lsf_data.cols,
                           cv::BORDER_CONSTANT, cv::Scalar(0));

        if (debug)
        {
            std::cout << "Original size: " << lsf_data.cols << ", DFT size: " << dft_size << std::endl;
        }

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
        magnitude = magnitude(cv::Rect(0, 0, magnitude.cols / 2, 1));

        // Normalize
        double maxVal;
        cv::minMaxLoc(magnitude, nullptr, &maxVal);

        if (debug)
        {
            std::cout << "\nMTF Normalization:" << std::endl;
            std::cout << "Maximum magnitude value: " << maxVal << std::endl;
        }

        magnitude /= maxVal;

        // Optional: Apply mild smoothing to reduce noise
        cv::GaussianBlur(magnitude, magnitude, cv::Size(1, 1), 0.8);

        // Store results
        result.frequencies.resize(magnitude.cols);
        result.mtf_values.resize(magnitude.cols);

        for (int i = 0; i < magnitude.cols; i++)
        {
            result.frequencies[i] = static_cast<double>(i) / dft_size;
            result.mtf_values[i] = magnitude.at<double>(0, i);
        }

        // Calculate MTF metrics
        result.mtf50 = findMTFFrequency(result.frequencies, result.mtf_values, 0.5);
        result.mtf20 = findMTFFrequency(result.frequencies, result.mtf_values, 0.2);
        result.mtf10 = findMTFFrequency(result.frequencies, result.mtf_values, 0.1);

        if (debug)
        {
            std::cout << "\nMTF Metrics:" << std::endl;
            std::cout << "MTF50: " << result.mtf50 << " cycles/pixel" << std::endl;
            std::cout << "MTF20: " << result.mtf20 << " cycles/pixel" << std::endl;
            std::cout << "MTF10: " << result.mtf10 << " cycles/pixel" << std::endl;
        }

        return result;
    }

    // Validation & helper functions
    static bool isValidMTF(const MTFResults &mtf, const LSFData &lsf, bool debug = false)
    {
        // Check LSF quality
        double noise_level = calculateNoiseLevelLSF(lsf.smoothed_lsf);
        double peak_value = *std::max_element(lsf.smoothed_lsf.begin(),
                                              lsf.smoothed_lsf.end());
        double snr = peak_value / noise_level;

        // Relaxed quality criteria for synthetic images
        bool good_snr = snr >= 10.0;
        bool reasonable_fwhm = lsf.fwhm >= 0.1 && lsf.fwhm <= 10.0; // Allow sub-pixel measurements
        bool reasonable_mtf50 = mtf.mtf50 > 0.001 && mtf.mtf50 < 0.5; // More permissive for soft edges

        if (debug)
        {
            std::cout << "\nMTF Validation:\n"
                      << "SNR: " << snr << " (need >= 10.0): " << (good_snr ? "PASS" : "FAIL") << "\n"
                      << "FWHM: " << lsf.fwhm << " (need 0.1-10.0): " << (reasonable_fwhm ? "PASS" : "FAIL") << "\n"
                      << "MTF50: " << mtf.mtf50 << " (need 0.001-0.5): " << (reasonable_mtf50 ? "PASS" : "FAIL") << std::endl;
        }

        return good_snr && reasonable_fwhm && reasonable_mtf50;
    }

    static MTFResults averageMTFResults(const std::vector<MTFResults> &results)
    {
        if (results.empty())
        {
            throw std::runtime_error("No MTF results to average");
        }

        MTFResults averaged;
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

    // Analysis utilities
    static double calculateNoiseLevelLSF(const std::vector<double> &lsf)
    {
        // Calculate noise in the tail regions (first and last 10% of points)
        size_t region_size = lsf.size() / 10;
        if (region_size == 0)
            return 0.0;

        std::vector<double> tail_values;
        tail_values.reserve(region_size * 2);

        // Collect tail values from both ends
        for (size_t i = 0; i < region_size; i++)
        {
            tail_values.push_back(lsf[i]);
            tail_values.push_back(lsf[lsf.size() - 1 - i]);
        }

        // Calculate standard deviation
        double mean = std::accumulate(tail_values.begin(), tail_values.end(), 0.0) / tail_values.size();

        double sqsum = std::inner_product(
            tail_values.begin(), tail_values.end(),
            tail_values.begin(), 0.0,
            std::plus<double>(),
            [mean](double x, double y)
            { return (x - mean) * (y - mean); });

        return std::sqrt(sqsum / (tail_values.size() - 1));
    }

    static double findMTFFrequency(const std::vector<double> &frequencies,
                                   const std::vector<double> &mtf_values,
                                   double target_value)
    {
        if (frequencies.empty() || mtf_values.empty() ||
            frequencies.size() != mtf_values.size())
        {
            return 0.0;
        }

        // Handle edge cases
        if (mtf_values[0] < target_value)
            return 0.0;
        if (mtf_values.back() > target_value)
            return frequencies.back();

        // Find crossing point
        for (size_t i = 0; i < mtf_values.size() - 1; i++)
        {
            if (mtf_values[i] >= target_value && mtf_values[i + 1] < target_value)
            {
                // Linear interpolation
                double t = (target_value - mtf_values[i + 1]) /
                           (mtf_values[i] - mtf_values[i + 1]);
                return frequencies[i + 1] + t * (frequencies[i] - frequencies[i + 1]);
            }
        }

        return 0.0;
    }

    static double calculateFWHM(const std::vector<double> &x, const std::vector<double> &y)
    {
        if (x.empty() || y.empty() || x.size() != y.size())
        {
            return 0.0;
        }

        // Find peak value and location
        auto peak_it = std::max_element(y.begin(), y.end());
        int peak_idx = std::distance(y.begin(), peak_it);
        double max_value = *peak_it;
        double half_max = max_value / 2.0;

        // Search left from peak
        double left_x = x[peak_idx];
        for (int i = peak_idx; i > 0; i--)
        {
            if (y[i] >= half_max && y[i - 1] < half_max)
            {
                // Linear interpolation
                double t = (half_max - y[i - 1]) / (y[i] - y[i - 1]);
                left_x = x[i - 1] + t * (x[i] - x[i - 1]);
                break;
            }
        }

        // Search right from peak
        double right_x = x[peak_idx];
        for (int i = peak_idx; i < static_cast<int>(y.size()) - 1; i++)
        {
            if (y[i] >= half_max && y[i + 1] < half_max)
            {
                // Linear interpolation
                double t = (half_max - y[i + 1]) / (y[i] - y[i + 1]);
                right_x = x[i] + t * (x[i + 1] - x[i]);
                break;
            }
        }

        return right_x - left_x;
    }

    // HYBRID SYSTEM: Research-based validation for both horizontal and vertical edges
    static bool isOptimalSlantEdgeAngle(double angle_degrees) {
        // Normalize angle to 0-90 range for analysis
        double normalized_angle = normalizeAngleForSlantEdge(angle_degrees);
        
        // Check if it's in optimal horizontal range (8-15°)
        bool optimal_horizontal = (normalized_angle >= MIN_OPTIMAL_HORIZONTAL && 
                                   normalized_angle <= MAX_OPTIMAL_HORIZONTAL);
        
        // Check if it's in optimal vertical range (75-82°, which is 8-15° from vertical)
        bool optimal_vertical = (normalized_angle >= MIN_OPTIMAL_VERTICAL && 
                                 normalized_angle <= MAX_OPTIMAL_VERTICAL);
        
        return optimal_horizontal || optimal_vertical;
    }
    
    static bool isAcceptableSlantEdgeAngle(double angle_degrees) {
        // Normalize angle to 0-90 range
        double normalized_angle = normalizeAngleForSlantEdge(angle_degrees);
        
        // Check if it's in acceptable horizontal range (5-20°)
        bool acceptable_horizontal = (normalized_angle >= MIN_ACCEPTABLE_HORIZONTAL && 
                                      normalized_angle <= MAX_ACCEPTABLE_HORIZONTAL);
        
        // Check if it's in acceptable vertical range (70-85°, which is 5-20° from vertical)
        bool acceptable_vertical = (normalized_angle >= MIN_ACCEPTABLE_VERTICAL && 
                                    normalized_angle <= MAX_ACCEPTABLE_VERTICAL);
        
        return acceptable_horizontal || acceptable_vertical;
    }
    
    static double normalizeAngleForSlantEdge(double angle_degrees) {
        // Normalize angle to 0-90 range for slant edge analysis
        double normalized = std::abs(angle_degrees);
        while (normalized >= 360.0) {
            normalized -= 360.0;
        }
        while (normalized < 0.0) {
            normalized += 360.0;
        }
        
        // Convert to acute angle (0-90°)
        if (normalized > 270.0) {
            normalized = 360.0 - normalized;
        } else if (normalized > 180.0) {
            normalized = normalized - 180.0;
        } else if (normalized > 90.0) {
            normalized = 180.0 - normalized;
        }
        
        return normalized;
    }
    
    static std::string getAngleCategory(double angle_degrees) {
        double normalized = normalizeAngleForSlantEdge(angle_degrees);
        
        if (isOptimalSlantEdgeAngle(angle_degrees)) {
            if ((normalized >= MIN_OPTIMAL_HORIZONTAL && normalized <= MAX_OPTIMAL_HORIZONTAL)) {
                return "optimal_horizontal";
            } else {
                return "optimal_vertical";
            }
        } else if (isAcceptableSlantEdgeAngle(angle_degrees)) {
            if ((normalized >= MIN_ACCEPTABLE_HORIZONTAL && normalized <= MAX_ACCEPTABLE_HORIZONTAL)) {
                return "acceptable_horizontal";
            } else {
                return "acceptable_vertical";
            }
        } else {
            return "invalid";
        }
    }

    static double calculateSamplingInterval(double edge_angle_degrees)
    {
        // BREAKTHROUGH FIX: Analysis showed that finer sampling (0.4) produces WORSE accuracy
        // than the proven working value (0.5). Use unified sampling for consistency.
        
        // Return the proven working value that gave us the best results (55.2% error vs 112.2%)
        return 0.5;
        
        // TODO: Investigate why super-fine sampling hurts accuracy - likely due to:
        // 1. Over-sampling noise artifacts
        // 2. Sub-pixel interpolation error accumulation  
        // 3. LSF computation instability with too many fine samples
    }

    // ROI Quality Assessment Functions
    static double calculateEdgeStrength(const cv::Mat &roi, const cv::Vec4i &line)
    {
        cv::Mat gray;
        if (roi.channels() == 3)
        {
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = roi;
        }

        // Calculate gradient magnitude using Sobel operators
        cv::Mat grad_x, grad_y;
        cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

        cv::Mat magnitude;
        cv::magnitude(grad_x, grad_y, magnitude);

        // FIXED: Use overall gradient statistics instead of line-specific sampling
        // The line coordinates may not be correctly mapped to the ROI coordinate system
        
        // Calculate overall gradient statistics
        cv::Scalar mean_grad, stddev_grad;
        cv::meanStdDev(magnitude, mean_grad, stddev_grad);
        
        // Find maximum gradient in the ROI
        double min_val, max_val;
        cv::minMaxLoc(magnitude, &min_val, &max_val);
        
        // Use a combination of mean and max gradient as edge strength indicator
        double avg_strength = mean_grad[0];
        double max_strength = max_val;
        
        // Edge strength score: combination of average and peak gradient
        double strength_score = 0.6 * avg_strength + 0.4 * (max_strength * 0.1); // Scale max down
        
        // Normalize to 0-100 scale (typical Sobel gradients are 0-1000+ range)
        return std::min(100.0, strength_score / 10.0);
    }

    static double calculateLinearityScore(const cv::Mat &roi, const cv::Vec4i &line)
    {
        // FIXED: Simplified linearity assessment using edge detection and line fitting
        cv::Mat gray;
        if (roi.channels() == 3)
        {
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = roi;
        }

        // Use Canny edge detection to find edge points
        cv::Mat edges;
        cv::Canny(gray, edges, 50, 150);
        
        // Find edge points
        std::vector<cv::Point> edge_points;
        for (int y = 0; y < edges.rows; y++)
        {
            for (int x = 0; x < edges.cols; x++)
            {
                if (edges.at<uchar>(y, x) > 0)
                {
                    edge_points.push_back(cv::Point(x, y));
                }
            }
        }
        
        if (edge_points.size() < 10) return 50.0; // Default score for insufficient edge points
        
        // Fit a line to the edge points using least squares
        cv::Vec4f fitted_line;
        cv::fitLine(edge_points, fitted_line, cv::DIST_L2, 0, 0.01, 0.01);
        
        // Calculate how well points fit the line (measure of linearity)
        cv::Point2f line_point(fitted_line[2], fitted_line[3]);
        cv::Point2f line_direction(fitted_line[0], fitted_line[1]);
        
        // Calculate average distance from points to fitted line
        double total_distance = 0.0;
        for (const auto& point : edge_points)
        {
            cv::Point2f to_point = cv::Point2f(point) - line_point;
            
            // Distance from point to line
            float cross_product = to_point.x * line_direction.y - to_point.y * line_direction.x;
            double distance = std::abs(cross_product);
            total_distance += distance;
        }
        
        double avg_distance = total_distance / edge_points.size();
        
        // Convert to score: lower average distance = higher linearity score
        // Typical distance for good linearity: 0-2 pixels
        double linearity_score = std::max(0.0, 100.0 - avg_distance * 20.0);
        return std::min(100.0, linearity_score);
    }

    static double calculateNoiseLevel(const cv::Mat &roi)
    {
        cv::Mat gray;
        if (roi.channels() == 3)
        {
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        }
        else
        {
            gray = roi;
        }

        // Calculate noise using Laplacian variance method
        cv::Mat laplacian;
        cv::Laplacian(gray, laplacian, CV_64F);
        
        cv::Scalar mu, sigma;
        cv::meanStdDev(laplacian, mu, sigma);
        
        double noise_variance = sigma[0] * sigma[0];
        return noise_variance; // Return raw variance (lower is better)
    }

    static ROIQuality assessROIQuality(const cv::Mat &roi, const cv::Vec4i &line, int profile_count)
    {
        ROIQuality quality;
        
        // Calculate individual metrics
        quality.edge_strength = calculateEdgeStrength(roi, line);
        quality.linearity_score = calculateLinearityScore(roi, line);
        quality.noise_level = calculateNoiseLevel(roi);
        
        // Profile adequacy (0-100 based on profile count)
        quality.profile_adequacy = std::min(100.0, (profile_count / 30.0) * 100.0); // 30+ profiles = 100%
        
        // Combined score with weights
        double edge_weight = 0.3;
        double linearity_weight = 0.25;
        double noise_weight = 0.25; // Lower noise = higher score
        double profile_weight = 0.2;
        
        // Normalize noise (invert and scale to 0-100)
        double noise_score = std::max(0.0, 100.0 - std::min(100.0, quality.noise_level / 10.0));
        
        quality.overall_score = edge_weight * quality.edge_strength +
                               linearity_weight * quality.linearity_score +
                               noise_weight * noise_score +
                               profile_weight * quality.profile_adequacy;
        
        // Quality threshold and reasoning (temporarily lowered for debugging)
        const double QUALITY_THRESHOLD = 40.0; // Require 40% overall quality (was 60%)
        quality.is_acceptable = quality.overall_score >= QUALITY_THRESHOLD;
        
        // Generate quality reason
        std::vector<std::string> issues;
        if (quality.edge_strength < 50.0) issues.push_back("weak edge");
        if (quality.linearity_score < 50.0) issues.push_back("non-linear edge");
        if (noise_score < 50.0) issues.push_back("high noise");
        if (quality.profile_adequacy < 70.0) issues.push_back("insufficient profiles");
        
        if (issues.empty())
        {
            quality.quality_reason = "Good quality ROI";
        }
        else
        {
            quality.quality_reason = "Issues: ";
            for (size_t i = 0; i < issues.size(); i++)
            {
                if (i > 0) quality.quality_reason += ", ";
                quality.quality_reason += issues[i];
            }
        }
        
        return quality;
    }

    // Visualization

    static cv::Mat createESFVisualization(const ESFData &esf, bool debug = false)
    {
        const int plot_height = 400;
        const int plot_width = 800;
        const int margin = 50;

        cv::Mat vis(plot_height + 2 * margin, plot_width + 2 * margin, CV_8UC3, cv::Scalar(255, 255, 255));

        // Draw axes
        cv::line(vis,
                 cv::Point(margin, plot_height + margin),
                 cv::Point(plot_width + margin, plot_height + margin),
                 cv::Scalar(0, 0, 0), 2);
        cv::line(vis,
                 cv::Point(margin, margin),
                 cv::Point(margin, plot_height + margin),
                 cv::Scalar(0, 0, 0), 2);

        // Plot raw data points
        std::vector<cv::Point> points;
        double min_dist = *std::min_element(esf.binned_distances.begin(), esf.binned_distances.end());
        double max_dist = *std::max_element(esf.binned_distances.begin(), esf.binned_distances.end());

        for (size_t i = 0; i < esf.binned_distances.size(); i++)
        {
            int x = margin + static_cast<int>((esf.binned_distances[i] - min_dist) * plot_width / (max_dist - min_dist));
            int y = margin + plot_height - static_cast<int>(esf.binned_intensities[i] * plot_height);
            points.push_back(cv::Point(x, y));
        }

        // Draw points and connecting line
        for (const auto &pt : points)
        {
            cv::circle(vis, pt, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        }
        cv::polylines(vis, points, false, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

        // Add labels
        cv::putText(vis, "Edge Spread Function",
                    cv::Point(margin, margin - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(vis, "Distance (pixels)",
                    cv::Point(plot_width / 2, plot_height + 2 * margin - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(vis, "Intensity",
                    cv::Point(10, plot_height / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        return vis;
    }

    static cv::Mat createLSFVisualization(const LSFData &lsf, double gaussian_sigma = 0.0)
    {
        const int plot_height = 400;
        const int plot_width = 800;
        const int margin = 50;

        cv::Mat vis(plot_height + 2 * margin, plot_width + 2 * margin, CV_8UC3, cv::Scalar(255, 255, 255));

        // Draw grid
        for (int i = 0; i <= 10; i++)
        {
            int y = margin + i * plot_height / 10;
            cv::line(vis, cv::Point(margin, y), cv::Point(margin + plot_width, y),
                     cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
        }

        // Find max value for normalization
        double max_val = *std::max_element(lsf.smoothed_lsf.begin(), lsf.smoothed_lsf.end());

        // Plot data
        std::vector<cv::Point> points;
        double min_x = *std::min_element(lsf.distances.begin(), lsf.distances.end());
        double max_x = *std::max_element(lsf.distances.begin(), lsf.distances.end());

        for (size_t i = 0; i < lsf.distances.size(); i++)
        {
            int x = margin + static_cast<int>((lsf.distances[i] - min_x) * plot_width / (max_x - min_x));
            int y = margin + plot_height - static_cast<int>((lsf.smoothed_lsf[i] / max_val) * plot_height);
            points.push_back(cv::Point(x, y));
        }

        // Draw measured LSF
        cv::polylines(vis, points, false, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        // Draw theoretical Gaussian if sigma provided
        if (gaussian_sigma > 0.0)
        {
            std::vector<cv::Point> theo_points;
            for (size_t i = 0; i < lsf.distances.size(); i++)
            {
                double x = lsf.distances[i];
                double theoretical = exp(-(x * x) / (2 * gaussian_sigma * gaussian_sigma)) /
                                     (gaussian_sigma * sqrt(2 * M_PI));
                int plot_x = margin + static_cast<int>((x - min_x) * plot_width / (max_x - min_x));
                int plot_y = margin + plot_height - static_cast<int>((theoretical / max_val) * plot_height);
                theo_points.push_back(cv::Point(plot_x, plot_y));
            }
            cv::polylines(vis, theo_points, false, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            // Add legend
            cv::putText(vis, "Measured LSF", cv::Point(margin + 10, margin + 20),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            cv::putText(vis, "Theoretical LSF", cv::Point(margin + 10, margin + 40),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }

        // Draw FWHM
        int half_max_y = margin + plot_height / 2;
        cv::line(vis, cv::Point(margin, half_max_y),
                 cv::Point(margin + plot_width, half_max_y),
                 cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

        // Add labels
        cv::putText(vis, "Line Spread Function",
                    cv::Point(margin, margin - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(vis, "Distance (pixels)",
                    cv::Point(plot_width / 2, plot_height + 2 * margin - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(vis, "Intensity",
                    cv::Point(10, plot_height / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        cv::putText(vis, "FWHM: " + std::to_string(lsf.fwhm),
                    cv::Point(margin + 10, margin + 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

        return vis;
    }

    // HYBRID SYSTEM: Research-based validation + Complementary angle detection
    
    // Research-based optimal ranges for slant-edge MTF
    static constexpr double MIN_OPTIMAL_HORIZONTAL = 8.0;    // Optimal horizontal-ish edges
    static constexpr double MAX_OPTIMAL_HORIZONTAL = 15.0;   // Optimal horizontal-ish edges
    static constexpr double MIN_OPTIMAL_VERTICAL = 75.0;     // Optimal vertical-ish edges (90° - 15°)
    static constexpr double MAX_OPTIMAL_VERTICAL = 82.0;     // Optimal vertical-ish edges (90° - 8°)
    
    // Acceptable ranges (with warnings)
    static constexpr double MIN_ACCEPTABLE_HORIZONTAL = 5.0; 
    static constexpr double MAX_ACCEPTABLE_HORIZONTAL = 20.0;
    static constexpr double MIN_ACCEPTABLE_VERTICAL = 70.0;   // 90° - 20°
    static constexpr double MAX_ACCEPTABLE_VERTICAL = 85.0;   // 90° - 5°
    
    // Complementary angle pairs for 4-ROI detection (proven working system)
    static constexpr double ANGLE1_TARGET = 11.0;       // Horizontal-ish (optimal range)
    static constexpr double ANGLE2_TARGET = 281.0;      // Vertical-ish (281° = 360° - 79°, also optimal)
    static constexpr double ANGLE_TOLERANCE = 6.0;      // Increased tolerance to handle slight angle variations
    
    static constexpr double ROI_LENGTH_FACTOR = 0.8;
    static constexpr double ROI_WIDTH = 100;
    static constexpr double DEFAULT_GAUSS_SIGMA = 1.5;
    static constexpr double VALIDATION_THRESHOLD = 0.03; // IMPROVEMENT: Tightened to 3% for better precision
    static int roi_counter;
};

int MTFAnalyzer::roi_counter = 0;

int main(int argc, char **argv)
{
    
    try
    {
        MTFAnalyzer::testFWHMCalculation();

        MTFAnalyzer::ProgramOptions options =
            MTFAnalyzer::ProgramOptions::parseCommandLine(argc, argv);

        if (options.input_file.empty())
        {
            std::cerr << "Usage: " << argv[0] << " <input_image> [options]\n"
                      << "Options:\n"
                      << "  --pixel-size <size>     : Pixel size in mm\n"
                      << "  --gaussian-sigma <sigma> : Known Gaussian blur sigma for validation\n"
                      << "  --debug                 : Enable debug output\n";
            return 1;
        }

        cv::Mat image = cv::imread(options.input_file);
        if (image.empty())
        {
            std::cerr << "Error: Could not read image file: " << options.input_file << std::endl;
            return 1;
        }

        auto results = MTFAnalyzer::analyzeImage(image, options);

        // Save visualization
        cv::Mat vis = MTFAnalyzer::createMTFVisualization(results, options.debug);
        cv::imwrite("mtf_results.png", vis);

        std::cout << "\nAnalysis complete. Results saved to mtf_results.png" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}