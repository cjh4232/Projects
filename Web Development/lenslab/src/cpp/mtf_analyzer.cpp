#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

struct EdgeROI {
    cv::Rect roi;
    float angle;
    bool darkToLight;
    float mtf50;
};

class MTFAnalyzer {
private:
    static constexpr float EDGE_ANGLE_MIN = 3.0f;    // minimum edge angle in degrees
    static constexpr float EDGE_ANGLE_MAX = 15.0f;   // maximum edge angle in degrees
    static constexpr int OVERSAMPLE = 4;             // oversampling factor
    static constexpr int ESF_BINS = 200;             // number of bins for ESF
    static constexpr int EDGE_WIDTH = 40;            // edge window width in pixels

    static bool isLinearEdge(const cv::Mat& component) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(component, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) return false;
        
        // Fit line to contour
        cv::Vec4f line;
        cv::fitLine(contours[0], line, cv::DIST_L2, 0, 0.01, 0.01);
        
        // Calculate distance of points to line
        float maxDist = 0;
        for (const cv::Point& pt : contours[0]) {
            float dist = std::abs((pt.x - line[2]) * line[1] - (pt.y - line[3]) * line[0]);
            maxDist = std::max(maxDist, dist);
        }
        
        // If max distance is too large, it's likely curved
        return maxDist < 5.0; // Adjust threshold as needed
    }

    static cv::Mat visualizeAngleDetection(const cv::Mat& angle, const cv::Mat& magnitude) {
        cv::Mat angleVis = cv::Mat::zeros(angle.size(), CV_8UC3);
        for(int y = 0; y < angle.rows; y++) {
            for(int x = 0; x < angle.cols; x++) {
                float ang = angle.at<float>(y, x);
                float mag = magnitude.at<float>(y, x);
                
                if(isValidEdgeAngle(ang) && mag > cv::mean(magnitude)[0] * 2.0f) {
                    // Close to 11 degrees (rotated horizontal)
                    if(std::abs(ang - 11.0f) < 5.0f || std::abs(ang - (180.0f + 11.0f)) < 5.0f)
                        angleVis.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);  // Red
                    // Close to 101 degrees (rotated vertical)
                    else if(std::abs(ang - 101.0f) < 5.0f || std::abs(ang - (180.0f + 101.0f)) < 5.0f)
                        angleVis.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 0, 0);  // Blue
                }
            }
        }
        return angleVis;
    }

    static bool isValidComponent(const cv::Mat& stats, int idx) {
        int area = stats.at<int>(idx, cv::CC_STAT_AREA);
        int width = stats.at<int>(idx, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(idx, cv::CC_STAT_HEIGHT);
        
        float aspectRatio = float(width) / float(height);
        
        return (area > 100 && area < 10000) && // Area constraints
               (aspectRatio > 0.1 && aspectRatio < 10.0); // Not too elongated
    }
    
    static std::vector<EdgeROI> detectEdges(const cv::Mat& img, const std::string& outputPrefix) {
    std::vector<EdgeROI> edges;
    cv::Mat gray;
    
    // Convert to grayscale
    if (img.channels() > 1) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        img.convertTo(gray, CV_8UC1);
    }
    
    // Convert to float for gradient computation
    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F, 1.0/255.0);

    // Compute gradients (for visualization)
    cv::Mat dx, dy;
    cv::Sobel(grayFloat, dx, CV_32F, 1, 0, 3);
    cv::Sobel(grayFloat, dy, CV_32F, 0, 1, 3);
    cv::Mat magnitude, angle;
    cv::cartToPolar(dx, dy, magnitude, angle, true);
    
    // Edge detection
    cv::Mat edges_img;
    cv::Canny(gray, edges_img, 50, 150);
    
    // Detect lines
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges_img, lines, 1, CV_PI/180, 50, 100, 10);
    
    // Create visualization for all detected lines
    cv::Mat allLinesVis = img.clone();
    for(const auto& l : lines) {
        cv::line(allLinesVis, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), 
                 cv::Scalar(255, 0, 0), 1);
    }
    cv::imwrite(outputPrefix + "_1_all_lines.png", allLinesVis);
    
    // Filter and sort lines by angle and length
    std::vector<std::pair<cv::Vec4i, double>> filtered_lines;
    cv::Mat filteredLinesVis = img.clone();
    for(const auto& l : lines) {
        double angle = std::atan2(l[3] - l[1], l[2] - l[0]) * 180.0 / CV_PI;
        while(angle < 0) angle += 180;
        
        if((std::abs(angle - 11) < 5) || (std::abs(angle - 101) < 5)) {
            double length = std::sqrt(std::pow(l[2] - l[0], 2) + std::pow(l[3] - l[1], 2));
            filtered_lines.push_back({l, length});
            // Draw filtered lines in green
            cv::line(filteredLinesVis, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), 
                    cv::Scalar(0, 255, 0), 2);
        }
    }
    cv::imwrite(outputPrefix + "_2_filtered_lines.png", filteredLinesVis);
    
    // Sort by length
    std::sort(filtered_lines.begin(), filtered_lines.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Take top 4 lines
    cv::Mat selectedLinesVis = img.clone();
    for(int i = 0; i < std::min(4, (int)filtered_lines.size()); i++) {
        const auto& line = filtered_lines[i].first;
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        int padding = 20;
        
        EdgeROI edge;
        edge.roi = cv::Rect(
            std::max(0, std::min(x1, x2) - padding),
            std::max(0, std::min(y1, y2) - padding),
            std::min(img.cols - 1, std::abs(x2 - x1) + 2*padding),
            std::min(img.rows - 1, std::abs(y2 - y1) + 2*padding)
        );
        edge.angle = std::atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI;
        
        // Draw selected line and its ROI
        cv::line(selectedLinesVis, cv::Point(x1, y1), cv::Point(x2, y2), 
                 cv::Scalar(0, 255, 0), 2);
        cv::rectangle(selectedLinesVis, edge.roi, cv::Scalar(0, 0, 255), 2);
        
        edges.push_back(edge);
    }
    cv::imwrite(outputPrefix + "_3_selected_lines.png", selectedLinesVis);
    
    // Create the rest of the debug visualizations
    debugVisualization(img, grayFloat, dx, dy, magnitude, angle, edges_img, 
                      cv::Mat::zeros(img.size(), CV_32S), outputPrefix);
    
    return edges;
}

    static float findEdgeLocation(const cv::Mat& line) {
        // Ensure input is single channel
        cv::Mat singleChannel;
        if (line.channels() > 1) {
            cv::cvtColor(line, singleChannel, cv::COLOR_BGR2GRAY);
        } else {
            singleChannel = line;
        }
        
        cv::Mat smooth;
        cv::GaussianBlur(singleChannel, smooth, cv::Size(5, 1), 1.0);
        
        cv::Mat derivative;
        cv::Sobel(smooth, derivative, CV_32F, 1, 0, 3);
        
        cv::Mat absDerivative;
        cv::convertScaleAbs(derivative, absDerivative);
        
        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(absDerivative, &minVal, &maxVal, &minLoc, &maxLoc);
        
        const int window = 3;
        float sum = 0.0f, weightedSum = 0.0f;
        for (int x = std::max(0, maxLoc.x - window); 
             x < std::min(derivative.cols, maxLoc.x + window + 1); x++) {
            float val = std::abs(derivative.at<float>(0, x));
            sum += val;
            weightedSum += val * x;
        }
        
        return sum > 0.0f ? weightedSum / sum : maxLoc.x;
    }

    static std::vector<float> computeESF(const cv::Mat& roi, float angle) {
        cv::Mat rotated;
        cv::Point2f center(roi.cols/2.0f, roi.rows/2.0f);
        cv::Mat rotMat = cv::getRotationMatrix2D(center, -angle, 1.0);
        cv::warpAffine(roi, rotated, rotMat, roi.size());
        
        std::vector<float> positions;
        std::vector<float> intensities;
        
        for (int y = 0; y < rotated.rows; y++) {
            cv::Mat row = rotated.row(y);
            float edgePos = findEdgeLocation(row);
            
            // Sample intensities around edge
            for (int x = std::max(0, int(edgePos) - EDGE_WIDTH/2);
                 x < std::min(rotated.cols, int(edgePos) + EDGE_WIDTH/2); x++) {
                positions.push_back((x - edgePos) * OVERSAMPLE);
                intensities.push_back(row.at<float>(0, x));
            }
        }
        
        // Bin the samples
        std::vector<float> esf(ESF_BINS, 0.0f);
        std::vector<int> counts(ESF_BINS, 0);
        
        float minPos = *std::min_element(positions.begin(), positions.end());
        float maxPos = *std::max_element(positions.begin(), positions.end());
        float binSize = (maxPos - minPos) / (ESF_BINS - 1);
        
        for (size_t i = 0; i < positions.size(); i++) {
            int bin = static_cast<int>((positions[i] - minPos) / binSize);
            if (bin >= 0 && bin < ESF_BINS) {
                esf[bin] += intensities[i];
                counts[bin]++;
            }
        }
        
        // Average bins
        for (int i = 0; i < ESF_BINS; i++) {
            if (counts[i] > 0) {
                esf[i] /= counts[i];
            } else if (i > 0) {
                esf[i] = esf[i-1];  // Fill gaps
            }
        }
        
        return esf;
    }

    static std::vector<float> computeMTF(const std::vector<float>& esf) {
        // Compute LSF (derivative of ESF)
        std::vector<float> lsf(esf.size() - 1);
        for (size_t i = 0; i < lsf.size(); i++) {
            lsf[i] = esf[i+1] - esf[i];
        }
        
        // Apply Hamming window
        for (size_t i = 0; i < lsf.size(); i++) {
            float w = 0.54f - 0.46f * std::cos(2.0f * CV_PI * i / (lsf.size() - 1));
            lsf[i] *= w;
        }
        
        // Prepare for DFT
        cv::Mat lsfMat(lsf.size(), 1, CV_32F);
        for (size_t i = 0; i < lsf.size(); i++) {
            lsfMat.at<float>(i) = lsf[i];
        }
        
        // Compute DFT
        cv::Mat dft;
        cv::dft(lsfMat, dft, cv::DFT_COMPLEX_OUTPUT);
        
        // Compute magnitude spectrum
        std::vector<float> mtf(dft.rows/2);
        for (size_t i = 0; i < mtf.size(); i++) {
            float re = dft.at<cv::Vec2f>(i)[0];
            float im = dft.at<cv::Vec2f>(i)[1];
            mtf[i] = std::sqrt(re*re + im*im);
        }
        
        // Normalize
        float maxVal = *std::max_element(mtf.begin(), mtf.end());
        if (maxVal > 0) {
            for (float& v : mtf) v /= maxVal;
        }
        
        return mtf;
    }

    static float findMTF50(const std::vector<float>& mtf) {
        for (size_t i = 0; i < mtf.size() - 1; i++) {
            if (mtf[i] >= 0.5f && mtf[i+1] < 0.5f) {
                float t = (0.5f - mtf[i+1]) / (mtf[i] - mtf[i+1]);
                return (i + t) / static_cast<float>(mtf.size());
            }
        }
        return 0.0f;
    }

    static void debugVisualization(const cv::Mat& img,           // Original input image
                                  const cv::Mat& grayFloat,    // Grayscale float image
                                  const cv::Mat& dx,           // X gradient
                                  const cv::Mat& dy,           // Y gradient
                                  const cv::Mat& magnitude,    // Gradient magnitude
                                  const cv::Mat& angle,        // Gradient angle
                                  const cv::Mat& validEdges,   // Valid edges mask
                                  const cv::Mat& labels,       // Connected components labels
                                  const std::string& outputPrefix) {
        // 1. Original grayscale
        cv::Mat grayVis;
        cv::normalize(grayFloat, grayVis, 0, 255, cv::NORM_MINMAX);
        grayVis.convertTo(grayVis, CV_8U);
        cv::imwrite(outputPrefix + "_1_gray.png", grayVis);

        // 2. Gradient magnitude visualization
        cv::Mat gradientVis;
        cv::normalize(magnitude, gradientVis, 0, 255, cv::NORM_MINMAX);
        gradientVis.convertTo(gradientVis, CV_8U);
        cv::imwrite(outputPrefix + "_2_gradients.png", gradientVis);

        // 3. Gradient direction visualization (using HSV color wheel)
        cv::Mat angleVis;
        cv::Mat angleHSV(angle.size(), CV_8UC3);
        for (int y = 0; y < angle.rows; y++) {
            for (int x = 0; x < angle.cols; x++) {
                float ang = angle.at<float>(y, x);
                float mag = magnitude.at<float>(y, x);
                // Normalize magnitude for value channel
                float val = std::min(1.0f, mag * 5.0f);
                angleHSV.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    static_cast<uchar>(ang / 2), // Hue (0-180)
                    255,                         // Saturation
                    static_cast<uchar>(val * 255) // Value
                );
            }
        }
        cv::cvtColor(angleHSV, angleVis, cv::COLOR_HSV2BGR);
        cv::imwrite(outputPrefix + "_3_angles.png", angleVis);

        // 4. Valid edges after angle filtering
        cv::imwrite(outputPrefix + "_4_valid_edges.png", validEdges);

        // 5. Angle-filtered edges visualization
        cv::Mat angleFilterVis = visualizeAngleDetection(angle, magnitude);
        cv::imwrite(outputPrefix + "_5_angle_detection.png", angleFilterVis);

        // 6. Connected components visualization
        cv::Mat labelVis;
        labels.convertTo(labelVis, CV_8U, 255.0 / std::max(1, labels.rows));
        cv::applyColorMap(labelVis, labelVis, cv::COLORMAP_JET);
        cv::imwrite(outputPrefix + "_6_components.png", labelVis);

        // 7. Combined visualization
        cv::Mat combinedVis = img.clone();
        cv::Mat validEdgesRGB;
        cv::cvtColor(validEdges, validEdgesRGB, cv::COLOR_GRAY2BGR);
        cv::addWeighted(combinedVis, 0.7, validEdgesRGB, 0.3, 0, combinedVis);
        cv::imwrite(outputPrefix + "_7_combined.png", combinedVis);
    }

    static bool isValidEdgeAngle(float angle) {
        // Normalize angle to 0-180
        while (angle > 180.0f) angle -= 180.0f;
        
        const float targetAngle1 = 11.0f;  // Rotated horizontal edge
        const float targetAngle2 = 101.0f; // Rotated vertical edge
        const float tolerance = 5.0f;      // Tolerance window
        
        float diff1 = std::abs(angle - targetAngle1);
        float diff2 = std::abs(angle - targetAngle2);
        float diff3 = std::abs(angle - (180.0f - targetAngle1));
        float diff4 = std::abs(angle - (180.0f - targetAngle2));
        
        return (diff1 < tolerance || diff2 < tolerance || 
                diff3 < tolerance || diff4 < tolerance);
    }

public:
    static void analyzeMTF(const cv::Mat& image, const std::string& outputPrefix) {
        // Detect edges
        std::vector<EdgeROI> edges = detectEdges(image, outputPrefix);
        
        // Create visualization image
        cv::Mat visualization = image.clone();
        
        // Process each edge
        for (size_t i = 0; i < edges.size(); i++) {
            // Convert ROI to float and normalize
            cv::Mat roi;
            image(edges[i].roi).convertTo(roi, CV_32F);
            roi /= 255.0f;
            
            // Compute ESF
            std::vector<float> esf = computeESF(roi, edges[i].angle);
            
            // Compute MTF
            std::vector<float> mtf = computeMTF(esf);
            
            // Find MTF50
            edges[i].mtf50 = findMTF50(mtf);
            
            // Draw ROI and MTF50 value on visualization
            cv::rectangle(visualization, edges[i].roi, cv::Scalar(0, 255, 0), 2);
            cv::putText(visualization, 
                       cv::format("MTF50: %.2f", edges[i].mtf50),
                       cv::Point(edges[i].roi.x, edges[i].roi.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            
            // Save MTF curve
            std::ofstream mtfFile(outputPrefix + "_mtf_" + std::to_string(i) + ".txt");
            for (size_t j = 0; j < mtf.size(); j++) {
                float freq = static_cast<float>(j) / mtf.size();
                mtfFile << freq << " " << mtf[j] << "\n";
            }
            mtfFile.close();
        }
        
        // Save visualization
        cv::imwrite(outputPrefix + "_visualization.png", visualization);
    }
};

// Main function for testing
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_prefix>\n";
        return -1;
    }
    
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Could not open image: " << argv[1] << "\n";
        return -1;
    }
    
    MTFAnalyzer::analyzeMTF(image, argv[2]);
    return 0;
}