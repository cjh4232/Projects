#include <opencv2/opencv.hpp>
#include <iostream>

class EdgeDetector {
private:
    static constexpr float TARGET_ANGLE1 = 11.0f;  // Rotated horizontal
    static constexpr float TARGET_ANGLE2 = 101.0f; // Rotated vertical
    static constexpr float ANGLE_TOLERANCE = 5.0f;

    static bool isValidEdgeAngle(float angle) {
        while (angle > 180.0f) angle -= 180.0f;
        while (angle < 0.0f) angle += 180.0f;
        
        float diff1 = std::abs(angle - TARGET_ANGLE1);
        float diff2 = std::abs(angle - TARGET_ANGLE2);
        float diff3 = std::abs(angle - (180.0f - TARGET_ANGLE1));
        float diff4 = std::abs(angle - (180.0f - TARGET_ANGLE2));
        
        return (diff1 < ANGLE_TOLERANCE || diff2 < ANGLE_TOLERANCE || 
                diff3 < ANGLE_TOLERANCE || diff4 < ANGLE_TOLERANCE);
    }

public:
    static void detectEdges(const cv::Mat& input, const std::string& output_prefix) {
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        
        // Calculate gradients
        cv::Mat grad_x, grad_y;
        cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
        
        // Calculate magnitude and angle
        cv::Mat magnitude, angle;
        cv::cartToPolar(grad_x, grad_y, magnitude, angle, true);
        
        // Create visualizations
        cv::Mat edge_mask = cv::Mat::zeros(input.size(), CV_8UC1);
        cv::Mat angle_vis = input.clone();
        cv::Mat combined_vis = input.clone();
        
        double mag_threshold = cv::mean(magnitude)[0] * 2.0;
        
        for(int y = 0; y < angle.rows; y++) {
            for(int x = 0; x < angle.cols; x++) {
                float ang = angle.at<float>(y,x);
                float mag = magnitude.at<float>(y,x);
                
                if(mag > mag_threshold && isValidEdgeAngle(ang)) {
                    edge_mask.at<uchar>(y,x) = 255;
                    
                    // Color code based on angle
                    if(std::abs(ang - TARGET_ANGLE1) < ANGLE_TOLERANCE || 
                       std::abs(ang - (180.0f - TARGET_ANGLE1)) < ANGLE_TOLERANCE) {
                        angle_vis.at<cv::Vec3b>(y,x) = cv::Vec3b(0, 0, 255);  // Red for ~11°
                    } else {
                        angle_vis.at<cv::Vec3b>(y,x) = cv::Vec3b(255, 0, 0);  // Blue for ~101°
                    }
                }
            }
        }
        
        // Create overlay visualization
        cv::addWeighted(input, 0.7, angle_vis, 0.3, 0, combined_vis);
        
        // Save debug images
        cv::imwrite(output_prefix + "_1_gray.png", gray);
        cv::imwrite(output_prefix + "_2_edge_mask.png", edge_mask);
        cv::imwrite(output_prefix + "_3_angle_vis.png", angle_vis);
        cv::imwrite(output_prefix + "_4_combined.png", combined_vis);
    }
};

int main(int argc, char** argv) {
    if(argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_prefix>\n";
        return -1;
    }
    
    cv::Mat image = cv::imread(argv[1]);
    if(image.empty()) {
        std::cerr << "Could not open image: " << argv[1] << "\n";
        return -1;
    }
    
    EdgeDetector::detectEdges(image, argv[2]);
    return 0;
}