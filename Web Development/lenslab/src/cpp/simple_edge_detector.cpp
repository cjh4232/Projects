#include <opencv2/opencv.hpp>
#include <iostream>

class EdgeDetector {
public:
    static void detectEdges(const cv::Mat& input) {
        cv::Mat masked = input.clone();
        
        // Create center mask
        int radius = std::min(input.rows, input.cols) / 100 * 5;  // 5% of image size
        cv::Point center(input.cols/2, input.rows/2);
        cv::circle(masked, center, radius, cv::Scalar(0,0,0), -1);  // -1 fills circle
        
        cv::Mat gray;
        cv::cvtColor(masked, gray, cv::COLOR_BGR2GRAY);
        
        cv::Mat edges;
        cv::Canny(gray, edges, 50, 150);
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 100, 10);
        
        cv::Mat line_image = masked.clone();
        
        for(const auto& l : lines) {
            double angle = std::atan2(-(l[3] - l[1]), l[2] - l[0]) * 180.0 / CV_PI;
            while(angle < 0) angle += 360.0;
            
            std::cout << "Line angle: " << angle << " degrees" << std::endl;
            cv::line(line_image, cv::Point(l[0], l[1]), 
                    cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 2);
        }
        
        cv::imwrite("masked_input.png", masked);
        cv::imwrite("detected_lines.png", line_image);
    }
};

int main(int argc, char** argv) {
    if(argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>\n";
        return -1;
    }
    
    cv::Mat image = cv::imread(argv[1]);
    if(image.empty()) {
        std::cerr << "Could not open image: " << argv[1] << "\n";
        return -1;
    }
    
    EdgeDetector::detectEdges(image);
    return 0;
}