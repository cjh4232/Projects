#pragma once

#include <string>
#include <opencv2/core.hpp>

namespace lenslab {

/**
 * Represents a single camera device
 */
class CameraDevice
{
public:
    CameraDevice();
    ~CameraDevice();

    bool connect(const std::string& deviceId);
    void disconnect();
    void update();

    bool isConnected() const { return m_connected; }
    bool hasNewFrame() const { return m_hasNewFrame; }

    // Frame access
    const cv::Mat& getCurrentFrame() const { return m_currentFrame; }
    void acknowledgeFrame() { m_hasNewFrame = false; }

    // Camera info
    std::string getVendor() const { return m_vendor; }
    std::string getModel() const { return m_model; }
    std::string getSerialNumber() const { return m_serialNumber; }
    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    double getFrameRate() const { return m_frameRate; }

    // Parameters
    double getExposure() const { return m_exposure; }
    void setExposure(double microseconds);

    double getGain() const { return m_gain; }
    void setGain(double db);

    // Acquisition control
    void startAcquisition();
    void stopAcquisition();
    bool isAcquiring() const { return m_acquiring; }

private:
    void generateTestFrame();  // For simulated camera

    bool m_connected = false;
    bool m_acquiring = false;
    bool m_hasNewFrame = false;
    bool m_simulated = true;  // TODO: Remove when GenTL is implemented

    // Device info
    std::string m_deviceId;
    std::string m_vendor = "Simulated";
    std::string m_model = "Test Camera";
    std::string m_serialNumber = "SIM001";

    // Frame data
    cv::Mat m_currentFrame;
    int m_width = 1920;
    int m_height = 1080;
    double m_frameRate = 30.0;

    // Parameters
    double m_exposure = 10000.0;  // microseconds
    double m_gain = 0.0;          // dB

    // Timing
    uint64_t m_lastFrameTime = 0;
    uint64_t m_frameCount = 0;
};

} // namespace lenslab
