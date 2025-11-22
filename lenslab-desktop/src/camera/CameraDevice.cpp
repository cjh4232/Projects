#include "CameraDevice.h"
#include "app/Logger.h"
#include <chrono>
#include <cmath>

namespace lenslab {

CameraDevice::CameraDevice() = default;
CameraDevice::~CameraDevice()
{
    disconnect();
}

bool CameraDevice::connect(const std::string& deviceId)
{
    LOG_INFO("CameraDevice connecting: {}", deviceId);

    m_deviceId = deviceId;

    // TODO: Implement real GenTL connection
    // For now, use simulated camera
    m_simulated = true;
    m_connected = true;

    // Initialize frame buffer
    m_currentFrame = cv::Mat(m_height, m_width, CV_8UC1, cv::Scalar(128));

    LOG_INFO("CameraDevice connected (simulated mode)");
    return true;
}

void CameraDevice::disconnect()
{
    if (!m_connected) return;

    LOG_INFO("CameraDevice disconnecting...");

    stopAcquisition();
    m_connected = false;
    m_currentFrame.release();

    LOG_INFO("CameraDevice disconnected");
}

void CameraDevice::update()
{
    if (!m_connected || !m_acquiring) return;

    auto now = std::chrono::steady_clock::now();
    auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    // Calculate frame interval
    double intervalMs = 1000.0 / m_frameRate;

    if (nowMs - m_lastFrameTime >= static_cast<uint64_t>(intervalMs)) {
        if (m_simulated) {
            generateTestFrame();
        } else {
            // TODO: Grab frame from GenTL
        }

        m_hasNewFrame = true;
        m_lastFrameTime = nowMs;
        m_frameCount++;
    }
}

void CameraDevice::startAcquisition()
{
    if (!m_connected || m_acquiring) return;

    LOG_INFO("Starting acquisition...");

    m_acquiring = true;
    m_frameCount = 0;
    m_lastFrameTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    // TODO: Start GenTL acquisition
}

void CameraDevice::stopAcquisition()
{
    if (!m_acquiring) return;

    LOG_INFO("Stopping acquisition...");

    m_acquiring = false;

    // TODO: Stop GenTL acquisition
}

void CameraDevice::setExposure(double microseconds)
{
    m_exposure = microseconds;
    LOG_DEBUG("Exposure set to: {} us", m_exposure);

    // TODO: Set GenTL parameter
}

void CameraDevice::setGain(double db)
{
    m_gain = db;
    LOG_DEBUG("Gain set to: {} dB", m_gain);

    // TODO: Set GenTL parameter
}

void CameraDevice::generateTestFrame()
{
    // Generate a test pattern with a slanted edge (for MTF testing)
    double t = static_cast<double>(m_frameCount) * 0.02;  // Slow animation

    for (int y = 0; y < m_height; y++) {
        uint8_t* row = m_currentFrame.ptr<uint8_t>(y);
        for (int x = 0; x < m_width; x++) {
            // Create slanted edge pattern
            double edgeX = m_width / 2.0 + std::sin(t) * 50;
            double slope = 0.1;  // ~6 degree angle
            double edgePos = edgeX + slope * (y - m_height / 2.0);

            // Soft edge (simulates blur)
            double dist = x - edgePos;
            double blurWidth = 3.0;  // Blur sigma in pixels
            double value = 0.5 * (1.0 + std::tanh(dist / blurWidth));

            // Scale to 8-bit with some noise
            double noise = (rand() % 10 - 5) / 255.0;
            row[x] = static_cast<uint8_t>(std::clamp((value + noise) * 200 + 28, 0.0, 255.0));
        }
    }
}

} // namespace lenslab
