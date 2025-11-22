#pragma once

#include <memory>
#include <vector>
#include <string>

namespace lenslab {

class CameraDevice;

/**
 * Camera discovery and management
 */
class CameraManager
{
public:
    CameraManager();
    ~CameraManager();

    void init();
    void shutdown();
    void update();

    // Discovery
    void refreshDeviceList();
    size_t getDeviceCount() const { return m_devices.size(); }

    // Connection
    bool connectDevice(size_t index);
    void disconnectDevice();
    bool isConnected() const;

    // Access
    CameraDevice* getActiveDevice() { return m_activeDevice.get(); }
    const std::vector<std::string>& getDeviceNames() const { return m_deviceNames; }

private:
    std::vector<std::string> m_deviceNames;
    std::vector<std::string> m_deviceIds;
    std::unique_ptr<CameraDevice> m_activeDevice;
};

} // namespace lenslab
