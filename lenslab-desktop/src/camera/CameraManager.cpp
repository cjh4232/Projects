#include "CameraManager.h"
#include "CameraDevice.h"
#include "app/Logger.h"

namespace lenslab {

CameraManager::CameraManager() = default;
CameraManager::~CameraManager() = default;

void CameraManager::init()
{
    LOG_INFO("CameraManager initializing...");

    // TODO: Initialize GenTL system
    // - Load GenTL producers (.cti files)
    // - Initialize transport layer

    refreshDeviceList();

    LOG_INFO("CameraManager initialized");
}

void CameraManager::shutdown()
{
    LOG_INFO("CameraManager shutting down...");

    disconnectDevice();

    // TODO: Cleanup GenTL system

    LOG_INFO("CameraManager shutdown complete");
}

void CameraManager::update()
{
    if (m_activeDevice) {
        m_activeDevice->update();
    }
}

void CameraManager::refreshDeviceList()
{
    LOG_INFO("Refreshing camera device list...");

    m_deviceNames.clear();
    m_deviceIds.clear();

    // TODO: Enumerate GenTL devices
    // For now, add a simulated camera for UI development
    m_deviceNames.push_back("[Simulated] Test Camera 1920x1080");
    m_deviceIds.push_back("SIM_001");

    LOG_INFO("Found {} camera(s)", m_deviceNames.size());
}

bool CameraManager::connectDevice(size_t index)
{
    if (index >= m_deviceIds.size()) {
        LOG_ERROR("Invalid device index: {}", index);
        return false;
    }

    LOG_INFO("Connecting to device: {}", m_deviceNames[index]);

    // Disconnect existing device
    disconnectDevice();

    // TODO: Create real device connection via GenTL
    m_activeDevice = std::make_unique<CameraDevice>();

    if (!m_activeDevice->connect(m_deviceIds[index])) {
        LOG_ERROR("Failed to connect to device");
        m_activeDevice.reset();
        return false;
    }

    LOG_INFO("Connected to device: {}", m_deviceNames[index]);
    return true;
}

void CameraManager::disconnectDevice()
{
    if (m_activeDevice) {
        LOG_INFO("Disconnecting active device...");
        m_activeDevice->disconnect();
        m_activeDevice.reset();
    }
}

bool CameraManager::isConnected() const
{
    return m_activeDevice && m_activeDevice->isConnected();
}

} // namespace lenslab
