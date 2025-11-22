#include "CameraPanel.h"
#include "app/Application.h"
#include "camera/CameraManager.h"
#include "camera/CameraDevice.h"

#include <imgui.h>

namespace lenslab {

CameraPanel::CameraPanel(Application& app)
    : m_app(app)
{
}

void CameraPanel::render()
{
    if (!m_visible) return;

    ImGui::Begin("Camera", &m_visible);

    renderDeviceList();
    ImGui::Separator();
    renderControls();
    ImGui::Separator();
    renderParameters();

    ImGui::End();
}

void CameraPanel::renderDeviceList()
{
    auto& camMgr = m_app.getCameraManager();
    const auto& devices = camMgr.getDeviceNames();

    ImGui::Text("Devices");

    if (ImGui::Button("Refresh")) {
        camMgr.refreshDeviceList();
        m_selectedDevice = -1;
    }

    ImGui::SameLine();
    bool canConnect = m_selectedDevice >= 0 && !camMgr.isConnected();
    if (!canConnect) ImGui::BeginDisabled();
    if (ImGui::Button("Connect")) {
        camMgr.connectDevice(m_selectedDevice);
    }
    if (!canConnect) ImGui::EndDisabled();

    ImGui::SameLine();
    bool canDisconnect = camMgr.isConnected();
    if (!canDisconnect) ImGui::BeginDisabled();
    if (ImGui::Button("Disconnect")) {
        camMgr.disconnectDevice();
    }
    if (!canDisconnect) ImGui::EndDisabled();

    // Device list
    ImGui::BeginChild("DeviceList", ImVec2(0, 100), true);
    for (size_t i = 0; i < devices.size(); i++) {
        bool isSelected = (m_selectedDevice == static_cast<int>(i));
        if (ImGui::Selectable(devices[i].c_str(), isSelected)) {
            m_selectedDevice = static_cast<int>(i);
        }
    }
    ImGui::EndChild();
}

void CameraPanel::renderControls()
{
    auto& camMgr = m_app.getCameraManager();
    bool connected = camMgr.isConnected();

    ImGui::Text("Acquisition");

    if (!connected) ImGui::BeginDisabled();

    auto* device = camMgr.getActiveDevice();
    bool acquiring = device && device->isAcquiring();

    if (acquiring) {
        if (ImGui::Button("Stop", ImVec2(100, 0))) {
            device->stopAcquisition();
        }
    } else {
        if (ImGui::Button("Start", ImVec2(100, 0))) {
            device->startAcquisition();
        }
    }

    ImGui::SameLine();
    if (ImGui::Button("Capture", ImVec2(100, 0))) {
        // TODO: Capture single frame
    }

    if (!connected) ImGui::EndDisabled();
}

void CameraPanel::renderParameters()
{
    auto& camMgr = m_app.getCameraManager();
    bool connected = camMgr.isConnected();

    ImGui::Text("Parameters");

    if (!connected) ImGui::BeginDisabled();

    auto* device = camMgr.getActiveDevice();

    // Exposure
    if (ImGui::SliderFloat("Exposure (ms)", &m_exposure, 0.1f, 100.0f, "%.1f")) {
        if (device) {
            device->setExposure(m_exposure * 1000.0);  // Convert to us
        }
    }

    // Gain
    if (ImGui::SliderFloat("Gain (dB)", &m_gain, 0.0f, 24.0f, "%.1f")) {
        if (device) {
            device->setGain(m_gain);
        }
    }

    // Frame rate (read-only display)
    if (device) {
        ImGui::Text("Frame Rate: %.1f FPS", device->getFrameRate());
        ImGui::Text("Resolution: %d x %d", device->getWidth(), device->getHeight());
    }

    if (!connected) ImGui::EndDisabled();

    // Camera info
    if (connected && device) {
        ImGui::Separator();
        ImGui::Text("Camera Info");
        ImGui::BulletText("Vendor: %s", device->getVendor().c_str());
        ImGui::BulletText("Model: %s", device->getModel().c_str());
        ImGui::BulletText("Serial: %s", device->getSerialNumber().c_str());
    }
}

} // namespace lenslab
