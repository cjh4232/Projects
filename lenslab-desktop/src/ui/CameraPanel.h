#pragma once

namespace lenslab {

class Application;

/**
 * Camera list and control panel
 */
class CameraPanel
{
public:
    explicit CameraPanel(Application& app);

    void render();

    bool& isVisible() { return m_visible; }

private:
    void renderDeviceList();
    void renderControls();
    void renderParameters();

    Application& m_app;
    bool m_visible = true;

    // UI state
    int m_selectedDevice = -1;
    float m_exposure = 10.0f;     // ms
    float m_gain = 0.0f;          // dB
    float m_frameRate = 30.0f;    // FPS
};

} // namespace lenslab
