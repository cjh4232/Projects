#pragma once

#include <memory>

namespace lenslab {

class Application;
class CameraPanel;
class PreviewPanel;
class AnalysisPanel;

/**
 * Manages all UI panels and ImGui rendering
 */
class UIManager
{
public:
    explicit UIManager(Application& app);
    ~UIManager();

    void init();
    void shutdown();
    void render();

private:
    void renderMainMenuBar();
    void renderStatusBar();

    Application& m_app;

    std::unique_ptr<CameraPanel> m_cameraPanel;
    std::unique_ptr<PreviewPanel> m_previewPanel;
    std::unique_ptr<AnalysisPanel> m_analysisPanel;

    // UI state
    bool m_showDemoWindow = false;
    bool m_showMetricsWindow = false;
};

} // namespace lenslab
