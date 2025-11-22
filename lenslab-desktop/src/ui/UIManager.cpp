#include "UIManager.h"
#include "CameraPanel.h"
#include "PreviewPanel.h"
#include "AnalysisPanel.h"
#include "app/Application.h"
#include "app/Logger.h"

#include <imgui.h>

namespace lenslab {

UIManager::UIManager(Application& app)
    : m_app(app)
{
}

UIManager::~UIManager() = default;

void UIManager::init()
{
    LOG_INFO("UIManager initializing...");

    m_cameraPanel = std::make_unique<CameraPanel>(m_app);
    m_previewPanel = std::make_unique<PreviewPanel>(m_app);
    m_analysisPanel = std::make_unique<AnalysisPanel>(m_app);

    LOG_INFO("UIManager initialized");
}

void UIManager::shutdown()
{
    LOG_INFO("UIManager shutting down...");

    m_cameraPanel.reset();
    m_previewPanel.reset();
    m_analysisPanel.reset();

    LOG_INFO("UIManager shutdown complete");
}

void UIManager::render()
{
    // Main menu bar
    renderMainMenuBar();

    // Enable docking
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

    // Render panels
    m_cameraPanel->render();
    m_previewPanel->render();
    m_analysisPanel->render();

    // Status bar
    renderStatusBar();

    // Debug windows
    if (m_showDemoWindow) {
        ImGui::ShowDemoWindow(&m_showDemoWindow);
    }

    if (m_showMetricsWindow) {
        ImGui::ShowMetricsWindow(&m_showMetricsWindow);
    }
}

void UIManager::renderMainMenuBar()
{
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Save Image", "Ctrl+S")) {
                // TODO: Save current frame
            }
            if (ImGui::MenuItem("Export Results", "Ctrl+E")) {
                // TODO: Export analysis results
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Settings")) {
                // TODO: Open settings dialog
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit", "Alt+F4")) {
                m_app.requestShutdown();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Camera Panel", nullptr, &m_cameraPanel->isVisible());
            ImGui::MenuItem("Preview Panel", nullptr, &m_previewPanel->isVisible());
            ImGui::MenuItem("Analysis Panel", nullptr, &m_analysisPanel->isVisible());
            ImGui::Separator();
            ImGui::MenuItem("ImGui Demo", nullptr, &m_showDemoWindow);
            ImGui::MenuItem("ImGui Metrics", nullptr, &m_showMetricsWindow);
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Camera")) {
            if (ImGui::MenuItem("Refresh List")) {
                m_app.getCameraManager().refreshDeviceList();
            }
            if (ImGui::MenuItem("Disconnect")) {
                m_app.getCameraManager().disconnectDevice();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Analysis")) {
            if (ImGui::MenuItem("Run MTF Analysis")) {
                // TODO: Trigger MTF analysis
            }
            if (ImGui::MenuItem("Reset ROIs")) {
                // TODO: Reset ROI positions
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("About")) {
                // TODO: Show about dialog
            }
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}

void UIManager::renderStatusBar()
{
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->Pos.x, viewport->Pos.y + viewport->Size.y - 25));
    ImGui::SetNextWindowSize(ImVec2(viewport->Size.x, 25));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoScrollWithMouse |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoBringToFrontOnFocus |
                             ImGuiWindowFlags_NoNav;

    if (ImGui::Begin("##StatusBar", nullptr, flags)) {
        auto& camMgr = m_app.getCameraManager();

        if (camMgr.isConnected()) {
            auto* device = camMgr.getActiveDevice();
            ImGui::Text("Connected: %s | %dx%d | %.1f FPS | Exp: %.1f ms",
                       device->getModel().c_str(),
                       device->getWidth(),
                       device->getHeight(),
                       device->getFrameRate(),
                       device->getExposure() / 1000.0);
        } else {
            ImGui::Text("No camera connected");
        }

        ImGui::SameLine(ImGui::GetWindowWidth() - 150);
        ImGui::Text("LensLab Desktop v1.0");
    }
    ImGui::End();
}

} // namespace lenslab
