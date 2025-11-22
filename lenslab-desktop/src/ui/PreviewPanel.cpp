#include "PreviewPanel.h"
#include "app/Application.h"
#include "camera/CameraManager.h"
#include "camera/CameraDevice.h"

#include <imgui.h>
#include <GLFW/glfw3.h>

namespace lenslab {

PreviewPanel::PreviewPanel(Application& app)
    : m_app(app)
{
}

PreviewPanel::~PreviewPanel()
{
    if (m_textureId != 0) {
        glDeleteTextures(1, &m_textureId);
    }
}

void PreviewPanel::render()
{
    if (!m_visible) return;

    ImGui::Begin("Preview", &m_visible);

    auto& camMgr = m_app.getCameraManager();

    if (camMgr.isConnected()) {
        auto* device = camMgr.getActiveDevice();

        if (device && device->hasNewFrame()) {
            updateTexture(device->getCurrentFrame());
            device->acknowledgeFrame();
        }

        renderImage();
        renderOverlay();
    } else {
        ImGui::Text("No camera connected");
        ImGui::Text("Connect a camera to see live preview");
    }

    ImGui::Separator();
    renderControls();

    ImGui::End();
}

void PreviewPanel::updateTexture(const cv::Mat& frame)
{
    if (frame.empty()) return;

    // Create texture if needed
    if (m_textureId == 0) {
        glGenTextures(1, &m_textureId);
    }

    // Check if size changed
    if (frame.cols != m_textureWidth || frame.rows != m_textureHeight) {
        m_textureWidth = frame.cols;
        m_textureHeight = frame.rows;

        glBindTexture(GL_TEXTURE_2D, m_textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    // Upload texture data
    glBindTexture(GL_TEXTURE_2D, m_textureId);

    GLenum format = (frame.channels() == 1) ? GL_RED : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0,
                 format, GL_UNSIGNED_BYTE, frame.data);
}

void PreviewPanel::renderImage()
{
    if (m_textureId == 0 || m_textureWidth == 0 || m_textureHeight == 0) {
        return;
    }

    // Calculate display size to fit in available space
    ImVec2 available = ImGui::GetContentRegionAvail();
    available.y -= 80;  // Reserve space for controls

    float aspectRatio = static_cast<float>(m_textureWidth) / m_textureHeight;
    float displayWidth = available.x * m_zoom;
    float displayHeight = displayWidth / aspectRatio;

    if (displayHeight > available.y * m_zoom) {
        displayHeight = available.y * m_zoom;
        displayWidth = displayHeight * aspectRatio;
    }

    // Center the image
    ImVec2 cursor = ImGui::GetCursorPos();
    ImGui::SetCursorPos(ImVec2(
        cursor.x + (available.x - displayWidth) * 0.5f,
        cursor.y + (available.y - displayHeight) * 0.5f
    ));

    // Render image
    ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(m_textureId)),
                 ImVec2(displayWidth, displayHeight));
}

void PreviewPanel::renderOverlay()
{
    // TODO: Draw ROI rectangles over the image
    // This will require tracking image position in screen coordinates
}

void PreviewPanel::renderControls()
{
    ImGui::Text("Display Options");

    ImGui::Checkbox("Show ROIs", &m_showROIs);
    ImGui::SameLine();
    ImGui::Checkbox("Show Grid", &m_showGrid);
    ImGui::SameLine();
    ImGui::Checkbox("Histogram", &m_showHistogram);

    ImGui::SliderFloat("Zoom", &m_zoom, 0.25f, 4.0f, "%.2fx");
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        m_zoom = 1.0f;
    }
}

} // namespace lenslab
