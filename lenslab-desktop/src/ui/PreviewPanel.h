#pragma once

#include <opencv2/core.hpp>

namespace lenslab {

class Application;

/**
 * Live camera preview with ROI overlay
 */
class PreviewPanel
{
public:
    explicit PreviewPanel(Application& app);
    ~PreviewPanel();

    void render();

    bool& isVisible() { return m_visible; }

private:
    void updateTexture(const cv::Mat& frame);
    void renderImage();
    void renderOverlay();
    void renderControls();

    Application& m_app;
    bool m_visible = true;

    // OpenGL texture
    unsigned int m_textureId = 0;
    int m_textureWidth = 0;
    int m_textureHeight = 0;

    // Display options
    bool m_showROIs = true;
    bool m_showGrid = false;
    bool m_showHistogram = false;
    float m_zoom = 1.0f;
};

} // namespace lenslab
