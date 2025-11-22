#pragma once

#include "analysis/FocusMetrics.h"
#include "analysis/MTFAnalyzer.h"
#include <vector>

namespace lenslab {

class Application;

/**
 * Focus metrics and MTF analysis results panel
 */
class AnalysisPanel
{
public:
    explicit AnalysisPanel(Application& app);

    void render();

    bool& isVisible() { return m_visible; }

private:
    void renderFocusMetrics();
    void renderMTFResults();
    void renderMTFPlot();

    Application& m_app;
    bool m_visible = true;

    // Focus metrics state
    FocusResult m_focusResults[5];  // C, UL, UR, LL, LR
    int m_focusUpdateRate = 15;      // Hz

    // MTF results
    MTFResult m_mtfResult;
    bool m_hasValidMTF = false;

    // UI state
    bool m_autoUpdateFocus = true;
    int m_selectedMetric = 3;  // Combined
};

} // namespace lenslab
