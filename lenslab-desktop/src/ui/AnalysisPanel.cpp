#include "AnalysisPanel.h"
#include "app/Application.h"

#include <imgui.h>
#include <implot.h>

namespace lenslab {

AnalysisPanel::AnalysisPanel(Application& app)
    : m_app(app)
{
}

void AnalysisPanel::render()
{
    if (!m_visible) return;

    ImGui::Begin("Analysis", &m_visible);

    if (ImGui::BeginTabBar("AnalysisTabs")) {
        if (ImGui::BeginTabItem("Focus Metrics")) {
            renderFocusMetrics();
            ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("MTF Analysis")) {
            renderMTFResults();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    ImGui::End();
}

void AnalysisPanel::renderFocusMetrics()
{
    ImGui::Text("Live Focus Quality");

    ImGui::Checkbox("Auto Update", &m_autoUpdateFocus);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::SliderInt("Rate (Hz)", &m_focusUpdateRate, 1, 30);

    // Metric selection
    const char* metrics[] = { "Brenner", "Tenengrad", "Mod. Laplacian", "Combined" };
    ImGui::SetNextItemWidth(150);
    ImGui::Combo("Display Metric", &m_selectedMetric, metrics, 4);

    ImGui::Separator();

    // ROI scores display
    ImGui::BeginTable("FocusScores", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg);
    ImGui::TableSetupColumn("ROI", ImGuiTableColumnFlags_WidthFixed, 60);
    ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 80);
    ImGui::TableSetupColumn("Bar", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();

    const char* roiNames[] = { "Center", "UL", "UR", "LL", "LR" };
    for (int i = 0; i < 5; i++) {
        ImGui::TableNextRow();

        double score = 0;
        switch (m_selectedMetric) {
            case 0: score = m_focusResults[i].brenner; break;
            case 1: score = m_focusResults[i].tenengrad; break;
            case 2: score = m_focusResults[i].modifiedLaplacian; break;
            case 3: score = m_focusResults[i].combined; break;
        }

        // ROI name
        ImGui::TableNextColumn();
        ImGui::Text("%s", roiNames[i]);

        // Score value
        ImGui::TableNextColumn();
        ImVec4 color;
        if (score >= 70) color = ImVec4(0.2f, 0.8f, 0.2f, 1.0f);      // Green
        else if (score >= 40) color = ImVec4(0.8f, 0.8f, 0.2f, 1.0f); // Yellow
        else color = ImVec4(0.8f, 0.2f, 0.2f, 1.0f);                   // Red

        ImGui::TextColored(color, "%.1f", score);

        // Progress bar
        ImGui::TableNextColumn();
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, color);
        ImGui::ProgressBar(static_cast<float>(score / 100.0), ImVec2(-1, 0), "");
        ImGui::PopStyleColor();
    }

    ImGui::EndTable();

    ImGui::Separator();
    ImGui::Text("Tip: Higher scores indicate better focus");
}

void AnalysisPanel::renderMTFResults()
{
    ImGui::Text("MTF Analysis");

    if (ImGui::Button("Run Analysis", ImVec2(120, 0))) {
        // TODO: Trigger MTF analysis on current frame
    }

    ImGui::SameLine();
    if (ImGui::Button("Export", ImVec2(80, 0))) {
        // TODO: Export results
    }

    ImGui::Separator();

    if (m_hasValidMTF) {
        // Results table
        ImGui::BeginTable("MTFResults", 2, ImGuiTableFlags_Borders);
        ImGui::TableSetupColumn("Metric", ImGuiTableColumnFlags_WidthFixed, 120);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        auto addRow = [](const char* label, const char* format, double value) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%s", label);
            ImGui::TableNextColumn();
            ImGui::Text(format, value);
        };

        addRow("MTF50", "%.4f cyc/px", m_mtfResult.mtf50);
        addRow("MTF20", "%.4f cyc/px", m_mtfResult.mtf20);
        addRow("MTF10", "%.4f cyc/px", m_mtfResult.mtf10);
        addRow("FWHM", "%.2f pixels", m_mtfResult.fwhm);
        addRow("Quality", "%.1f / 100", m_mtfResult.quality.overallScore);

        ImGui::EndTable();

        ImGui::Separator();

        // MTF curve plot
        renderMTFPlot();
    } else {
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f),
                          "No MTF data available.\n"
                          "Ensure a slant-edge target is visible\n"
                          "and click 'Run Analysis'.");
    }
}

void AnalysisPanel::renderMTFPlot()
{
    if (!m_hasValidMTF || m_mtfResult.frequencies.empty()) {
        return;
    }

    if (ImPlot::BeginPlot("MTF Curve", ImVec2(-1, 250))) {
        ImPlot::SetupAxes("Frequency (cyc/px)", "MTF");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, 0.5);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1.1);

        // Plot MTF curve
        ImPlot::PlotLine("MTF",
                        m_mtfResult.frequencies.data(),
                        m_mtfResult.mtfValues.data(),
                        static_cast<int>(m_mtfResult.frequencies.size()));

        // Reference lines
        double y50[] = {0.5, 0.5};
        double y20[] = {0.2, 0.2};
        double xRange[] = {0.0, 0.5};

        ImPlot::PushStyleColor(ImPlotCol_Line, ImVec4(0.5f, 0.5f, 0.5f, 0.5f));
        ImPlot::PlotLine("50%", xRange, y50, 2);
        ImPlot::PlotLine("20%", xRange, y20, 2);
        ImPlot::PopStyleColor();

        // Mark MTF50 point
        if (m_mtfResult.mtf50 > 0) {
            double x[] = {m_mtfResult.mtf50};
            double y[] = {0.5};
            ImPlot::PushStyleColor(ImPlotCol_MarkerFill, ImVec4(0.2f, 0.8f, 0.2f, 1.0f));
            ImPlot::PlotScatter("MTF50", x, y, 1);
            ImPlot::PopStyleColor();
        }

        ImPlot::EndPlot();
    }
}

} // namespace lenslab
