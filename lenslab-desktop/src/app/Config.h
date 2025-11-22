#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace lenslab {

/**
 * Application configuration management
 */
class Config
{
public:
    Config();

    bool load(const std::string& filename);
    bool save(const std::string& filename);

    // General settings
    bool isDebugMode() const { return m_debugMode; }
    void setDebugMode(bool debug) { m_debugMode = debug; }

    // Camera settings
    double getDefaultExposure() const { return m_defaultExposure; }
    void setDefaultExposure(double exposure) { m_defaultExposure = exposure; }

    double getDefaultGain() const { return m_defaultGain; }
    void setDefaultGain(double gain) { m_defaultGain = gain; }

    // Analysis settings
    int getFocusUpdateRate() const { return m_focusUpdateRate; }
    void setFocusUpdateRate(int hz) { m_focusUpdateRate = hz; }

    int getDefaultROISize() const { return m_defaultROISize; }
    void setDefaultROISize(int size) { m_defaultROISize = size; }

    // Export settings
    std::string getDefaultSavePath() const { return m_defaultSavePath; }
    void setDefaultSavePath(const std::string& path) { m_defaultSavePath = path; }

    std::string getDefaultImageFormat() const { return m_defaultImageFormat; }
    void setDefaultImageFormat(const std::string& format) { m_defaultImageFormat = format; }

private:
    // General
    bool m_debugMode = false;

    // Camera defaults
    double m_defaultExposure = 10000.0;  // microseconds
    double m_defaultGain = 0.0;          // dB

    // Analysis
    int m_focusUpdateRate = 15;          // Hz
    int m_defaultROISize = 100;          // pixels

    // Export
    std::string m_defaultSavePath = "./captures";
    std::string m_defaultImageFormat = "tiff";
};

} // namespace lenslab
