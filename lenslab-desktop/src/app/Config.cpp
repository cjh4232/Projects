#include "Config.h"
#include "Logger.h"
#include <fstream>

namespace lenslab {

Config::Config() = default;

bool Config::load(const std::string& filename)
{
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            LOG_INFO("Config file not found, using defaults: {}", filename);
            return false;
        }

        nlohmann::json j;
        file >> j;

        // Load values with defaults
        m_debugMode = j.value("debug_mode", m_debugMode);
        m_defaultExposure = j.value("default_exposure", m_defaultExposure);
        m_defaultGain = j.value("default_gain", m_defaultGain);
        m_focusUpdateRate = j.value("focus_update_rate", m_focusUpdateRate);
        m_defaultROISize = j.value("default_roi_size", m_defaultROISize);
        m_defaultSavePath = j.value("default_save_path", m_defaultSavePath);
        m_defaultImageFormat = j.value("default_image_format", m_defaultImageFormat);

        LOG_INFO("Config loaded from: {}", filename);
        return true;

    } catch (const std::exception& e) {
        LOG_WARNING("Failed to load config: {}", e.what());
        return false;
    }
}

bool Config::save(const std::string& filename)
{
    try {
        nlohmann::json j;
        j["debug_mode"] = m_debugMode;
        j["default_exposure"] = m_defaultExposure;
        j["default_gain"] = m_defaultGain;
        j["focus_update_rate"] = m_focusUpdateRate;
        j["default_roi_size"] = m_defaultROISize;
        j["default_save_path"] = m_defaultSavePath;
        j["default_image_format"] = m_defaultImageFormat;

        std::ofstream file(filename);
        if (!file.is_open()) {
            LOG_ERROR("Cannot open config file for writing: {}", filename);
            return false;
        }

        file << j.dump(4);
        LOG_INFO("Config saved to: {}", filename);
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to save config: {}", e.what());
        return false;
    }
}

} // namespace lenslab
