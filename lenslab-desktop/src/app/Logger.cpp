#include "Logger.h"
#include <iostream>

namespace lenslab {

std::ofstream Logger::s_file;
std::mutex Logger::s_mutex;
LogLevel Logger::s_minLevel = LogLevel::Info;
bool Logger::s_initialized = false;

void Logger::init(const std::string& filename)
{
    std::lock_guard<std::mutex> lock(s_mutex);
    if (s_initialized) return;

    s_file.open(filename, std::ios::out | std::ios::app);
    s_initialized = true;

    log(LogLevel::Info, "Logger initialized");
}

void Logger::shutdown()
{
    std::lock_guard<std::mutex> lock(s_mutex);
    if (s_file.is_open()) {
        s_file.close();
    }
    s_initialized = false;
}

void Logger::log(LogLevel level, const std::string& message)
{
    if (level < s_minLevel) return;

    std::lock_guard<std::mutex> lock(s_mutex);

    std::string timestamp = getCurrentTimestamp();
    std::string levelStr = levelToString(level);
    std::string logLine = timestamp + " [" + levelStr + "] " + message;

    // Output to console
    std::cout << logLine << std::endl;

    // Output to file
    if (s_file.is_open()) {
        s_file << logLine << std::endl;
        s_file.flush();
    }
}

std::string Logger::levelToString(LogLevel level)
{
    switch (level) {
        case LogLevel::Debug:   return "DEBUG";
        case LogLevel::Info:    return "INFO";
        case LogLevel::Warning: return "WARN";
        case LogLevel::Error:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

std::string Logger::getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

} // namespace lenslab
