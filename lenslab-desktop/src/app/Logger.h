#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace lenslab {

enum class LogLevel {
    Debug,
    Info,
    Warning,
    Error
};

/**
 * Simple thread-safe logger
 */
class Logger
{
public:
    static void init(const std::string& filename);
    static void shutdown();

    static void log(LogLevel level, const std::string& message);

    template<typename... Args>
    static void log(LogLevel level, const std::string& format, Args&&... args)
    {
        // Simple format replacement (replaces {} with args)
        std::string message = formatString(format, std::forward<Args>(args)...);
        log(level, message);
    }

    static void setMinLevel(LogLevel level) { s_minLevel = level; }

private:
    static std::string levelToString(LogLevel level);
    static std::string getCurrentTimestamp();

    template<typename T>
    static std::string toString(T&& value)
    {
        std::ostringstream oss;
        oss << value;
        return oss.str();
    }

    template<typename T, typename... Args>
    static std::string formatString(const std::string& format, T&& first, Args&&... rest)
    {
        size_t pos = format.find("{}");
        if (pos == std::string::npos) {
            return format;
        }
        std::string result = format.substr(0, pos) + toString(std::forward<T>(first));
        return formatString(result + format.substr(pos + 2), std::forward<Args>(rest)...);
    }

    static std::string formatString(const std::string& format)
    {
        return format;
    }

    static std::ofstream s_file;
    static std::mutex s_mutex;
    static LogLevel s_minLevel;
    static bool s_initialized;
};

// Convenience macros
#define LOG_DEBUG(...)   lenslab::Logger::log(lenslab::LogLevel::Debug, __VA_ARGS__)
#define LOG_INFO(...)    lenslab::Logger::log(lenslab::LogLevel::Info, __VA_ARGS__)
#define LOG_WARNING(...) lenslab::Logger::log(lenslab::LogLevel::Warning, __VA_ARGS__)
#define LOG_ERROR(...)   lenslab::Logger::log(lenslab::LogLevel::Error, __VA_ARGS__)

} // namespace lenslab
