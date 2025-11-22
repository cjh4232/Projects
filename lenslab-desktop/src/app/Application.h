#pragma once

#include <memory>
#include <string>

// Forward declarations
struct GLFWwindow;

namespace lenslab {

// Forward declarations
class UIManager;
class CameraManager;
class Config;

/**
 * Main application class - handles lifecycle and coordination
 */
class Application
{
public:
    Application();
    ~Application();

    // Non-copyable
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    /**
     * Initialize the application
     * @param argc Command line argument count
     * @param argv Command line arguments
     * @return true if initialization successful
     */
    bool init(int argc, char* argv[]);

    /**
     * Run the main application loop
     * @return Exit code
     */
    int run();

    /**
     * Request application shutdown
     */
    void requestShutdown();

    /**
     * Check if shutdown has been requested
     */
    bool isShutdownRequested() const { return m_shutdownRequested; }

    // Accessors
    Config& getConfig() { return *m_config; }
    CameraManager& getCameraManager() { return *m_cameraManager; }
    GLFWwindow* getWindow() { return m_window; }

    // Singleton access (for ImGui callbacks)
    static Application* getInstance() { return s_instance; }

private:
    bool initWindow();
    bool initImGui();
    void shutdown();
    void processFrame();

    // Window
    GLFWwindow* m_window = nullptr;
    int m_windowWidth = 1600;
    int m_windowHeight = 900;
    std::string m_windowTitle = "LensLab Desktop";

    // Subsystems
    std::unique_ptr<Config> m_config;
    std::unique_ptr<UIManager> m_uiManager;
    std::unique_ptr<CameraManager> m_cameraManager;

    // State
    bool m_shutdownRequested = false;

    // Singleton instance
    static Application* s_instance;
};

} // namespace lenslab
