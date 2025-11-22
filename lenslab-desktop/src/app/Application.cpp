#include "Application.h"
#include "Config.h"
#include "Logger.h"
#include "camera/CameraManager.h"
#include "ui/UIManager.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

namespace lenslab {

Application* Application::s_instance = nullptr;

Application::Application()
{
    s_instance = this;
}

Application::~Application()
{
    shutdown();
    s_instance = nullptr;
}

bool Application::init(int argc, char* argv[])
{
    LOG_INFO("Initializing application...");

    // Initialize config
    m_config = std::make_unique<Config>();
    m_config->load("lenslab_config.json");

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--debug") {
            m_config->setDebugMode(true);
        }
        // Add more argument parsing as needed
    }

    // Initialize window and graphics
    if (!initWindow()) {
        LOG_ERROR("Failed to initialize window");
        return false;
    }

    // Initialize ImGui
    if (!initImGui()) {
        LOG_ERROR("Failed to initialize ImGui");
        return false;
    }

    // Initialize camera manager
    m_cameraManager = std::make_unique<CameraManager>();
    m_cameraManager->init();

    // Initialize UI manager
    m_uiManager = std::make_unique<UIManager>(*this);
    m_uiManager->init();

    LOG_INFO("Application initialized successfully");
    return true;
}

bool Application::initWindow()
{
    LOG_INFO("Initializing GLFW window...");

    if (!glfwInit()) {
        LOG_ERROR("Failed to initialize GLFW");
        return false;
    }

    // GL context hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create window
    m_window = glfwCreateWindow(
        m_windowWidth,
        m_windowHeight,
        m_windowTitle.c_str(),
        nullptr,
        nullptr
    );

    if (!m_window) {
        LOG_ERROR("Failed to create GLFW window");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(1); // Enable vsync

    LOG_INFO("GLFW window created: {}x{}", m_windowWidth, m_windowHeight);
    return true;
}

bool Application::initImGui()
{
    LOG_INFO("Initializing Dear ImGui...");

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Setup style
    ImGui::StyleColorsDark();

    // Customize style
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 4.0f;
    style.FrameRounding = 2.0f;
    style.GrabRounding = 2.0f;

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    LOG_INFO("Dear ImGui initialized");
    return true;
}

int Application::run()
{
    LOG_INFO("Entering main loop...");

    while (!glfwWindowShouldClose(m_window) && !m_shutdownRequested) {
        processFrame();
    }

    LOG_INFO("Exiting main loop");
    return 0;
}

void Application::processFrame()
{
    // Poll events
    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Update camera (grab frames)
    m_cameraManager->update();

    // Render UI
    m_uiManager->render();

    // Rendering
    ImGui::Render();

    int display_w, display_h;
    glfwGetFramebufferSize(m_window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(m_window);
}

void Application::requestShutdown()
{
    LOG_INFO("Shutdown requested");
    m_shutdownRequested = true;
}

void Application::shutdown()
{
    LOG_INFO("Shutting down...");

    // Cleanup UI
    if (m_uiManager) {
        m_uiManager->shutdown();
        m_uiManager.reset();
    }

    // Cleanup camera
    if (m_cameraManager) {
        m_cameraManager->shutdown();
        m_cameraManager.reset();
    }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // Cleanup GLFW
    if (m_window) {
        glfwDestroyWindow(m_window);
        m_window = nullptr;
    }
    glfwTerminate();

    // Save config
    if (m_config) {
        m_config->save("lenslab_config.json");
    }

    LOG_INFO("Shutdown complete");
}

} // namespace lenslab
