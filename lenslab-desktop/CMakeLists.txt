cmake_minimum_required(VERSION 3.16)
project(LensLabDesktop VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(LENSLAB_BUILD_TESTS "Build test executables" ON)
option(LENSLAB_USE_SYSTEM_OPENCV "Use system OpenCV instead of bundled" ON)
option(LENSLAB_USE_SYSTEM_GLFW "Use system GLFW instead of bundled" OFF)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

# OpenCV
if(LENSLAB_USE_SYSTEM_OPENCV)
    find_package(OpenCV 4.0 REQUIRED COMPONENTS core imgproc imgcodecs)
else()
    # TODO: Add FetchContent for OpenCV
    message(FATAL_ERROR "Bundled OpenCV not yet supported. Use -DLENSLAB_USE_SYSTEM_OPENCV=ON")
endif()

# OpenGL
find_package(OpenGL REQUIRED)

# GLFW
if(LENSLAB_USE_SYSTEM_GLFW)
    find_package(glfw3 3.3 REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.3.8
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
endif()

# Dear ImGui (fetched)
include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.1
)
FetchContent_MakeAvailable(imgui)

# ImPlot (fetched)
FetchContent_Declare(
    implot
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG v0.16
)
FetchContent_MakeAvailable(implot)

# nlohmann/json (fetched)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# -----------------------------------------------------------------------------
# ImGui library target
# -----------------------------------------------------------------------------
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL)

# -----------------------------------------------------------------------------
# ImPlot library target
# -----------------------------------------------------------------------------
add_library(implot_lib STATIC
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)
target_include_directories(implot_lib PUBLIC ${implot_SOURCE_DIR})
target_link_libraries(implot_lib PUBLIC imgui_lib)

# -----------------------------------------------------------------------------
# Main application sources
# -----------------------------------------------------------------------------
set(LENSLAB_SOURCES
    src/main.cpp
    src/app/Application.cpp
    src/app/Config.cpp
    src/app/Logger.cpp
    src/camera/CameraManager.cpp
    src/camera/CameraDevice.cpp
    src/camera/FrameBuffer.cpp
    src/analysis/FocusMetrics.cpp
    src/analysis/MTFAnalyzer.cpp
    src/analysis/ROIManager.cpp
    src/ui/UIManager.cpp
    src/ui/CameraPanel.cpp
    src/ui/PreviewPanel.cpp
    src/ui/AnalysisPanel.cpp
)

set(LENSLAB_HEADERS
    src/app/Application.h
    src/app/Config.h
    src/app/Logger.h
    src/camera/CameraManager.h
    src/camera/CameraDevice.h
    src/camera/FrameBuffer.h
    src/analysis/FocusMetrics.h
    src/analysis/MTFAnalyzer.h
    src/analysis/ROIManager.h
    src/ui/UIManager.h
    src/ui/CameraPanel.h
    src/ui/PreviewPanel.h
    src/ui/AnalysisPanel.h
)

# -----------------------------------------------------------------------------
# Main executable
# -----------------------------------------------------------------------------
add_executable(lenslab-desktop ${LENSLAB_SOURCES} ${LENSLAB_HEADERS})

target_include_directories(lenslab-desktop PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(lenslab-desktop PRIVATE
    imgui_lib
    implot_lib
    nlohmann_json::nlohmann_json
    ${OpenCV_LIBS}
    glfw
    OpenGL::GL
)

# Platform-specific libraries
if(UNIX AND NOT APPLE)
    target_link_libraries(lenslab-desktop PRIVATE dl pthread)
endif()

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
if(LENSLAB_BUILD_TESTS)
    enable_testing()

    # Test MTF Analyzer
    add_executable(test_mtf_analyzer
        test/test_mtf_analyzer.cpp
        src/analysis/MTFAnalyzer.cpp
        src/analysis/FocusMetrics.cpp
    )
    target_include_directories(test_mtf_analyzer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(test_mtf_analyzer PRIVATE ${OpenCV_LIBS})
    add_test(NAME MTFAnalyzerTest COMMAND test_mtf_analyzer)

    # Test Focus Metrics
    add_executable(test_focus_metrics
        test/test_focus_metrics.cpp
        src/analysis/FocusMetrics.cpp
    )
    target_include_directories(test_focus_metrics PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(test_focus_metrics PRIVATE ${OpenCV_LIBS})
    add_test(NAME FocusMetricsTest COMMAND test_focus_metrics)
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
install(TARGETS lenslab-desktop
    RUNTIME DESTINATION bin
)

install(DIRECTORY resources/
    DESTINATION share/lenslab-desktop
)

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "LensLab Desktop Configuration Summary")
message(STATUS "======================================")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV:         ${OpenCV_VERSION}")
message(STATUS "  Build tests:    ${LENSLAB_BUILD_TESTS}")
message(STATUS "")
