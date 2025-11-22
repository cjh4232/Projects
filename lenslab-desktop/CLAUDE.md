# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

LensLab Desktop is a standalone USB3 Vision camera application with integrated focus analysis and MTF measurement capabilities. It provides real-time focus metrics for active adjustment and full MTF analysis for optical quality verification.

## Technology Stack

- **Language:** C++17
- **UI Framework:** Dear ImGui + ImPlot
- **Rendering:** OpenGL 3.3 via GLFW
- **Image Processing:** OpenCV 4.x
- **Camera Interface:** GenICam/GenTL (planned)
- **Build System:** CMake

## Architecture

```
src/
├── app/          # Application lifecycle, config, logging
├── camera/       # Camera discovery, connection, acquisition
├── analysis/     # Focus metrics, MTF analysis, ROI management
├── ui/           # ImGui panels and rendering
└── utils/        # Helpers (image I/O, timers, threading)
```

## Build Commands

### Prerequisites

**Ubuntu/Debian:**
```bash
sudo apt install build-essential cmake libopencv-dev libglfw3-dev libgl1-mesa-dev
```

**macOS:**
```bash
brew install cmake opencv glfw
```

**Windows:**
- Install Visual Studio 2019+
- Install vcpkg and: `vcpkg install opencv4 glfw3`

### Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Running Tests

```bash
cd build
ctest --output-on-failure
```

## Key Components

### CameraManager (src/camera/)
- Discovers USB3 Vision cameras via GenTL
- Manages camera lifecycle (connect/disconnect)
- Handles frame acquisition

### AnalysisEngine (src/analysis/)
- **FocusMetrics:** Brenner, Tenengrad, Modified Laplacian
- **MTFAnalyzer:** ISO 12233 slant-edge MTF calculation (ported from LensLab)
- **ROIManager:** Region of interest handling

### UIManager (src/ui/)
- ImGui initialization and main loop
- Panel management (Camera, Preview, Analysis)

## Code Conventions

- Use `snake_case` for file names
- Use `PascalCase` for class names
- Use `camelCase` for method/function names
- Use `m_` prefix for member variables
- Header guards: `#pragma once`
- Prefer `std::unique_ptr` and `std::shared_ptr` over raw pointers

## Integration with LensLab

The MTF analyzer is ported from `/Web Development/lenslab/src/cpp/mtf_analyzer_6.cpp`.
Key changes from original:
1. Removed OpenCV highgui dependencies (display via ImGui)
2. Added streaming interface for live frame analysis
3. Extracted as reusable library class

## Testing

Validation targets (from LensLab):
- FWHM accuracy: < 20% error
- ROI consistency: < 1% variance
- MTF trend: MTF50 > MTF20 > MTF10

Test images are in `test/test_images/` and should include slant-edge targets at various blur levels.

## Performance Targets

- Live focus metrics: 10-30 Hz update rate
- Sensor-to-screen latency: < 50ms
- Frame rate: > 95% of camera capability
