# LensLab Desktop

A standalone USB3 Vision camera application with integrated focus analysis and MTF measurement capabilities.

## Features

- **Universal Camera Support**: Connect to any USB3 Vision camera (Basler, FLIR, IDS, Allied Vision)
- **Real-Time Focus Metrics**: Brenner, Tenengrad, and Modified Laplacian focus quality scores
- **MTF Analysis**: ISO 12233 compliant slant-edge MTF measurement
- **Professional UI**: ImGui-based interface with live preview and ROI management

## Building

### Prerequisites

**Ubuntu/Debian:**
```bash
sudo apt install build-essential cmake libopencv-dev libglfw3-dev libgl1-mesa-dev
```

**macOS:**
```bash
brew install cmake opencv glfw
```

### Compile

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Run

```bash
./lenslab-desktop
```

## Usage

1. **Connect Camera**: Select a camera from the device list and click Connect
2. **Start Acquisition**: Click Start to begin live preview
3. **Monitor Focus**: Watch real-time focus scores in the Analysis panel
4. **Run MTF Analysis**: Position ROIs over slant-edge targets and click Run Analysis

## Project Structure

```
lenslab-desktop/
├── src/
│   ├── app/          # Application lifecycle, config, logging
│   ├── camera/       # Camera discovery, connection, acquisition
│   ├── analysis/     # Focus metrics, MTF analysis
│   └── ui/           # ImGui panels and rendering
├── external/         # Third-party dependencies
├── test/             # Unit tests
└── docs/             # Documentation
```

## License

MIT License - See LICENSE file for details.

## Related Projects

- [LensLab Web](../Web%20Development/lenslab/) - Original web-based MTF analyzer
