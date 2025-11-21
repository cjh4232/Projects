# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LensLab is an optical lens testing and MTF (Modulation Transfer Function) analysis system. It measures lens optical quality through slant-edge test patterns and focus metrics, combining C++ for performance-critical analysis, Python for test generation and validation, and WebAssembly for browser integration.

## Architecture

```
lenslab/
├── src/
│   ├── cpp/          # Core MTF analysis (C++17, OpenCV 4.8.1)
│   ├── python/       # Test generation, WebSocket server
│   └── web/          # JavaScript/WASM integration, Bubble.io plugin
├── scripts/          # Python validation and analysis scripts
├── test_patterns/    # Generated test images
├── working_targets/  # Validated test patterns with known blur levels
└── results/          # Analysis output and debug visualizations
```

**Key Entry Points:**
- Production executable: `mtf_analyzer_6_final` or `mtf_analyzer_production`
- Main source: `src/cpp/mtf_analyzer_6.cpp`
- WebSocket server: `src/python/websocket_server.py`

## Build Commands

### C++ Compilation
```bash
g++ -std=c++17 -O3 src/cpp/mtf_analyzer_6.cpp -o mtf_analyzer_6 \
    -I/opt/homebrew/include/opencv4 \
    -L/opt/homebrew/lib \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
```

### WebAssembly Build
```bash
./compile_mtf_wasm.sh
```

### Python Dependencies
Python 3.9+ with numpy, opencv-python, PIL, scipy, websockets

## Testing

No formal test framework. Validation is done through Python scripts comparing theoretical vs measured results:

```bash
# Generate test patterns with known blur
python scripts/generate_working_targets.py

# Analyze and validate results
python scripts/analyze_working_targets.py

# Run FWHM validation
python scripts/test_fwhm_accuracy.py
```

**Success Criteria:**
- FWHM accuracy within 5-15% of theoretical values
- <1% variance between similar ROIs
- Correct MTF trend (MTF50 > MTF20 > MTF10)

## Technical Conventions

**MTF Analysis:**
- Slant-edge angles: 8-15° (ISO 12233 compliant)
- Angle-adaptive sampling: 0.2x to 0.3x pixel spacing
- FFT-based MTF calculation from ESF/LSF derivatives

**Coordinate Systems:**
- Always translate edge coordinates from full image space to ROI local space before analysis
- Apply 0.5 distance scaling factor for sub-pixel FWHM accuracy

**Quality Assessment:**
- Multi-metric ROI scoring (edge strength, linearity, noise) before accepting results
- Generate debug visualizations for verification of each analysis step

## Key Output Metrics

- MTF50, MTF20, MTF10 (cycles/pixel and lp/mm)
- FWHM measurements
- Quality assessment scores
- Debug images showing edge detection, ESF, LSF, and MTF curves
