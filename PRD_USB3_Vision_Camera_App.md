# Product Requirements Document (PRD)
## Standalone USB3 Vision Camera Application with Focus Analysis

**Version:** 2.0
**Date:** November 22, 2025
**Project Code Name:** LensLab Desktop
**Status:** Approved - Ready for Development

---

## 1. Executive Summary

This document outlines the requirements for a standalone desktop application that enables users to connect, control, and capture images from any USB3 Vision-compliant camera, with integrated real-time focus quality assessment and MTF (Modulation Transfer Function) analysis.

**Problem Statement:** Currently, using industrial USB3 Vision cameras requires:
- Installing vendor-specific SDKs and drivers for each camera brand
- No unified tool for real-time focus quality assessment
- Separate workflows for camera control and optical analysis
- Complex setup processes that create friction for users

**Solution:** A unified, high-performance application that:
- Handles all USB3 Vision cameras through a single interface
- Provides real-time focus metrics for live adjustment
- Includes full MTF analysis for optical quality verification
- Built on proven, production-ready analysis code (LensLab MTF Analyzer)

---

## 2. Product Overview

### 2.1 Product Vision
Create a professional-grade optical testing tool that combines universal camera support with real-time focus analysis, enabling users to both adjust and verify optical system performance in a single workflow.

### 2.2 Target Users
- **Optical Engineers:** Aligning camera sensors to optical systems
- **Machine Vision Integrators:** Setting up and verifying camera focus
- **Quality Control Teams:** Validating optical system performance
- **R&D / Prototyping Teams:** Evaluating lens/camera combinations
- **Production Line Operators:** Verifying focus during assembly

### 2.3 Key Differentiators
- **Universal Camera Support:** Works with Basler, FLIR, IDS, Allied Vision, and other USB3 Vision cameras
- **Integrated Focus Analysis:** Real-time focus metrics + full MTF analysis in one tool
- **Production-Proven Core:** Built on validated MTF analyzer (<1% measurement variance)
- **High Performance:** Native C++ with GPU-accelerated rendering
- **Zero Dependencies:** Single executable, no driver installation required

---

## 3. Goals and Objectives

### 3.1 Primary Goals
1. **Universal Camera Access:** Connect to any USB3 Vision camera without vendor software
2. **Real-Time Focus Feedback:** Enable live focus adjustment with instant visual feedback
3. **Accurate MTF Analysis:** Provide ISO 12233-compliant optical quality measurements
4. **Professional Performance:** Handle high-speed cameras (100+ FPS) without frame drops

### 3.2 Success Metrics
| Metric | Target |
|--------|--------|
| Time to first image | < 60 seconds |
| Camera discovery rate | > 98% of USB3 Vision cameras |
| Focus metric update rate | 10-30 Hz (user configurable) |
| MTF measurement accuracy | < 15% error vs theoretical |
| MTF measurement variance | < 1% between similar ROIs |
| Frame rate achieved | > 95% of camera specification |

### 3.3 Scope for v1.0
**In Scope:**
- USB3 Vision camera connection and control
- Live preview with ROI overlay
- Real-time focus metrics (Brenner, Tenengrad, Modified Laplacian)
- Full MTF analysis (MTF50, MTF20, MTF10, FWHM)
- Image capture and export
- Camera parameter control

**Out of Scope (Future Versions):**
- GigE Vision camera support
- Multi-camera synchronization
- Video recording with encoding
- Automated focus optimization (closed-loop control)
- Camera calibration tools

---

## 4. Core Features

### 4.1 Camera Discovery and Connection (P0 - Critical)

**Description:** Automatic detection and connection of USB3 Vision cameras via GenICam/GenTL.

**Requirements:**
- Enumerate all USB3 Vision cameras on system startup
- Display: Vendor, Model, Serial Number, Firmware Version
- Hot-plug detection (cameras connected after launch)
- Support multiple simultaneous cameras (list with selection)
- Graceful error handling for busy/unavailable cameras

**Technical Implementation:**
- GenICam GenTL producer discovery
- Support vendor GenTL files: Basler, FLIR/Teledyne, IDS, Allied Vision
- Fallback to generic USB3 Vision producer if available

### 4.2 Live Camera Preview (P0 - Critical)

**Description:** Real-time video display with ROI overlay for focus analysis.

**Requirements:**
- Display live stream at full camera resolution
- GPU-accelerated rendering via OpenGL
- Adjustable display scaling (fit, 50%, 100%, 200%)
- Real-time frame rate display
- ROI overlay with drag-to-position capability
- Crosshair and grid overlays (toggleable)
- Histogram overlay (optional)

**Performance Targets:**
- < 50ms sensor-to-screen latency
- Support up to 500 FPS acquisition
- Smooth rendering at 60 Hz display refresh

### 4.3 Focus Quality Analysis - Live Mode (P0 - Critical)

**Description:** Real-time focus metrics for active focus adjustment.

**Metrics Provided:**
| Metric | Description | Use Case |
|--------|-------------|----------|
| Brenner Gradient | Sum of squared horizontal/vertical gradients | General focus |
| Tenengrad | Sobel gradient magnitude | Edge sharpness |
| Modified Laplacian | Second derivative response | Fine focus |
| Combined Score | Weighted average (0-100 scale) | Quick assessment |

**Requirements:**
- Update rate: 10-30 Hz (user configurable)
- Per-ROI scoring (up to 5 ROIs: Center, UL, UR, LL, LR)
- Visual feedback: Color-coded quality indicator (Red/Yellow/Green)
- Trend indicator: Arrow showing improvement/degradation direction
- Audio feedback option: Tone pitch proportional to focus score

**UI Display:**
```
┌─────────────────────────────────────────┐
│  Live Preview                           │
│  ┌─────┐                    ┌─────┐    │
│  │ UL  │                    │ UR  │    │
│  │ 78  │                    │ 82  │    │
│  └─────┘                    └─────┘    │
│              ┌─────┐                    │
│              │  C  │                    │
│              │ 85  │                    │
│              └─────┘                    │
│  ┌─────┐                    ┌─────┐    │
│  │ LL  │                    │ LR  │    │
│  │ 76  │                    │ 79  │    │
│  └─────┘                    └─────┘    │
└─────────────────────────────────────────┘
```

### 4.4 MTF Analysis - Capture Mode (P0 - Critical)

**Description:** Full MTF analysis on captured frames for optical quality verification.

**Analysis Pipeline (from LensLab MTF Analyzer):**
1. **Edge Detection:** Identify slant edges in ROI (8-15° optimal angle)
2. **ESF Extraction:** Sample Edge Spread Function perpendicular to edge
3. **LSF Computation:** Derivative of ESF = Line Spread Function
4. **FFT Analysis:** Compute MTF curve from LSF
5. **Metric Extraction:** Calculate MTF50, MTF20, MTF10, FWHM

**Metrics Provided:**
| Metric | Description | Units |
|--------|-------------|-------|
| MTF50 | Frequency at 50% contrast | cycles/pixel, lp/mm |
| MTF20 | Frequency at 20% contrast | cycles/pixel, lp/mm |
| MTF10 | Frequency at 10% contrast | cycles/pixel, lp/mm |
| FWHM | Full Width at Half Maximum | pixels |
| Quality Score | ROI quality assessment | 0-100 |

**Requirements:**
- Support slant-edge targets (ISO 12233 compliant)
- Support crosshair patterns (sagittal/tangential analysis)
- Quality filtering: Reject poor-quality ROIs with explanation
- Debug visualization: ESF, LSF, and MTF curve plots
- Export: PNG images + JSON/CSV data

**UI Display:**
```
┌─────────────────────────────────────────────────────┐
│  MTF Analysis Results                               │
├─────────────────────────────────────────────────────┤
│  MTF50: 0.142 cyc/px  (71.0 lp/mm @ 5μm pixel)     │
│  MTF20: 0.089 cyc/px  (44.5 lp/mm)                 │
│  MTF10: 0.052 cyc/px  (26.0 lp/mm)                 │
│  FWHM:  3.24 pixels                                 │
│  Quality: 92/100 (Excellent)                        │
├─────────────────────────────────────────────────────┤
│  [MTF Curve Graph]                                  │
│  1.0 ─┬─────────────────────────                   │
│       │ ╲                                           │
│  0.5 ─┼───╲─────────────────────  ← MTF50          │
│       │     ╲                                       │
│  0.2 ─┼───────╲─────────────────  ← MTF20          │
│       │         ╲                                   │
│  0.0 ─┴───────────────────────────                 │
│       0    0.1   0.2   0.3   0.4   0.5 cyc/px      │
└─────────────────────────────────────────────────────┘
```

### 4.5 Camera Parameter Control (P0 - Critical)

**Description:** Full access to camera settings via GenICam interface.

**Core Parameters:**
- **Exposure:** Time (μs/ms), Auto-exposure enable, AE ROI
- **Gain:** Value (dB), Auto-gain enable
- **Frame Rate:** Target FPS, enable/disable limit
- **Pixel Format:** Mono8, Mono12, RGB8, BayerRG8, etc.
- **ROI/AOI:** Offset X/Y, Width, Height
- **Trigger:** Mode (Freerun/Software/Hardware), Source, Polarity

**Advanced Parameters (collapsible):**
- Black Level, Gamma, White Balance
- Binning, Decimation
- LUT selection
- Test pattern generation

**UI Requirements:**
- Slider + numeric input for continuous values
- Immediate preview update on change
- Reset to defaults button
- Save/Load parameter presets

### 4.6 Image Capture and Export (P1 - Important)

**Description:** Capture and save images with analysis results.

**Capture Modes:**
- Single frame capture
- Burst capture (N frames at max rate)
- Timed capture (every N seconds)

**Export Formats:**
- Images: TIFF (16-bit), PNG, JPEG, BMP
- Analysis: JSON, CSV
- Reports: PNG with embedded metrics

**File Naming:**
- Template system: `{date}_{time}_{camera}_{sequence}.{ext}`
- Auto-increment sequence numbers
- Configurable save location

---

## 5. Technology Stack (Finalized)

### 5.1 Core Technologies

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Language** | C++17 | Performance, direct hardware access, existing MTF code |
| **UI Framework** | Dear ImGui + ImPlot | Fast iteration, real-time capable, familiar |
| **Rendering** | OpenGL 3.3 | Cross-platform GPU acceleration |
| **Camera SDK** | GenICam + GenTL | Universal USB3 Vision support |
| **Image Processing** | OpenCV 4.x | Industry standard, comprehensive |
| **Build System** | CMake | Cross-platform builds |

### 5.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        APPLICATION                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    UI LAYER (ImGui)                       │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │   │
│  │  │ Camera List │ │ Live View   │ │ Analysis Results    │ │   │
│  │  │ & Controls  │ │ + ROI       │ │ + MTF Curves        │ │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌───────────────────────────▼──────────────────────────────┐   │
│  │                   APPLICATION CORE                        │   │
│  │  ┌─────────────────┐  ┌─────────────────────────────────┐│   │
│  │  │ CameraManager   │  │ AnalysisEngine                  ││   │
│  │  │ - Discovery     │  │ - FocusMetrics (Brenner, etc)   ││   │
│  │  │ - Connection    │  │ - MTFAnalyzer (from LensLab)    ││   │
│  │  │ - Parameters    │  │ - ROI Management                ││   │
│  │  │ - Acquisition   │  │ - Quality Assessment            ││   │
│  │  └─────────────────┘  └─────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌───────────────────────────▼──────────────────────────────┐   │
│  │              HARDWARE ABSTRACTION LAYER                   │   │
│  │  ┌─────────────────────────────────────────────────────┐ │   │
│  │  │              GenICam / GenTL Interface               │ │   │
│  │  └─────────────────────────────────────────────────────┘ │   │
│  │       │              │              │              │      │   │
│  │  ┌────▼────┐   ┌────▼────┐   ┌────▼────┐   ┌────▼────┐  │   │
│  │  │ Basler  │   │  FLIR   │   │   IDS   │   │ Allied  │  │   │
│  │  │  .cti   │   │  .cti   │   │  .cti   │   │  .cti   │  │   │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
└──────────────────────────────▼───────────────────────────────────┘
                        USB3 Vision Cameras
```

### 5.3 Key Dependencies

| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| Dear ImGui | 1.90+ | UI Framework | MIT |
| ImPlot | 0.16+ | Plotting/Graphs | MIT |
| OpenCV | 4.8+ | Image Processing | Apache 2.0 |
| GLFW | 3.3+ | Window/OpenGL | Zlib |
| GenICam Reference | 3.4+ | Camera Interface | GenICam License |
| stb_image | Latest | Image I/O | Public Domain |
| nlohmann/json | 3.11+ | JSON Export | MIT |

### 5.4 Integration with LensLab Codebase

**Files to Integrate:**
| Source File | Purpose | Integration |
|-------------|---------|-------------|
| `mtf_analyzer_6.cpp` | Core MTF analysis | Extract as library |
| `pixel_lab_brenner.cpp` | Focus metrics | Port from WASM to native |
| Validation scripts | Testing | Adapt for native builds |

**Refactoring Required:**
1. Extract `MTFAnalyzer` class into standalone header/source
2. Remove OpenCV `highgui` dependencies (use ImGui for display)
3. Add streaming interface for live frame analysis
4. Create C++ focus metrics (currently Emscripten-specific)

---

## 6. Development Phases

### Phase 1: Foundation (Weeks 1-2)
**Goal:** Basic camera connection and live preview

**Deliverables:**
- [ ] CMake project structure
- [ ] ImGui + OpenGL window with basic layout
- [ ] GenICam integration (camera discovery)
- [ ] Single camera connection
- [ ] Live preview (grayscale, basic)
- [ ] Frame rate display

**Success Criteria:**
- Can list connected USB3 Vision cameras
- Can display live video from a camera
- Achieves target frame rate

### Phase 2: Core Analysis (Weeks 3-4)
**Goal:** Integrate MTF analyzer and focus metrics

**Deliverables:**
- [ ] Port LensLab MTF analyzer to library
- [ ] Implement native focus metrics (Brenner, Tenengrad, ML)
- [ ] ROI overlay system (draggable regions)
- [ ] Live focus score display
- [ ] Single-frame MTF analysis
- [ ] Basic results display

**Success Criteria:**
- Real-time focus metrics updating at 10+ Hz
- MTF analysis produces valid results
- Results match LensLab validation targets

### Phase 3: Full Features (Weeks 5-6)
**Goal:** Complete camera control and analysis features

**Deliverables:**
- [ ] Full camera parameter control UI
- [ ] Multiple ROI support (5 positions)
- [ ] MTF curve visualization (ImPlot)
- [ ] Image capture (single, burst)
- [ ] Export (TIFF, PNG, JSON)
- [ ] Preset save/load
- [ ] Histogram display

**Success Criteria:**
- All camera parameters accessible
- MTF curves render correctly
- Images save without corruption

### Phase 4: Polish & Testing (Weeks 7-8)
**Goal:** Production-ready release

**Deliverables:**
- [ ] Multi-camera brand testing (Basler, FLIR, IDS, Allied Vision)
- [ ] Cross-platform builds (Windows, Linux, macOS)
- [ ] Error handling and logging
- [ ] Performance optimization
- [ ] User documentation
- [ ] Installer/packaging

**Success Criteria:**
- Works with all target camera brands
- Stable 24+ hour operation
- Clean install on fresh systems

---

## 7. Project Structure

```
lenslab-desktop/
├── CMakeLists.txt
├── README.md
├── CLAUDE.md
│
├── src/
│   ├── main.cpp                    # Entry point
│   ├── app/
│   │   ├── Application.h/cpp       # Main application class
│   │   ├── Config.h/cpp            # Settings management
│   │   └── Logger.h/cpp            # Logging system
│   │
│   ├── camera/
│   │   ├── CameraManager.h/cpp     # Camera discovery & management
│   │   ├── CameraDevice.h/cpp      # Single camera abstraction
│   │   ├── FrameBuffer.h/cpp       # Ring buffer for frames
│   │   └── GenICamHelpers.h/cpp    # GenICam utilities
│   │
│   ├── analysis/
│   │   ├── FocusMetrics.h/cpp      # Brenner, Tenengrad, ML
│   │   ├── MTFAnalyzer.h/cpp       # Ported from LensLab
│   │   ├── ROIManager.h/cpp        # ROI definition & tracking
│   │   └── QualityAssessment.h/cpp # ROI quality scoring
│   │
│   ├── ui/
│   │   ├── UIManager.h/cpp         # ImGui setup & main loop
│   │   ├── CameraPanel.h/cpp       # Camera list & controls
│   │   ├── PreviewPanel.h/cpp      # Live view with ROI overlay
│   │   ├── AnalysisPanel.h/cpp     # Results & MTF curves
│   │   └── SettingsPanel.h/cpp     # Preferences
│   │
│   └── utils/
│       ├── ImageIO.h/cpp           # Save/load images
│       ├── Timer.h/cpp             # Performance timing
│       └── ThreadPool.h/cpp        # Background processing
│
├── external/                        # Third-party libs (git submodules)
│   ├── imgui/
│   ├── implot/
│   ├── glfw/
│   └── json/
│
├── resources/
│   ├── fonts/
│   └── icons/
│
├── test/
│   ├── test_mtf_analyzer.cpp
│   ├── test_focus_metrics.cpp
│   └── test_images/
│
└── docs/
    ├── user_guide.md
    └── api_reference.md
```

---

## 8. Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GenTL producer compatibility | High | Medium | Test early with all vendors, document requirements |
| MTF analyzer port introduces bugs | High | Low | Validate against existing test suite |
| ImGui learning curve | Low | Low | Extensive documentation, familiar to user |
| Cross-platform OpenGL issues | Medium | Medium | Use GLFW, test early on all platforms |
| Performance bottlenecks | Medium | Low | Profile early, optimize hot paths |
| Camera vendor-specific quirks | Medium | High | Build compatibility notes, graceful fallbacks |

---

## 9. Success Criteria

### MVP Success (Phase 4 Complete)
- [ ] Connects to cameras from 4 vendors (Basler, FLIR, IDS, Allied Vision)
- [ ] Live focus metrics update at 15+ Hz
- [ ] MTF analysis accuracy within 15% of theoretical
- [ ] MTF variance < 1% between similar ROIs
- [ ] Stable operation for 8+ hours
- [ ] Clean build on Windows and Linux

### Quality Targets (from LensLab Validation)
- FWHM accuracy: < 20% error in controlled conditions
- ROI consistency: < 1% variance between similar ROIs
- MTF trend validation: MTF50 > MTF20 > MTF10 (correct ordering)
- Edge angle support: 5-20° (8-15° optimal)

---

## 10. Appendices

### Appendix A: LensLab Integration Reference

**Source Location:** `/Web Development/lenslab/`

**Key Files:**
- `src/cpp/mtf_analyzer_6.cpp` - Production MTF analyzer
- `src/cpp/pixel_lab_brenner.cpp` - Focus metrics (WASM)
- `MTF_ANALYZER_DEVELOPMENT_REPORT.md` - Validation results
- `CLAUDE.md` - Build instructions

**Validation Data:**
- FWHM accuracy: 15-37% error (acceptable for real-world use)
- Consistency: <1% variance achieved after coordinate fix
- Test patterns: `working_targets/` directory

### Appendix B: Camera Vendor GenTL Producers

| Vendor | Producer Name | Download |
|--------|---------------|----------|
| Basler | Basler pylon | baslerweb.com |
| FLIR/Teledyne | Spinnaker SDK | flir.com |
| IDS | IDS peak | ids-imaging.com |
| Allied Vision | Vimba X | alliedvision.com |

### Appendix C: Glossary

- **MTF:** Modulation Transfer Function - measure of optical resolution
- **ESF:** Edge Spread Function - intensity profile across an edge
- **LSF:** Line Spread Function - derivative of ESF
- **FWHM:** Full Width at Half Maximum - sharpness metric
- **GenICam:** Generic Interface for Cameras - standard API
- **GenTL:** GenICam Transport Layer - hardware abstraction
- **ROI:** Region of Interest - analysis area
- **Slant Edge:** Tilted edge target for MTF measurement (ISO 12233)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-14 | Claude | Initial draft |
| 2.0 | 2025-11-22 | Claude | Updated with C++/ImGui stack, integrated LensLab MTF analyzer, finalized architecture |
