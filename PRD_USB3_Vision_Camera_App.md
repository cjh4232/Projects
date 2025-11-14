# Product Requirements Document (PRD)
## Standalone USB3 Vision Camera Application

**Version:** 1.0
**Date:** November 14, 2025
**Project Code Name:** Pixel Lab Standalone
**Status:** Draft for Review

---

## 1. Executive Summary

This document outlines the requirements for a standalone desktop application that enables users to connect, control, and capture images from any USB3 Vision-compliant camera without requiring installation of proprietary drivers or vendor-specific software.

**Problem Statement:** Currently, using industrial USB3 Vision cameras requires:
- Installing vendor-specific SDKs and drivers
- Learning different software interfaces for each camera brand
- Managing compatibility issues across different systems
- Complex setup processes that create friction for users

**Solution:** A unified, plug-and-play application that handles all USB3 Vision cameras through a single interface, with embedded driver support and intuitive controls.

---

## 2. Product Overview

### 2.1 Product Vision
Create a professional-grade camera application that makes USB3 Vision cameras as easy to use as consumer webcams, while maintaining full access to industrial camera features and performance.

### 2.2 Target Market
- Machine vision engineers and integrators
- Quality control and inspection systems
- Research laboratories and academic institutions
- Prototype and development teams
- Industrial automation companies
- Small to medium manufacturers

### 2.3 Competitive Differentiation
- **Universal Compatibility:** Works with ANY USB3 Vision camera (Basler, FLIR, Allied Vision, IDS, etc.)
- **Zero Driver Installation:** Bundled drivers eliminate setup friction
- **Unified Interface:** Consistent UI regardless of camera brand
- **Rapid Deployment:** Install once, connect any camera
- **Cross-Platform:** Windows, Linux, and macOS support

---

## 3. Goals and Objectives

### 3.1 Primary Goals
1. **Eliminate Setup Friction:** Users should connect a camera and start capturing within 60 seconds
2. **Universal Camera Support:** Support 95%+ of USB3 Vision cameras on the market
3. **Professional Features:** Provide access to all camera parameters and capabilities
4. **Reliable Performance:** Achieve sustained frame rates matching camera specifications

### 3.2 Success Metrics
- Time from installation to first image capture: < 2 minutes
- Camera discovery success rate: > 98%
- Frame rate performance: > 95% of camera specification
- User satisfaction score: > 4.5/5.0
- Support tickets per user: < 0.1

### 3.3 Non-Goals (Out of Scope for v1.0)
- GigE Vision camera support (future consideration)
- Real-time image processing/analysis
- Multi-camera synchronization
- Video recording with encoding (raw capture only)
- Camera calibration tools

---

## 4. User Personas

### 4.1 Primary Persona: "Testing Tom"
**Role:** Quality Control Engineer
**Experience:** Moderate technical skills, familiar with cameras
**Pain Points:** Wastes hours installing drivers, needs to test different cameras
**Goals:** Quick camera evaluation, capture test images, verify specifications
**Use Case:** Needs to test 5 different camera models to select one for production line

### 4.2 Secondary Persona: "Research Rachel"
**Role:** Laboratory Researcher
**Experience:** Domain expert, limited IT knowledge
**Pain Points:** IT restrictions prevent driver installation, needs simple tools
**Goals:** Capture images for analysis, adjust exposure, save to specific formats
**Use Case:** Microscopy imaging with industrial camera for publication-quality images

### 4.3 Tertiary Persona: "Integration Ian"
**Role:** Machine Vision Integrator
**Experience:** Expert level, needs full camera control
**Pain Points:** Vendor software is limited, needs scriptable/automatable solution
**Goals:** Access all camera features, optimize performance, batch operations
**Use Case:** Configuring cameras for deployment in industrial inspection systems

---

## 5. Core Features

### 5.1 Camera Discovery and Connection (P0 - Critical)
**Description:** Automatic detection and connection of USB3 Vision cameras

**Requirements:**
- Auto-detect all connected USB3 Vision cameras on launch
- Display camera information (vendor, model, serial number, firmware version)
- Hot-plug support (detect cameras connected after launch)
- Connection status indicators (connected, disconnected, error states)
- Support for multiple cameras (list view with selection)
- Graceful handling of unsupported or malfunctioning cameras

**Acceptance Criteria:**
- Camera detected within 3 seconds of connection
- Display complete camera information before connection
- User can connect/disconnect cameras without application restart
- Error messages are clear and actionable

### 5.2 Live Camera Preview (P0 - Critical)
**Description:** Real-time video stream from connected camera

**Requirements:**
- Display live stream at camera's native resolution
- Adjustable preview window size (fit to window, 50%, 100%, 200%)
- Frame rate display (actual FPS achieved)
- Histogram overlay (optional, per channel)
- Crosshair/grid overlay (optional)
- Zoom and pan controls for high-resolution images
- Freeze frame capability
- Exposure warning indicators (over/under exposure highlighting)

**Acceptance Criteria:**
- Preview starts within 1 second of camera connection
- Frame rate matches camera capability (tolerance: -5%)
- Preview remains responsive during parameter changes
- Memory usage stable during extended preview sessions

### 5.3 Image Capture (P0 - Critical)
**Description:** Single frame and sequence capture functionality

**Requirements:**
- Single-shot capture (capture current frame)
- Burst mode (capture N frames as fast as possible)
- Time-lapse mode (capture every N seconds for M minutes)
- Save formats: TIFF (uncompressed), PNG, JPEG, BMP, RAW (camera-specific format)
- Metadata embedding (camera settings, timestamp, camera info)
- File naming templates (timestamp, sequence number, custom prefix)
- Save location selector with recent locations
- Capture queue indicator (for burst/time-lapse modes)
- Image counter (images captured this session)

**Acceptance Criteria:**
- Image saved within 500ms of capture trigger
- No dropped frames during burst capture at max camera rate
- All metadata correctly embedded in file headers
- File naming conflicts handled gracefully

### 5.4 Camera Parameter Control (P0 - Critical)
**Description:** Access and adjust camera settings and features

**Requirements:**
- **Exposure Control:**
  - Exposure time (manual, slider + numeric input)
  - Auto-exposure mode with ROI selection
  - Gain control (manual, auto)
  - Black level adjustment

- **Image Format:**
  - Resolution selection (if camera supports multiple)
  - Pixel format selection (Mono8, Mono12, RGB8, BayerRG8, etc.)
  - Binning and decimation controls
  - ROI (Region of Interest) selection

- **Timing:**
  - Frame rate control
  - Trigger mode (freerun, software, hardware)
  - Trigger source and polarity

- **Image Quality:**
  - White balance (manual, auto, one-push)
  - Gamma correction
  - Sharpness
  - Saturation (color cameras)

- **Advanced:**
  - LUT (Look-Up Table) selection
  - Defect pixel correction
  - Flat field correction
  - Test pattern generation

**UI Requirements:**
- Organized in collapsible categories
- Real-time preview updates as parameters change
- Reset to defaults button per category
- Preset save/load functionality (save current settings as named preset)
- Display parameter ranges and current values
- Disable unavailable features gracefully

**Acceptance Criteria:**
- All GenICam features exposed (read/write parameters)
- Parameter changes applied within 100ms
- Preview reflects changes immediately
- No crashes from invalid parameter combinations

### 5.5 Camera Information Display (P1 - Important)
**Description:** Comprehensive camera and image information

**Requirements:**
- Camera details panel:
  - Vendor, model, serial number
  - Firmware version
  - USB connection speed/bandwidth
  - Sensor information (size, pixel size, type)
  - Temperature (if available)

- Image information:
  - Resolution and pixel format
  - Current frame rate
  - Exposure time and gain values
  - Timestamp
  - Buffer status (frames in queue)

- Statistics:
  - Min/max/mean pixel values
  - Histogram data
  - Dropped frames counter

**Acceptance Criteria:**
- All information updates in real-time
- Information exportable to text/JSON format
- Temperature monitoring for cameras with sensors

### 5.6 User Preferences (P1 - Important)
**Description:** Application configuration and personalization

**Requirements:**
- Default save location
- Default file format and naming template
- Auto-start preview on camera connect
- Preview overlay preferences
- Keyboard shortcuts customization
- Theme selection (light/dark mode)
- Auto-check for updates
- Telemetry opt-in/out

**Acceptance Criteria:**
- Preferences persist across sessions
- Changes take effect immediately (no restart required)
- Import/export preferences file

---

## 6. Technical Requirements

### 6.1 USB3 Vision Standard Compliance
- Full USB3 Vision 1.0 specification compliance
- GenICam GenApi 2.0+ support
- Support for standard pixel formats defined in PFNC (Pixel Format Naming Convention)
- Handle cameras with custom/vendor-specific features

### 6.2 Performance Requirements
- **Latency:** < 50ms from sensor to screen display
- **Frame Rate:** Support up to 500 FPS (limited by camera/USB bandwidth)
- **Resolution:** Support up to 20 MP cameras
- **CPU Usage:** < 25% during live preview at 30 FPS (1080p)
- **Memory:** < 500 MB baseline, + buffer memory for high-speed capture
- **Startup Time:** < 5 seconds to ready state

### 6.3 Reliability Requirements
- **Uptime:** Application should run continuously for 24+ hours without degradation
- **Error Recovery:** Automatic recovery from temporary USB disconnections
- **Crash Rate:** < 0.1% (1 crash per 1000 hours of operation)
- **Data Integrity:** Zero data corruption in saved images

### 6.4 Platform Support
**Windows:**
- Windows 10 (64-bit) and newer
- Bundled WinUSB/LibUSB driver package
- Microsoft Visual C++ Redistributable handling

**Linux:**
- Ubuntu 20.04 LTS and newer
- Fedora 35+
- Generic Linux support via AppImage
- Built-in libusb support

**macOS:**
- macOS 11 (Big Sur) and newer
- Apple Silicon (M1/M2/M3) and Intel support
- Unsigned kernel extension alternative solution

### 6.5 Hardware Requirements
**Minimum:**
- CPU: Dual-core 2.0 GHz
- RAM: 4 GB
- USB: USB 3.0 port
- Display: 1280x720

**Recommended:**
- CPU: Quad-core 3.0 GHz
- RAM: 8 GB
- USB: USB 3.1 Gen 2 port
- Display: 1920x1080 or higher

---

## 7. Non-Functional Requirements

### 7.1 Usability
- First-time users should capture an image within 5 minutes without documentation
- All primary functions accessible within 2 clicks
- Tooltips on all controls
- Keyboard shortcuts for common operations
- Drag-and-drop file saving
- Undo/redo for parameter changes

### 7.2 Accessibility
- High contrast mode support
- Keyboard-only navigation
- Screen reader compatibility (ARIA labels)
- Minimum font size 12pt
- Colorblind-friendly UI (not relying solely on color)

### 7.3 Security
- No network communication required for core functionality
- Optional telemetry clearly disclosed
- No collection of sensitive data
- File system access limited to user-selected directories
- USB device access limited to USB3 Vision cameras

### 7.4 Maintainability
- Modular architecture for easy updates
- Comprehensive logging system
- Automated testing (unit, integration, UI)
- Clear error messages with diagnostic codes
- Plugin architecture for future extensions

### 7.5 Documentation
- In-app help system
- Quick start guide
- User manual (PDF + online)
- API documentation for scripting
- Troubleshooting guide
- FAQ section

---

## 8. Technology Stack Recommendations

### 8.1 Programming Language
**Option A: Python (Recommended for MVP)**
- **Pros:** Rapid development, rich ecosystem, cross-platform, strong imaging libraries
- **Cons:** Performance overhead, distribution complexity
- **Key Libraries:**
  - `harvesters` - GenICam/USB3Vision implementation
  - `genicam` - GenICam standard interface
  - `PyQt6` or `PySide6` - Professional UI framework
  - `opencv-python` - Image processing and display
  - `numpy` - Array operations
  - `pillow` - Image file I/O
  - `PyInstaller` or `cx_Freeze` - Standalone packaging

**Option B: C++ (For Production)**
- **Pros:** Maximum performance, direct hardware access, professional standard
- **Cons:** Longer development time, complexity
- **Key Libraries:**
  - `GenApi` - GenICam reference implementation
  - `Qt6` - UI framework
  - `OpenCV` - Image processing
  - `libusbp` - USB communication
  - `Aravis` - USB3 Vision implementation (Linux-friendly)

**Option C: Rust (Modern Alternative)**
- **Pros:** Performance + safety, growing ecosystem, modern tooling
- **Cons:** Smaller camera library ecosystem, steeper learning curve
- **Key Libraries:**
  - Custom USB3 Vision implementation needed
  - `egui` or `iced` - UI frameworks
  - `image` - Image handling

**Recommendation:** Start with **Python** for rapid prototyping and MVP validation. Consider C++ port for v2.0 if performance requirements demand it.

### 8.2 Architecture Pattern
**Model-View-Controller (MVC) with Service Layer**

```
┌─────────────────────────────────────────┐
│           User Interface (View)          │
│  [Qt Widgets/QML - Camera Control GUI]  │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│         Controller Layer                 │
│  [Event Handlers, Command Dispatcher]   │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│         Service Layer                    │
│  [Business Logic, Image Processing]     │
├─────────────────────────────────────────┤
│  • CameraManager Service                │
│  • ImageCapture Service                 │
│  • ParameterControl Service             │
│  • FileManager Service                  │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│         Model Layer                      │
│  [Data Models, State Management]        │
├─────────────────────────────────────────┤
│  • Camera Model                         │
│  • ImageBuffer Model                    │
│  • Settings Model                       │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│      Hardware Abstraction Layer         │
│   [USB3Vision/GenICam Interface]        │
└───────────────┬─────────────────────────┘
                │
┌───────────────▼─────────────────────────┐
│         USB3 Vision Cameras             │
└─────────────────────────────────────────┘
```

### 8.3 Key Components

**1. Camera Discovery Engine**
- USB device enumeration
- USB3Vision device identification (TLType filter)
- Vendor and model parsing from XML

**2. GenICam Parameter Manager**
- XML parsing (camera device description files)
- Feature tree navigation
- Type-safe parameter access
- Callback system for parameter changes

**3. Image Acquisition Pipeline**
- Buffer management (ring buffer implementation)
- Frame grabbing thread
- Pixel format conversion
- Display buffer preparation

**4. UI Framework**
- Main window (camera list, preview, controls)
- Parameter tree widget (collapsible, searchable)
- Image viewer with zoom/pan
- Settings dialog
- File save dialog with options

**5. File I/O Manager**
- Multi-format saving (TIFF, PNG, JPEG, BMP)
- Metadata encoding
- Batch operations
- Background saving thread

---

## 9. Development Phases

### Phase 1: Foundation (Weeks 1-3)
**Goal:** Prove core technology and basic functionality

**Deliverables:**
- [ ] Development environment setup
- [ ] USB3Vision/GenICam library integration
- [ ] Camera discovery working (enumerate and identify cameras)
- [ ] Basic Qt UI shell (window, menu, camera list)
- [ ] Single camera connection/disconnection
- [ ] Basic live preview (grayscale only, no controls)

**Success Criteria:**
- Can detect and list connected USB3 Vision cameras
- Can connect to a test camera and display live preview

### Phase 2: Core Features (Weeks 4-7)
**Goal:** Complete essential functionality for MVP

**Deliverables:**
- [ ] Full parameter control UI (exposure, gain, resolution)
- [ ] Image capture (single shot, save to TIFF/PNG)
- [ ] Preview enhancements (zoom, pan, histogram)
- [ ] Camera information display
- [ ] File management (naming, location selection)
- [ ] Color camera support (RGB, Bayer)
- [ ] Multi-camera support (connect multiple, switch between)

**Success Criteria:**
- Complete "first image capture" workflow from camera connect to saved file
- All basic camera parameters controllable
- Works with at least 2 different camera brands

### Phase 3: Advanced Features (Weeks 8-10)
**Goal:** Add professional-grade capabilities

**Deliverables:**
- [ ] Burst capture mode
- [ ] Time-lapse capture
- [ ] Preset save/load system
- [ ] Advanced parameter access (trigger, LUT, etc.)
- [ ] Keyboard shortcuts
- [ ] User preferences system
- [ ] Image metadata embedding
- [ ] Statistics display (histogram, min/max/mean)

**Success Criteria:**
- Can capture sequences without dropped frames
- Settings persist across sessions
- Advanced users can access all camera features

### Phase 4: Polish & Packaging (Weeks 11-12)
**Goal:** Production-ready release

**Deliverables:**
- [ ] Cross-platform testing and fixes
- [ ] Installer creation (Windows, Linux packages, macOS DMG)
- [ ] Bundled driver packaging
- [ ] User documentation
- [ ] Error handling and logging
- [ ] Performance optimization
- [ ] Automated testing suite
- [ ] Beta testing program

**Success Criteria:**
- Installer works on clean systems
- No critical bugs in issue tracker
- Documentation complete
- Positive feedback from beta testers

### Phase 5: Release & Iteration (Week 13+)
**Goal:** Launch and improve based on feedback

**Deliverables:**
- [ ] v1.0 release
- [ ] Marketing website/landing page
- [ ] User onboarding flow
- [ ] Support system setup
- [ ] Analytics and crash reporting
- [ ] Regular updates based on feedback

---

## 10. User Stories (MVP)

### Epic 1: Camera Connection
- As a user, I want to see all connected cameras when I launch the app
- As a user, I want to connect to a camera with one click
- As a user, I want to see if a camera disconnects unexpectedly
- As a user, I want to reconnect without restarting the application

### Epic 2: Image Preview
- As a user, I want to see a live preview immediately after connecting
- As a user, I want to zoom in to inspect image details
- As a user, I want to see a histogram to judge exposure
- As a user, I want to freeze the preview to examine a frame

### Epic 3: Camera Control
- As a user, I want to adjust exposure time to control brightness
- As a user, I want to adjust gain for low-light conditions
- As a user, I want to enable auto-exposure for quick setup
- As a user, I want to save my settings as a preset

### Epic 4: Image Capture
- As a user, I want to capture the current frame with one click
- As a user, I want to choose where to save images
- As a user, I want images saved in TIFF format for maximum quality
- As a user, I want to capture a burst of 10 images rapidly

---

## 11. Open Questions & Decisions Needed

### 11.1 Licensing & Distribution
- [ ] **Software License:** Open source (MIT/Apache) or proprietary?
- [ ] **Business Model:** Free, freemium, or paid?
- [ ] **Driver Licensing:** Confirm redistribution rights for USB3Vision drivers
- [ ] **GenICam License:** Review GenICam usage terms

### 11.2 Technical Decisions
- [ ] **Primary Language:** Python (fast dev) vs C++ (performance)?
- [ ] **UI Framework:** Qt (professional) vs web-based (Electron/Tauri)?
- [ ] **Harvesters vs Vimba:** Which GenICam library for Python?
- [ ] **Buffer Strategy:** How many frames to buffer for high-speed capture?
- [ ] **Thread Model:** Single UI thread + worker threads, or async architecture?

### 11.3 Feature Prioritization
- [ ] **Recording:** Is video recording (not just single frames) needed for v1.0?
- [ ] **Processing:** Should basic processing (rotate, crop, adjust levels) be included?
- [ ] **Calibration:** Camera calibration tools in scope for v1.0?
- [ ] **Scripting:** Command-line or Python API for automation?
- [ ] **Multi-camera Sync:** Hardware sync for multi-camera capture in v1.0?

### 11.4 UX Decisions
- [ ] **Default Layout:** Single window or multi-window interface?
- [ ] **Preset Sharing:** Should presets be exportable/shareable between users?
- [ ] **Cloud Features:** Any cloud storage or sharing features?
- [ ] **Wizard:** Should there be a setup wizard for first-time users?

### 11.5 Testing & Validation
- [ ] **Test Cameras:** Which camera brands/models to test with?
  - Suggested: Basler ace, FLIR Blackfly S, IDS uEye+, Allied Vision Alvium
- [ ] **Beta Testers:** Who are target beta users?
- [ ] **Success Metrics:** What metrics to track for v1.0 validation?

---

## 12. Risks and Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GenICam library compatibility issues | High | Medium | Early testing with multiple cameras, fallback to alternative libraries |
| Driver bundling legal/technical problems | High | Medium | Legal review of licenses, test installer on clean systems |
| Performance inadequate for high-speed cameras | Medium | Low | Optimize critical path, consider C++ for bottlenecks |
| USB3 bandwidth limitations | Medium | Medium | Implement buffer management, educate users on USB3 requirements |
| Camera vendor-specific quirks | Medium | High | Build compatibility database, implement vendor-specific workarounds |
| Cross-platform Qt deployment complexity | Low | Medium | Use Qt installer framework, test early on all platforms |
| Scope creep delaying MVP | Medium | High | Strict phase gates, defer non-critical features |

---

## 13. Success Criteria

### MVP Success (Phase 4 Complete)
- [ ] Successfully tested with 5+ different camera brands
- [ ] 10+ beta users complete first-image-capture workflow
- [ ] Average time to first image < 5 minutes for new users
- [ ] Zero critical bugs in issue tracker
- [ ] Installer works on 3 platforms without manual intervention
- [ ] Positive feedback from all beta testers

### Product-Market Fit (Post-Launch)
- [ ] 100+ active installations within 3 months
- [ ] User satisfaction score > 4.0/5.0
- [ ] < 5% churn rate (users who uninstall within 30 days)
- [ ] Organic word-of-mouth growth
- [ ] Feature requests align with roadmap

---

## 14. Next Steps

1. **Review & Approve PRD:** Stakeholder review and sign-off
2. **Technology Proof-of-Concept:** Validate Python + Harvesters + Qt stack
3. **Test Camera Acquisition:** Obtain 2-3 different USB3Vision cameras for testing
4. **Architecture Design:** Detailed component design and API specifications
5. **Development Environment:** Set up dev tools, repos, CI/CD
6. **Phase 1 Kickoff:** Begin foundation development

---

## Appendix A: Reference Materials

### USB3 Vision Standard
- [AIA USB3 Vision Specification](https://www.visiononline.org/vision-standards-details.cfm?type=5)
- GenICam Standard Documentation: http://www.genicam.org/

### Recommended Libraries
- **Harvesters:** https://github.com/genicam/harvesters
- **PyQt6:** https://www.riverbankcomputing.com/software/pyqt/
- **OpenCV:** https://opencv.org/
- **Aravis:** https://github.com/AravisProject/aravis (Linux/GStreamer)

### Camera Vendor Resources
- Basler: https://www.baslerweb.com/
- FLIR (Teledyne): https://www.flir.com/products/blackfly-s-usb3/
- Allied Vision: https://www.alliedvision.com/
- IDS Imaging: https://en.ids-imaging.com/

### Competitive Analysis
- Basler Pylon Viewer (free, vendor-specific)
- FLIR Spinnaker (free, vendor-specific)
- MVTec HALCON (expensive, professional)
- Cognex VisionPro (expensive, professional)

---

## Appendix B: Glossary

- **USB3 Vision:** Standard protocol for industrial cameras over USB 3.0
- **GenICam:** Generic standard interface for cameras (vendor-agnostic)
- **GenApi:** API specification within GenICam for camera control
- **PFNC:** Pixel Format Naming Convention (standard pixel formats)
- **ROI:** Region of Interest (subset of image sensor)
- **FPS:** Frames Per Second
- **Bayer Pattern:** Color filter array for single-sensor color cameras
- **LUT:** Look-Up Table (for gamma/tone correction)
- **AOI:** Area of Interest (synonym for ROI)
- **Binning:** Combining adjacent pixels to reduce resolution and increase sensitivity
- **Decimation:** Skipping pixels to reduce resolution
- **Trigger:** External signal to initiate frame capture

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-14 | Claude | Initial draft for review |

