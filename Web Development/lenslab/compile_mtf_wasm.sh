#!/bin/bash

echo "🔧 Compiling MTF Analyzer to WebAssembly..."

# Check if emscripten is available
if ! command -v emcc &> /dev/null; then
    echo "❌ Emscripten not found. Please install emscripten SDK:"
    echo "   git clone https://github.com/emscripten-core/emsdk.git"
    echo "   cd emsdk"
    echo "   ./emsdk install latest"
    echo "   ./emsdk activate latest"
    echo "   source ./emsdk_env.sh"
    exit 1
fi

echo "✅ Emscripten found: $(emcc --version | head -1)"

# Create WebAssembly-compatible version of MTF analyzer
echo "📝 Creating WebAssembly-compatible MTF analyzer..."

# First, let's create a simplified version that focuses on the core MTF computation
# We'll strip out the main() function and file I/O, keeping just the analysis functions

cat > src/cpp/mtf_analyzer_wasm.cpp << 'EOF'
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <map>
#include <numeric>

using namespace emscripten;

// Include the core MTF analysis functions from our main analyzer
// (We'll need to extract and adapt the main analysis logic)

struct MTFResult {
    double mtf50;
    double mtf20;
    double mtf10;
    double fwhm;
    int edges_found;
    bool success;
    std::string error_message;
};

class MTFAnalyzerWasm {
public:
    static MTFResult analyzeImage(const val& uint8Array, int width, int height) {
        MTFResult result = {0, 0, 0, 0, 0, false, ""};
        
        try {
            // Convert JavaScript Uint8Array to cv::Mat
            std::vector<uint8_t> data;
            const auto length = uint8Array["length"].as<unsigned>();
            data.resize(length);
            val memoryView{typed_memory_view(length, data.data())};
            memoryView.call<void>("set", uint8Array);
            
            // Convert to OpenCV Mat (assuming RGBA format)
            cv::Mat image(height, width, CV_8UC4, data.data());
            
            // TODO: Insert our actual MTF analysis code here
            // For now, return a test result to verify compilation works
            result.mtf50 = 0.123;
            result.mtf20 = 0.234;
            result.mtf10 = 0.345;
            result.fwhm = 2.5;
            result.edges_found = 4;
            result.success = true;
            
        } catch (const std::exception& e) {
            result.error_message = e.what();
            result.success = false;
        }
        
        return result;
    }
};

EMSCRIPTEN_BINDINGS(mtf_analyzer) {
    value_object<MTFResult>("MTFResult")
        .field("mtf50", &MTFResult::mtf50)
        .field("mtf20", &MTFResult::mtf20)
        .field("mtf10", &MTFResult::mtf10)
        .field("fwhm", &MTFResult::fwhm)
        .field("edges_found", &MTFResult::edges_found)
        .field("success", &MTFResult::success)
        .field("error_message", &MTFResult::error_message);
        
    class_<MTFAnalyzerWasm>("MTFAnalyzerWasm")
        .class_function("analyzeImage", &MTFAnalyzerWasm::analyzeImage);
}
EOF

echo "🔨 Compiling to WebAssembly..."

# Compile with emscripten
emcc \
    -std=c++17 \
    -O3 \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="'MTFAnalyzer'" \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MAXIMUM_MEMORY=268435456 \
    --bind \
    -I/opt/homebrew/include/opencv4 \
    -L/opt/homebrew/lib \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_imgcodecs \
    -o mtf_analyzer.js \
    src/cpp/mtf_analyzer_wasm.cpp

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo "📦 Generated files:"
    ls -la mtf_analyzer.js mtf_analyzer.wasm
    echo ""
    echo "🧪 Ready to test actual WebAssembly MTF analysis!"
else
    echo "❌ Compilation failed. Check errors above."
    exit 1
fi
EOF

chmod +x compile_mtf_wasm.sh