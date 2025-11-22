/**
 * LensLab Desktop - USB3 Vision Camera Application with Focus Analysis
 *
 * Main entry point
 */

#include "app/Application.h"
#include "app/Logger.h"

#include <iostream>
#include <exception>

int main(int argc, char* argv[])
{
    try {
        // Initialize logging
        lenslab::Logger::init("lenslab.log");
        LOG_INFO("LensLab Desktop starting...");

        // Create and run application
        lenslab::Application app;

        if (!app.init(argc, argv)) {
            LOG_ERROR("Failed to initialize application");
            return 1;
        }

        // Main loop
        int result = app.run();

        LOG_INFO("LensLab Desktop shutting down...");
        return result;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        LOG_ERROR("Fatal error: {}", e.what());
        return 1;
    }
}
