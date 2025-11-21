async function analyzeROIFrames() {
    const startTime = performance.now();
    
    const videoElement = instance.data.videoElement;
    const overlayElement = instance.data.overlayElement;
    
    if (!videoElement || !overlayElement) {
        console.error('Video or overlay element not found');
        return;
    }

    try {
        await instance.data.wasmReady;

        // Debug video and overlay dimensions and positions
        console.log('Video element dimensions:', {
            width: videoElement.clientWidth,
            height: videoElement.clientHeight,
            offsetLeft: videoElement.offsetLeft,
            offsetTop: videoElement.offsetTop,
            boundingRect: videoElement.getBoundingClientRect()
        });

        console.log('Overlay dimensions:', {
            width: overlayElement.width,
            height: overlayElement.height,
            offsetLeft: overlayElement.offsetLeft,
            offsetTop: overlayElement.offsetTop,
            boundingRect: overlayElement.getBoundingClientRect()
        });

        // Verify ROI coordinates
        console.log('ROI Coordinates:', instance.data.roiCoordinates);

        // Create temporary canvas for ROI extraction
        const tempCanvas = document.createElement('canvas');
        const roiSize = Math.min(videoElement.clientWidth, videoElement.clientHeight) / 10;
        tempCanvas.width = roiSize;
        tempCanvas.height = roiSize;
        const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });

        // Clean metric name
        const cleanMetricName = (metric) => {
            if (!metric) return "Combined";
            return metric.replace(/"/g, '').trim();
        };

        const metricMap = {
            "Modified Laplacian": "Modified Laplacian",
            "Tenengrad": "Tenengrad",
            "Energy Gradient": "Energy Gradient",
            "Combined": "Combined"
        };

        const cleanedMetric = metricMap[cleanMetricName(selectedMetric)] || "Combined";

        // Add boundary check for ROIs
        const isWithinBounds = (x, y, size) => {
            return x >= 0 && 
                   y >= 0 && 
                   (x + size) <= videoElement.clientWidth && 
                   (y + size) <= videoElement.clientHeight;
        };

        // Capture current frame data with bounds checking
        const currentFrameData = instance.data.roiCoordinates.map((roi, index) => {
            const flippedX = videoElement.clientWidth - (roi.x + roi.size);
            
            console.log(`Capturing ROI ${roi.label}:`, {
                original: { x: roi.x, y: roi.y },
                flipped: { x: flippedX, y: roi.y },
                size: roi.size,
                videoWidth: videoElement.clientWidth,
                videoHeight: videoElement.clientHeight
            });

            tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
            
            try {
                // Create an intermediate canvas to handle the flipping
                const intermediateCanvas = document.createElement('canvas');
                intermediateCanvas.width = roi.size;
                intermediateCanvas.height = roi.size;
                const intermediateCtx = intermediateCanvas.getContext('2d');

                // First, capture the ROI region directly
                intermediateCtx.drawImage(videoElement,
                    roi.x, roi.y,  // Use original coordinates
                    roi.size, roi.size,
                    0, 0,
                    roi.size, roi.size
                );

                // Flip the intermediate canvas horizontally
                tempCtx.save();
                tempCtx.scale(-1, 1);
                tempCtx.translate(-tempCanvas.width, 0);
                tempCtx.drawImage(intermediateCanvas,
                    0, 0,
                    roi.size, roi.size,
                    0, 0,
                    tempCanvas.width, tempCanvas.height
                );
                tempCtx.restore();

                const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);

                // Debug the pixel values
                const hasNonZeroPixels = Array.from(imageData.data).some(value => value > 0);
                console.log(`ROI ${roi.label} pixel check:`, {
                    hasNonZeroPixels,
                    firstFewPixels: Array.from(imageData.data.slice(0, 40)),
                    width: imageData.width,
                    height: imageData.height,
                    position: { x: roi.x, y: roi.y }
                });

                return {
                    roi: index,
                    imageData: imageData
                };
            } catch (error) {
                console.error(`Error capturing ROI ${roi.label}:`, error);
                return {
                    roi: index,
                    imageData: null
                };
            }
        });

        // Filter out any failed ROI captures
        const validFrameData = currentFrameData.filter(data => data.imageData !== null);

        if (validFrameData.length !== currentFrameData.length) {
            console.warn(`Some ROIs were not captured successfully. Expected: ${currentFrameData.length}, Got: ${validFrameData.length}`);
        }

        // Add to frame buffer only if we have valid data
        if (validFrameData.length > 0) {
            frameBuffer.push(validFrameData);
        }

        // If we have enough frames, analyze the averaged data
        if (frameBuffer.length >= FRAME_BUFFER_SIZE) {
            // Average the image data for each ROI
            const results = instance.data.roiCoordinates.map((_, roiIndex) => {
                // Create averaged image data
                const width = tempCanvas.width;
                const height = tempCanvas.height;
                const avgData = new Uint8ClampedArray(width * height * 4);

                // Sum all frames
                for (let i = 0; i < avgData.length; i++) {
                    let sum = 0;
                    for (let frame of frameBuffer) {
                        if (frame[roiIndex] && frame[roiIndex].imageData) {
                            sum += frame[roiIndex].imageData.data[i];
                        }
                    }
                    avgData[i] = sum / FRAME_BUFFER_SIZE;
                }

                try {
                    // Create new ImageData with the averaged values
                    const avgImageData = new ImageData(avgData, width, height);

                    // Analyze using WebAssembly
                    const analysisResult = instance.data.wasmModule.analyzeImage(
                        avgImageData.data,
                        width,
                        height,
                        cleanedMetric
                    );

                    console.log(`Analysis result for ROI ${roiIndex}:`, analysisResult);

                    return {
                        roi: roiIndex,
                        quality_score: analysisResult.quality_score,
                        details: analysisResult.details
                    };
                } catch (error) {
                    console.error(`Error analyzing ROI ${roiIndex}:`, error);
                    return {
                        roi: roiIndex,
                        quality_score: 0,
                        details: { error: error.message }
                    };
                }
            });

            if (results && results.length > 0) {
                if (instance.data.isUpdatingLabels) {
                    console.warn('Already updating labels! Skipping this update.');
                } else {
                    instance.data.isUpdatingLabels = true;
                    console.log('Starting label update');
                    console.log('Pre-update results:', results);
                    results.forEach(result => {
                        if (result.quality_score === 0) {
                            console.warn(`Zero score detected for ROI ${result.roi}`);
                        }
                    });
                    updateROILabels(results);
                    instance.data.isUpdatingLabels = false;
                    console.log('Finished label update');
                }
            } else {
                console.warn('No results to update labels with');
            }

            // Clear buffer
            frameBuffer = [];
        }

        // Calculate processing time and adjust interval
        const processingTime = performance.now() - startTime;
        processingTimes.push(processingTime);
        if (processingTimes.length > HISTORY_SIZE) {
            processingTimes.shift();
        }

        // Calculate average processing time
        const avgProcessingTime = processingTimes.reduce((a, b) => a + b) / processingTimes.length;

        // Adjust interval based on processing time
        if (avgProcessingTime > currentInterval) {
            currentInterval = Math.min(MAX_INTERVAL, avgProcessingTime * 1.2);
        } else if (avgProcessingTime < currentInterval / 2) {
            currentInterval = Math.max(MIN_INTERVAL, currentInterval * 0.8);
        }

        console.log(`Processing time: ${processingTime.toFixed(2)}ms, Current interval: ${currentInterval.toFixed(2)}ms`);

        // If still measuring, schedule next analysis
        if (instance.data.isMeasuring) {
            instance.data.captureTimeout = setTimeout(analyzeROIFrames, currentInterval);
        }

    } catch (error) {
        console.error('Error analyzing ROIs:', error);
        stopMeasurement();
    }
}