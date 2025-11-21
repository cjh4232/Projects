function startMeasurement() {
      const metricSelector = document.getElementById('qualityMetricSelector');
      const selectedMetric = metricSelector.value;
      console.log('Selected metric:', selectedMetric);
  
      if (!selectedMetric) {
          alert('Please select a quality metric before starting measurement.');
          return;
      }
      
      if (!selectedMetric) {
          console.error('No metric selected. Please choose a metric before starting measurement.');
          return;
      }
  
      if (!isCapturing) {
          startCapture();
      }
      isMeasuring = true;
      captureAndSendFrames();
  }


  function startCapture() {  
    if (isCapturing) return;
    isCapturing = true;
    const ngrokUrl = 'wss://lenslab.ngrok.app/';
    console.log('Attempting to connect to WebSocket at:', ngrokUrl);
    websocket = new WebSocket(ngrokUrl);
    websocket.onopen = () => {
        console.log('WebSocket connection established');
        isWebSocketReady = true; 
    };
    websocket.onmessage = (event) => {
        console.log('Received message from server:', event.data);
        const results = JSON.parse(event.data);
        displayResults(results);
    };
    websocket.onerror = (error) => {
    console.error('WebSocket error:', error);
    console.error('WebSocket readyState:', websocket.readyState);
    stopCapture();
};
    websocket.onclose = () => {
        console.log('WebSocket connection closed');
        isWebSocketReady = false;
        stopCapture();
    };
}


function captureAndSendFrames() {
    if (!isMeasuring) return;

    const video = document.getElementById('localVideo');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');

    ctx.scale(-1, 1);
    ctx.translate(-canvas.width, 0);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    const frames = roiCoordinates.map((roi, index) => {
        const roiCanvas = document.createElement('canvas');
        roiCanvas.width = roi.size;
        roiCanvas.height = roi.size;
        const roiCtx = roiCanvas.getContext('2d');
        roiCtx.drawImage(canvas, roi.x, roi.y, roi.size, roi.size, 0, 0, roi.size, roi.size);
        return {
            index: index,
            dataUrl: roiCanvas.toDataURL('image/jpeg', 0.8)
        };
    });

    frameBuffer.push(frames);
    if (frameBuffer.length >= 10 && isWebSocketReady) {
        const metricSelector = document.getElementById('qualityMetricSelector');
        const selectedMetric = metricSelector.value;

        if (!selectedMetric) {
            console.error('No metric selected. Stopping measurement.');
            stopMeasurement();
            return;
        }
        
        const dataToSend = {
            frames: frameBuffer,
            metric: selectedMetric
        };
        
        console.log('Sending frame buffer to server');
        console.log('Frame buffer size:', JSON.stringify(dataToSend).length, 'bytes');
        websocket.send(JSON.stringify(dataToSend));
        frameBuffer = [];
    }

    requestAnimationFrame(captureAndSendFrames);
}