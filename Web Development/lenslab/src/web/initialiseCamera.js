let stream;

function initializeCamera() {
      const localVideo = document.getElementById('localVideo');
      const overlay = document.getElementById('overlay');
      const startButton = document.getElementById('startButton');
      const stopButton = document.getElementById('stopButton');
      const status = document.getElementById('status');
  
      async function startCamera() {
          try {
              if (stream) {
                  stream.getTracks().forEach(track => track.stop());
              }
              stream = await navigator.mediaDevices.getUserMedia({video: true});
              localVideo.srcObject = stream;
              status.textContent = 'Camera started';
              startButton.textContent = 'Camera Started';
              startButton.disabled = true;
              stopButton.disabled = false;
  
              localVideo.onloadedmetadata = () => {
                  overlay.width = localVideo.videoWidth;
                  overlay.height = localVideo.videoHeight;
                  drawROIs();
              };
          } catch (error) {
              console.error('Error accessing the camera:', error);
              status.textContent = 'Error accessing the camera: ' + error.message;
          }
      }
  
      startCamera();
  }


  function drawROIs() {
    const overlay = document.getElementById('overlay');
    const ctx = overlay.getContext('2d');
    const width = overlay.width;
    const height = overlay.height;
    const squareSize = Math.min(width, height) / 10;

    ctx.clearRect(0, 0, width, height);
    
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    ctx.scale(-1, 1);
    ctx.translate(-width, 0);

    roiCoordinates = [
        {x: width / 2 - squareSize / 2, y: height / 2 - squareSize / 2, size: squareSize, label: 'C'},
        {x: squareSize, y: squareSize, size: squareSize, label: 'UL'},
        {x: width - 2 * squareSize, y: squareSize, size: squareSize, label: 'UR'},
        {x: squareSize, y: height - 2 * squareSize, size: squareSize, label: 'LL'},
        {x: width - 2 * squareSize, y: height - 2 * squareSize, size: squareSize, label: 'LR'}
    ];

    roiCoordinates.forEach((roi, index) => {
        drawSquare(ctx, roi.x, roi.y, roi.size);
        drawLabel(ctx, roi.x, roi.y, roi.size, roi.label, index);
    });

    ctx.setTransform(1, 0, 0, 1, 0, 0);
}


function drawLabel(ctx, x, y, size, label, index) {
    const padding = 5;
    const bgHeight = 40;
    const bgWidth = 60;
    
    // Determine position for label background
    let bgX, bgY;
    if (label === 'C') {
        bgX = x + size + padding;
        bgY = y + (size - bgHeight) / 2;
    } else if (label === 'UL') {
        bgX = x + size + padding;
        bgY = y;
    } else if (label === 'UR') {
        bgX = x - bgWidth - padding;
        bgY = y;
    } else if (label === 'LL') {
        bgX = x + size + padding;
        bgY = y + size - bgHeight;
    } else if (label === 'LR') {
        bgX = x - bgWidth - padding;
        bgY = y + size - bgHeight;
    }

    // Draw white background
    ctx.fillStyle = 'white';
    ctx.fillRect(bgX, bgY, bgWidth, bgHeight);

    // Draw label and value
    ctx.fillStyle = 'red';
    ctx.fillText(label, bgX + 5, bgY + 20);
    if (analysisResults[index] !== undefined) {
        ctx.fillText(analysisResults[index].toFixed(2), bgX + 5, bgY + 35);
    }
}
