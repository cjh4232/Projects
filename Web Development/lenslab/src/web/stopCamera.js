function stopCamera() {
      const localVideo = document.getElementById('localVideo');
      const overlay = document.getElementById('overlay');
      const startButton = document.getElementById('startButton');
      const stopButton = document.getElementById('stopButton');
      const status = document.getElementById('status');
  
      if (stream) {
          stream.getTracks().forEach(track => track.stop());
          localVideo.srcObject = null;
          stream = null;
          status.textContent = 'Camera stopped';
          startButton.textContent = 'Start Camera';
          startButton.disabled = false;
          stopButton.disabled = true;
  
          const ctx = overlay.getContext('2d');
          ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
      stopCapture();
  }