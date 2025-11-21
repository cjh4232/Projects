"""
WebSocket Server for Real-time Image Analysis

This module implements a WebSocket server that receives batches of
image frames, processes them, and returns analysis results.
It's designed to work in conjunction with a web application
that captures and sends video frames for analysis.

The server performs the following main functions:
1. Establishes a WebSocket connection to receive frame data.
2. Averages multiple frames for each Region of Interest (ROI).
3. Analyzes the averaged frames based on the selected metric.
4. Sends the analysis results back to the client.

This server is intended for use in image quality assessment scenarios, such as
camera alignment or focus checking. The actual image analysis logic should be
implemented in the analyze_averaged_frames function.

Usage:
    Run this script to start the WebSocket server:
    $ python websocket_server.py

    The server will start listening for WebSocket connections on
    localhost:8765.

Dependencies:
    - websockets: For WebSocket communication
    - numpy: For numerical operations on image data
    - Pillow (PIL): For image processing
    - (Optional) OpenCV (cv2): For advanced image analysis

Note:
    This is a basic implementation and may need to be extended or modified
    for production use, including proper error handling, logging, and
    security measures.
"""

import asyncio
import traceback
import time
import json
import sys
import subprocess
from base64 import b64decode
from io import BytesIO
import logging
from typing import List, Dict, Any
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude

try:
    import cv2 as cv
    print(f"OpenCV version: {cv.__version__}")
except ImportError as e:
    print(f"Error importing OpenCV: {e}")
    print("Make sure opencv-python is installed correctly.")
    cv = None

logging.basicConfig(level=logging.INFO)

NGROK_URL = 'wss://lenslab.ngrok.app'

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import websockets
    print(f"Websockets version: {websockets.__version__}")
except ImportError as e:
    print(f"Error importing websockets: {e}")
    print("\nInstalled packages:")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list"],
        capture_output=True,
        text=True,
        check=False
        )
    print(result.stdout)
    sys.exit(1)


class MetricType(Enum):
    """Enumeration of supported focus metrics"""
    MODIFIED_LAPLACIAN = auto()
    TENENGRAD = auto()
    ENERGY_GRADIENT = auto()
    COMBINED = auto()
    UNKNOWN = auto()

    @classmethod
    def from_string(cls, metric: str) -> 'MetricType':
        """
        Convert string to MetricType with robust cleaning:
        - Strips quotation marks
        - Converts to lowercase
        - Replaces spaces with underscores

        Example:
        '"Modified Laplacian"' -> 'modified_laplacian'
        """
        if not metric:
            return cls.UNKNOWN

        # Clean the metric string
        cleaned_metric = (
            metric.strip('"\'')  # Remove quotes
                  .lower()       # Convert to lowercase
                  .replace(' ', '_')  # Replace spaces with underscores
        )

        # Map cleaned strings to metric types
        metric_map = {
            'modified_laplacian': cls.MODIFIED_LAPLACIAN,
            'tenengrad': cls.TENENGRAD,
            'energy_gradient': cls.ENERGY_GRADIENT,
            'combined': cls.COMBINED
        }

        logging.info(f"Cleaned metric '{metric}' to '{cleaned_metric}'")
        return metric_map.get(cleaned_metric, cls.UNKNOWN)


class ImageAnalyzer:
    """Class to handle image analysis with cached computations"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def process_batch(self, frame_batch: List[List[Dict[str, Any]]],
                          metric_type: MetricType) -> List[Dict[str, Any]]:
        """Process entire batch of frames with logging"""
        logging.info(f"Processing batch of {len(frame_batch)} frames")
        results = []

        for roi_index in range(len(frame_batch[0])):
            logging.info(f"Processing ROI {roi_index}")
            start_time = time.time()

            # Decode frames for this ROI
            try:
                roi_frames = []
                for frame in frame_batch:
                    img_data = b64decode(frame[roi_index]['dataUrl'].split(',')[1])
                    frame_array = np.array(Image.open(BytesIO(img_data)))
                    roi_frames.append(frame_array)

                # Process ROI in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.process_roi,
                    roi_frames,
                    metric_type
                )

                process_time = time.time() - start_time
                logging.info(f"ROI {roi_index} processed in {process_time:.3f} seconds")

                results.append({'roi': roi_index, **result})

            except Exception as e:
                logging.error(f"Error processing ROI {roi_index}: {e}")
                logging.error(traceback.format_exc())
                results.append({
                    'roi': roi_index,
                    'error': str(e),
                    'quality_score': 0.0
                })

        return results

    def process_roi(self, frames: List[np.ndarray], metric_type: MetricType) -> Dict[str, Any]:
        """Process a single ROI with cached computations"""
        # Average frames efficiently
        averaged = self._average_frames(frames)

        # Convert to grayscale once
        gray = cv.cvtColor(averaged, cv.COLOR_BGR2GRAY)

        # Calculate score based on metric type
        score = self._calculate_metric(averaged, gray, metric_type)

        return {'quality_score': float(score)}

    def _average_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """Efficient frame averaging with pre-allocation"""
        accumulated = np.zeros_like(frames[0], dtype=np.float32)
        for frame in frames:
            accumulated += frame
        return (accumulated / len(frames)).astype(np.uint8)

    def _calculate_metric(self, frame: np.ndarray,
                          gray: np.ndarray,
                          metric_type: MetricType) -> float:
        """Calculate metric with cached grayscale"""
        if metric_type == MetricType.MODIFIED_LAPLACIAN:
            return self._measure_modified_laplacian(gray)
        elif metric_type == MetricType.TENENGRAD:
            return self._measure_tenengrad(gray)
        elif metric_type == MetricType.ENERGY_GRADIENT:
            return self._measure_energy_gradient(gray)
        else:  # COMBINED or UNKNOWN
            return self._measure_combined_focus(frame, gray)

    def _measure_modified_laplacian(self, gray: np.ndarray) -> float:
        """Optimized Modified Laplacian calculation"""
        gray_float = gray.astype(np.float64)
        kernel_x = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]], dtype=np.float32)
        kernel_y = kernel_x.transpose()

        lap_x = np.abs(cv.filter2D(gray_float, -1, kernel_x))
        lap_y = np.abs(cv.filter2D(gray_float, -1, kernel_y))
        ml = lap_x + lap_y

        threshold = np.percentile(ml, 90)
        return normalize_score(np.mean(ml[ml > threshold]))

    def _measure_tenengrad(self, gray: np.ndarray) -> float:
        """Optimized Tenengrad calculation"""
        kernel_size = 3  # Fixed size for performance
        gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=kernel_size)
        gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=kernel_size)

        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        threshold = np.mean(gradient_magnitude) * 1.5
        gradient_magnitude[gradient_magnitude < threshold] = 0

        return normalize_score(np.mean(gradient_magnitude**2))

    def _measure_energy_gradient(self, gray: np.ndarray) -> float:
        """Optimized Energy Gradient calculation"""
        gray_float = gray.astype(np.float64)
        gradient_energy = gaussian_gradient_magnitude(gray_float, sigma=2)
        high_freq = gradient_energy - gaussian_gradient_magnitude(gray_float, sigma=4)

        focus_measure = (0.7 * np.mean(high_freq**2) + 
                        0.3 * np.mean(gradient_energy**2))

        return normalize_score(focus_measure)

    def _measure_combined_focus(self, frame: np.ndarray, gray: np.ndarray) -> float:
        """Optimized combined focus measure"""
        # Quick characteristic check
        freq_measure = cv.Laplacian(gray, cv.CV_64F).var()
        noise_measure = self._estimate_noise(gray)

        # Determine weights based on characteristics
        if freq_measure > 1000 and noise_measure < 50:
            weights = (0.4, 0.4, 0.2)  # ml, tenengrad, energy
        elif freq_measure < 500:
            weights = (0.3, 0.3, 0.4)
        elif noise_measure > 100:
            weights = (0.2, 0.3, 0.5)
        else:
            weights = (0.4, 0.4, 0.2)

        # Calculate only needed metrics
        scores = [
            self._measure_modified_laplacian(gray),
            self._measure_tenengrad(gray),
            self._measure_energy_gradient(gray)
        ]

        return sum(score * weight for score, weight in zip(scores, weights))

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Optimized noise estimation"""
        noise_kernel = np.array([[1, -2, 1],
                               [-2, 4, -2],
                               [1, -2, 1]])
        noise = cv.filter2D(gray.astype(float), -1, noise_kernel)
        return np.median(np.abs(noise - np.median(noise)))


def normalize_score(score: float, max_score: float = 1000) -> float:
    """Normalize score to 0-100 range"""
    return max(0.0, min(100.0, 100 * score / max_score))


async def analyze_frames(websocket: Any) -> None:
    """Main WebSocket handler"""
    analyzer = ImageAnalyzer()
    logging.info("New client connected")

    try:
        async for message in websocket:
            try:
                logging.info("Received message of length: %i", len(message))
                data = json.loads(message)
                frame_batch = data['frames']
                raw_metric = data['metric']
                logging.info("Processing batch of %i frames", len(frame_batch))
                logging.info("Received data for metric: %s", raw_metric)

                if not raw_metric or not frame_batch:
                    logging.error(
                        "Invalid data received. Missing metric or frames."
                        )
                    await websocket.send(json.dumps(
                        {"error": "Invalid data. Missing metric or frames."}
                        ))
                    continue

                metric_type = MetricType.from_string(raw_metric)
                results = await analyzer.process_batch(frame_batch, metric_type)

                logging.info("Sending back results: %s", results)
                await websocket.send(json.dumps(results))

            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON received: {e}")
                await websocket.send(json.dumps({
                    "error": "Invalid JSON data"}))

            except Exception as e:
                logging.error("Unexpected error processing message: %s", e)
                logging.error(traceback.format_exc())
                await websocket.send(json.dumps(
                    {"error": "An unexpected error occurred"}
                    ))

    except websockets.exceptions.ConnectionClosed as e:
        logging.info(
            "Client disconnected. Code: %s, Reason: %s",
            e.code,
            e.reason
            )
    except Exception as e:
        logging.error("Unexpected error in WebSocket connection: %s", e)
        logging.error(traceback.format_exc())
    finally:
        logging.info("WebSocket connection closed")
        await websocket.close()


async def main():
    """
    Main function to start the WebSocket server.
    Handles server setup and exception handling
    """
    try:
        server = await websockets.serve(analyze_frames, "0.0.0.0", 8765)
        print("WebSocket server is running on wss://localhost:8765")
        await server.wait_closed()
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped")
