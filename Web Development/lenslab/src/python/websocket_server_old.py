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
import numpy as np
from PIL import Image

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


async def analyze_frames(websocket: Any) -> None:
    logging.info("New client connected")
    try:
        async for message in websocket:
            try:
                logging.info("Received message of length: %i", len(message))
                data = json.loads(message)
                frame_batch = data['frames']
                selected_metric = data['metric']
                logging.info("Data: %f", data)
                logging.info("Processing batch of %i frames", len(frame_batch))
                logging.info("Received data for metric: %s", selected_metric)

                if not selected_metric or not frame_batch:
                    logging.error(
                        "Invalid data received. Missing metric or frames."
                        )
                    await websocket.send(json.dumps(
                        {"error": "Invalid data. Missing metric or frames."}
                        ))
                    continue

                averaged_frames = average_frames(frame_batch)
                analysis_results = analyze_averaged_frames(
                    averaged_frames,
                    selected_metric
                    )
                logging.info("Sending back results: %s", analysis_results)
                await websocket.send(json.dumps(analysis_results))

            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON received: {e}")
                await websocket.send(json.dumps({"error": "Invalid JSON data"}))

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


def average_frames(
        frame_batch: List[List[Dict[str, Any]]]
        ) -> List[np.ndarray]:
    """
    Asynchronous function to handle WebSocket connections.
    Receives frame batches, averages them, and sends back analysis results.

    Args:
    websocket: The WebSocket connection object
    """
    averaged_frames = []
    for roi_index in range(len(frame_batch[0])):
        roi_frames = [
            np.array(Image.open(BytesIO(b64decode(
                frame[roi_index]['dataUrl'].split(',')[1])))
                ) for frame in frame_batch
            ]
        averaged_frame = np.mean(roi_frames, axis=0).astype(np.uint8)
        averaged_frames.append(averaged_frame)
    return averaged_frames


def analyze_averaged_frames(
        averaged_frames: List[np.ndarray],
        metric: str) -> List[Dict[str, Any]]:
    """
    Analyzes the averaged frames using the specified quality metric.
    """
    logging.info("Analyzing frames with metric: %s", metric)
    results = []
    for i, frame in enumerate(averaged_frames):
        logging.info(
            "Processing frame {i}, shape: %s, dtype: %s",
            frame.shape,
            frame.dtype
            )

        # Convert to lowercase for case-insensitive comparison
        # TODO: Add replace to handle metrics with spaces
        cleaned_metric = metric.strip('"').lower().replace(" ", "_")
        logging.info("Cleaned metric: %s", cleaned_metric)

        if cleaned_metric in ["sharpness", "shaprness"]:
            method_time, score = measure_performance(measure_sharpness, frame)
            logging.info(
                "Sharpness score for frame %i: %f, time: %f",
                i,
                score,
                method_time
                )
        elif cleaned_metric == 'contrast':
            score = measure_contrast(frame)
            logging.info("Contrast score for frame %i: %f", i, score)
        elif cleaned_metric == 'brightness':
            score = measure_brightness(frame)
            logging.info("Brightness score for frame %i: %f", i, score)
        # TODO: Add branch 
        else:
            logging.warning("Unknown metric: %s. Using default score.", metric)
            score = 1.0

        results.append({'roi': i, 'quality_score': float(score)})

    logging.info("Analysis results: %s", results)
    return results


def measure_sharpness(image: np.ndarray) -> float:
    """
    Measures the sharpness of an image using the Laplacian variance method or a fallback method.
    """
    if cv is not None:
        logging.info("Using CV Method")
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return max(1.0, cv.Laplacian(gray, cv.CV_64F).var())

    # Fallback method: use numpy to calculate a simple edge detection
    logging.info("Using fallback method")
    gray = np.mean(image, axis=2)
    edges_x = np.diff(gray, axis=1)
    edges_y = np.diff(gray, axis=0)
    edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
    return max(1.0, np.mean(edge_magnitude))


def measure_contrast(image: np.ndarray) -> float:
    """
    Measures the contrast of an image using standard deviation.
    """
    gray = np.mean(image, axis=2) if image.ndim == 3 else image
    return max(1.0, np.std(gray))


def measure_brightness(image: np.ndarray) -> float:
    """
    Measures the brightness of an image using the mean pixel value.
    """
    return max(1.0, np.mean(image))


def measure_performance(method: object, inner_image: np.ndarray) -> float:
    """Simple function that calculates the time taken for a function to run"""
    start_time = time.time()
    performance_result = method(inner_image)
    end_time = time.time()
    return end_time - start_time, performance_result


def gradient_magnitude(image):
    """Calculates the average magnitude of image gradients"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    mag = cv.magnitude(gx, gy)
    return np.mean(mag)


def tenengrad(image):
    """Calculates the sum of the squared gradient magnitudes"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    return np.mean(gx**2 + gy**2)


def brenner_focus(image):
    """Calculates the sum of squared differences between adjacent pixels"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    diff = gray[:, 2:] - gray[:, :-2]
    return np.mean(diff**2)


async def main():
    """
    Main function to start the WebSocket server.
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
