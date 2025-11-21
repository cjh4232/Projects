import asyncio
import websockets
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('websockets')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


async def echo(websocket, path):
    try:
        logging.info(f"New connection from {websocket.remote_address}")
        async for message in websocket:
            ogging.info(f"Received message: {message}")
            await websocket.send(f"Echo: {message}")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.error(f"Connection closed unexpectedly: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


async def main():
    server = await websockets.serve(echo, "localhost", 8765)
    logging.info("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
