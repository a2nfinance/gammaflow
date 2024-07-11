import asyncio
from websockets.server import serve
import subprocess

async def run_command(websocket):
    async for message in websocket:
        process = subprocess.Popen(message, stdout=subprocess.PIPE, pipesize=2048, shell=True)
        for c in iter(lambda: process.stdout.readline(), b""):
            await websocket.send(c)

        await websocket.send(b"\n--end-process--")

async def main():
    async with serve(run_command, "0.0.0.0", 5000):
        await asyncio.Future()  # run forever

asyncio.run(main())