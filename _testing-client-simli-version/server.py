from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import asyncio
import uvicorn

app = FastAPI()


async def GetDecodeOutput(
    websocket: WebSocket, decodeProcess: asyncio.subprocess.Process
):
    while True:
        data = await decodeProcess.stdout.read(6000)
        if not data:
            break
        await websocket.send_bytes(data)


@app.websocket("/")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        decodeTask = await asyncio.subprocess.create_subprocess_exec(
            *[
                "ffmpeg",
                "-i",
                "pipe:0",
                "-f",
                "s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-",
            ],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        sendTask = asyncio.create_task(GetDecodeOutput(websocket, decodeTask))
        while (
            websocket.client_state == WebSocketState.CONNECTED
            and websocket.application_state == WebSocketState.CONNECTED
        ):
            data = await websocket.receive_bytes()
            decodeTask.stdin.write(data)
    except WebSocketDisconnect:
        pass
    finally:
        decodeTask.stdin.close()
        await decodeTask.wait()
        await sendTask
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, port=8080)
