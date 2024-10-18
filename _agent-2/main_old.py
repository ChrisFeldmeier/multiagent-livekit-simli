import os
import asyncio

from livekit.agents import JobContext, WorkerOptions, cli, JobProcess
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, silero, cartesia, openai

from dotenv import load_dotenv

load_dotenv()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="You are a voice assistant. Please only speak if I am calling you by your name Mark",
            )
        ]
    )

    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(
            base_url="https://api.cerebras.ai/v1",
            api_key=os.environ.get("CEREBRAS_API_KEY"),
            model="llama3.1-8b",
        ),
        tts=cartesia.TTS(voice="384b625b-da5d-49e8-a76d-a2855d4f31eb"),
        chat_ctx=initial_ctx,
    )   

    await ctx.connect()
    assistant.start(ctx.room)
    await asyncio.sleep(1)
    await assistant.say("Hi there, how are you doing today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name="mark",
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            port=8082,
        )
    )
