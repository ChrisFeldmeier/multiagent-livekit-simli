# Insanely fast AI voice assistant in 50 LOC

This repo contains everything you need to run your own AI voice assistant that responds to you in less than 500ms.

It uses:
- 🌐 [LiveKit](https://github.com/livekit) transport
- 👂 [Deepgram](https://deepgram.com/) STT
- 🧠 [Cerebras](https://inference.cerebras.ai/) LLM
- 🗣️ [Cartesia](https://cartesia.ai/) TTS

## Run the assistant

1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `cp .env.example .env`
5. add values for keys in `.env`
6. `python main.py dev`

## build docker file
`docker build -t agent-1 .`

## run docker build
`docker run -t agent-1`

python main.py connect --room warp_demo

## Run a client

1. Choose the same LiveKit Cloud project you used in the agent's `.env` and click `Connect`


export LIVEKIT_URL=wss://warpme-a9huji5k.livekit.cloud
export LIVEKIT_API_KEY=APIH384486MDq43
export LIVEKIT_API_SECRET=edNzuEpeEWFAfdXv0zzZGaQ4xucN2LzLoDiZ28nG6e9B
export DEEPGRAM_API_KEY=cd8cc4ac0a9f07a084808dfcb01624393e4738f8
export CARTESIA_API_KEY=755f9ff5-d7a5-4758-850a-856352d26414
export CEREBRAS_API_KEY=csk-fnm6jre49fr9cvhtxe2knmcpd9h6jdr3em6mr283rcmd9ftd
export SIMLI_API_KEY=14kg6yd4wjxlm5se40ei1
export SIMLI_FACE_ID=00b344ed-8757-42a3-b37b-6fa53588b7e7