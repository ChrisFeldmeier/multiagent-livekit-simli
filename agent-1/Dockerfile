# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.6
FROM python:${PYTHON_VERSION}-slim 

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# Install gcc and other build dependencies.
RUN apt-get update && \
    apt-get install -y \
    gcc \
    python3-dev \
    unzip \
    wget \
    procps \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    gstreamer1.0-plugins-base \
    libgstreamer1.0-0 \
    && rm -rf /var/lib/apt/lists/*

USER root

USER appuser

RUN mkdir -p /home/appuser/.cache
RUN chown -R appuser /home/appuser/.cache

WORKDIR /home/appuser

COPY requirements.txt .
RUN python -m pip install --user --no-cache-dir -r requirements.txt

COPY . .

#Environment Variablen
ENV LIVEKIT_URL=wss://warpme-a9huji5k.livekit.cloud
ENV LIVEKIT_API_KEY=APIH384486MDq43
ENV LIVEKIT_API_SECRET=edNzuEpeEWFAfdXv0zzZGaQ4xucN2LzLoDiZ28nG6e9B

ENV DEEPGRAM_API_KEY=cd8cc4ac0a9f07a084808dfcb01624393e4738f8
ENV CARTESIA_API_KEY=755f9ff5-d7a5-4758-850a-856352d26414
ENV CEREBRAS_API_KEY=csk-fnm6jre49fr9cvhtxe2knmcpd9h6jdr3em6mr283rcmd9ftd

ENV SIMLI_API_KEY=14kg6yd4wjxlm5se40ei1
ENV SIMLI_FACE_ID=00b344ed-8757-42a3-b37b-6fa53588b7e7

# Application port
EXPOSE 8081  

# API, WebSocket
#EXPOSE 7880

# ICE/UDP range for WebRTC
#EXPOSE 50000-60000 

# ICE/TCP
#EXPOSE 7881 

# ICE/UDP Mux (optional)
#EXPOSE 7882   

# TURN/TLS (optional)
#EXPOSE 5349  

# TURN/UDP (optional)
#EXPOSE 3478   

RUN python main.py download-files

# vad fix noch überarbeiten, sonst läuft docker nicht - geht in neuer version nicht
#RUN rm -rf /home/appuser/.cache/torch/hub/snakers4_silero-vad_master/* && \
#    wget -qO- https://github.com/Baulig-Consulting-GmbH/voice_vad/zipball/master > /home/appuser/.cache/torch/hub/snakers4_silero-vad_master/voice_vad.zip && \
#    unzip /home/appuser/.cache/torch/hub/snakers4_silero-vad_master/voice_vad.zip -d /home/appuser/.cache/torch/hub/snakers4_silero-vad_master/ && \
#    mv /home/appuser/.cache/torch/hub/snakers4_silero-vad_master/Baulig-Consulting-GmbH-voice_vad-*/* /home/appuser/.cache/torch/hub/snakers4_silero-vad_master/ && \
#    rm -rf /home/appuser/.cache/torch/hub/snakers4_silero-vad_master/Baulig-Consulting-GmbH-voice_vad-* /home/appuser/.cache/torch/hub/snakers4_silero-vad_master/voice_vad.zip

# Run the application.
ENTRYPOINT ["python", "main.py"]
CMD ["connect", "--room", "warp_demo"]