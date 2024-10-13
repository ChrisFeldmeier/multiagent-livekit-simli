import os
import asyncio
import logging
import aiohttp
import numpy as np
import time
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, silero, cartesia, openai
from livekit import rtc
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamTrack
)
from av import VideoFrame, AudioFrame
from aiortc.contrib.media import MediaRelay
from dotenv import load_dotenv
import cv2

load_dotenv()

# Set global logging level to DEBUG for more comprehensive logs
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  # Create a logger for this module

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

class SilentAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()  # Initialize base class
        self.sample_rate = 16000
        self.channels = 1
        self.frame_duration = 0.1  # 100ms
        self.samples_per_frame = int(self.sample_rate * self.frame_duration)
        self.silence = np.zeros(self.samples_per_frame, dtype=np.int16)
        self.start_time = time.time()
        self.next_frame_time = self.start_time
        self.frame_count = 0

    async def recv(self):
        current_time = time.time()
        if current_time < self.next_frame_time:
            await asyncio.sleep(self.next_frame_time - current_time)
        else:
            self.next_frame_time += self.frame_duration

        # Create an AudioFrame with silence
        frame = AudioFrame.from_ndarray(self.silence, format="s16")
        return frame

class BlackVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, width=512, height=512):
        super().__init__()
        self.width = width
        self.height = height
        self.start_time = time.time()
        self.frame_interval = 1 / 30  # 30 FPS
        self.next_frame_time = self.start_time

    async def recv(self):
        current_time = time.time()
        if current_time < self.next_frame_time:
            await asyncio.sleep(self.next_frame_time - current_time)
        else:
            self.next_frame_time += self.frame_interval

        # Create a black video frame
        frame = VideoFrame.from_ndarray(
            np.zeros((self.height, self.width, 3), dtype=np.uint8), format="bgr24"
        )
        return frame

async def entrypoint(ctx: JobContext):
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are a voice assistant. Pretend we're having a human conversation, "
                    "no special formatting or headings, just natural speech. "
                    "Only say something if I call you with your name John. If not please don't say anything."
                ),
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
        tts=cartesia.TTS(voice="248be419-c632-4f23-adf1-5324ed7dbf1d"),
        chat_ctx=initial_ctx,
    )

    await ctx.connect()
    assistant.start(ctx.room)
    room = ctx.room

    # Simli API key and Face ID from environment variables
    SIMLI_API_KEY = os.environ.get("SIMLI_API_KEY")
    SIMLI_FACE_ID = os.environ.get("SIMLI_FACE_ID")

    if not SIMLI_API_KEY or not SIMLI_FACE_ID:
        logger.error("SIMLI_API_KEY or SIMLI_FACE_ID are not set in the environment variables.")
        return

    # Start the audio-to-video session and get the session_token
    session_token = await start_audio_to_video_session(SIMLI_API_KEY, SIMLI_FACE_ID)
    if not session_token:
        logger.error("Error obtaining session_token from Simli.")
        return

    # Define ICE servers as RTCIceServer objects
    ice_servers = [
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ]

    # Create RTCConfiguration object
    config = RTCConfiguration(iceServers=ice_servers)

    # Create a PeerConnection with RTCConfiguration
    pc = RTCPeerConnection(configuration=config)
    pcs = set()
    pcs.add(pc)

    # Create a DataChannel
    dc = pc.createDataChannel("datachannel")

    # Send the session_token over the DataChannel
    @dc.on("open")
    async def on_open():
        logger.info("DataChannel is open")
        dc.send(session_token)
        logger.info(f"Session token sent: {session_token}")

        # Important: Send something to Simli to receive audio & video data
        dc.send((0).to_bytes(1, "little") * 6000)

        logger.info("Now receiving audio & video")

    @dc.on("message")
    def on_message(message):
        logger.info(f"Message received on DataChannel: {message}")

    # Initialize the MediaRelay before use
    relay = MediaRelay()

    # Register event handler before setting the remote description
    @pc.on("track")
    def on_track(track):
        logger.info(f"Track {track.kind} received")
        logger.debug(f"Track Details: {track}")

        if track.kind == "video":
            try:
                # Subscribe to the incoming video track
                relayed_video = relay.subscribe(track)
                if relayed_video is None:
                    logger.error("relayed_video is None after subscription")
                else:
                    logger.debug("Successfully subscribed to video track")
            except Exception as e:
                logger.error(f"Error subscribing to video track: {e}")
                return

            # Create a VideoSource for LiveKit
            WIDTH = 512  # Adjust as needed
            HEIGHT = 512
            source = rtc.VideoSource(WIDTH, HEIGHT)
            logger.debug("VideoSource for LiveKit created")

            async def forward_frames():
                logger.info("Starting forward_frames loop")
                while True:
                    try:
                        # Receiving the frame from Simli
                        frame = await relayed_video.recv()

                        if frame:
                            logger.info(f"Video frame received from Simli: {frame.width}x{frame.height}")

                            # Convert the frame to an ndarray (YUV format)
                            ndarray_yuv = frame.to_ndarray(format="yuv420p")

                            if ndarray_yuv is not None:
                                logger.debug(f"Frame-NDArray Shape (YUV): {ndarray_yuv.shape}")

                                # Convert YUV to RGB using OpenCV
                                ndarray_rgb = cv2.cvtColor(ndarray_yuv, cv2.COLOR_YUV2RGB_I420)
                                logger.debug(f"Frame-NDArray Shape (RGB): {ndarray_rgb.shape}")

                                # Initialize the LiveKit VideoFrame (without 'buffer_type')
                                livekit_frame = rtc.VideoFrame(
                                    width=frame.width,
                                    height=frame.height,
                                    type=rtc.VideoBufferType.RGB24,  # Specify the RGB format type
                                    data=ndarray_rgb.tobytes()  # Convert ndarray to bytes
                                )

                                # Send the frame directly to LiveKit
                                source.capture_frame(livekit_frame)
                                logger.debug("VideoFrame successfully captured in VideoSource")
                            else:
                                logger.warning("Received frame NDArray is None")
                        else:
                            logger.debug("No video frame received")

                    except asyncio.CancelledError:
                        logger.info("forward_frames task was cancelled")
                        break
                    except Exception as e:
                        logger.error(f"Error processing video frame: {e}")

                    await asyncio.sleep(0)  # Yield control


            task_forward_frames = asyncio.create_task(forward_frames())
            logger.debug("forward_frames task started")

          # Create a LocalVideoTrack with the VideoSource
            try:
                livekit_video_track = rtc.LocalVideoTrack.create_video_track("simli-video", source)
                logger.debug("LocalVideoTrack successfully created")
            except Exception as e:
                logger.error(f"Error creating LocalVideoTrack: {e}")
                return

            # Publish the track in the LiveKit room
            async def publish_video_track():
                try:
                    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
                    publication = await room.local_participant.publish_track(livekit_video_track, options)
                    logger.info(
                        "Simli video track published",
                        extra={"track_sid": publication.sid},
                    )
                except Exception as e:
                    logger.error(f"Error publishing video track: {e}")

            task_publish_video = asyncio.create_task(publish_video_track())
            logger.debug("publish_video_track task started")

        elif track.kind == "audio":
            try:
                # Subscribe to the incoming audio track
                relayed_audio = relay.subscribe(track)
                if relayed_audio is None:
                    logger.error("relayed_audio is None after subscription")
                else:
                    logger.debug("Successfully subscribed to audio track")
            except Exception as e:
                logger.error(f"Error subscribing to audio track: {e}")
                return

            # Create an AudioSource for LiveKit
            SAMPLE_RATE = 48000  # Adjust as needed
            NUM_CHANNELS = 2
            audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
            logger.debug("AudioSource for LiveKit created")

        async def forward_audio():
            logger.debug("Starting forward_audio loop")
            while True:
                try:
                    frame = await relayed_audio.recv()
                    if frame:
                        logger.info("Audio frame received from Simli")
                        
                        # Extract the audio data directly from the frame and forward it
                        audio_data = frame.planes[0].to_bytes()

                        # LiveKit expects an AudioFrame with raw audio data, sample rate, and number of channels
                        livekit_frame = rtc.AudioFrame(
                            samples=audio_data,
                            sample_rate=frame.sample_rate,
                            num_channels=frame.layout.channels
                        )

                        # Set the appropriate timestamps
                        livekit_frame.pts = frame.pts
                        livekit_frame.time_base = frame.time_base

                        # Send the frame directly to LiveKit
                        source.capture_frame(livekit_frame)
                        logger.debug("AudioFrame successfully captured in AudioSource")
                except asyncio.CancelledError:
                    logger.info("forward_audio task was cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing audio frame: {e}")

                await asyncio.sleep(0)  # Yield control


            task_forward_audio = asyncio.create_task(forward_audio())

            logger.debug("forward_audio task started")

            # Create a LocalAudioTrack with the AudioSource
            try:
                livekit_audio_track = rtc.LocalAudioTrack.create_audio_track("simli-audio", audio_source)
                logger.debug("LocalAudioTrack successfully created")
            except Exception as e:
                logger.error(f"Error creating LocalAudioTrack: {e}")
                return

            # Publish the audio track in the LiveKit room
            async def publish_audio_track():
                try:
                    publication = await room.local_participant.publish_track(livekit_audio_track)
                    logger.info(
                        "Simli audio track published",
                        extra={"track_sid": publication.sid},
                    )
                except Exception as e:
                    logger.error(f"Error publishing audio track: {e}")

            task_publish_audio = asyncio.create_task(publish_audio_track())
            logger.debug("publish_audio_track task started")

    # Use transceivers with sendrecv for audio and video
    logger.debug("Creating transceivers with sendrecv directions")

    # Add transceivers with the corresponding tracks
    silent_audio = SilentAudioTrack()
    audio_transceiver = pc.addTransceiver(silent_audio, direction="sendrecv")
    if audio_transceiver.sender:
        logger.debug("Audio transceiver with SilentAudioTrack successfully added")

    black_video = BlackVideoTrack(width=512, height=512)
    video_transceiver = pc.addTransceiver(black_video, direction="sendrecv")
    if video_transceiver.sender:
        logger.debug("Video transceiver with BlackVideoTrack successfully added")

    # Wait until ICE gathering is complete before creating the offer
    async def gather_ice_candidates():
        try:
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            logger.debug(f"Local SDP offer:\n{pc.localDescription.sdp}")
            # Wait until ICE gathering is complete
            while pc.iceGatheringState != 'complete':
                await asyncio.sleep(0.1)
            logger.info("ICE gathering complete")
        except Exception as e:
            logger.error(f"Error during ICE gathering: {e}")

    await gather_ice_candidates()

    # Now that ICE gathering is complete, send the offer
    offer = pc.localDescription
    logger.debug(f"Local SDP offer:\n{offer.sdp}")

    # Send the offer to Simli and receive the answer
    answer_sdp = await start_webrtc_session(
        offer.sdp,
        offer.type,
        SIMLI_API_KEY,
        session_token,
    )
    if not answer_sdp:
        logger.error("Error receiving SDP answer from Simli.")
        return

    answer = RTCSessionDescription(sdp=answer_sdp, type="answer")

    # Set the remote description
    await pc.setRemoteDescription(answer)
    logger.info(f"Remote SDP answer:\n{pc.remoteDescription.sdp}")

    # Monitor the ICE connection state
    @pc.on("iceconnectionstatechange")
    async def on_ice_connection_state_change():
        logger.info(f"ICE connection state is {pc.iceConnectionState}")
        if pc.iceConnectionState == "connected":
            logger.info("ICE connection successfully established")
        elif pc.iceConnectionState == "completed":
            logger.info("ICE connection completed")
        elif pc.iceConnectionState == "failed":
            logger.error("ICE connection failed")
        elif pc.iceConnectionState == "disconnected":
            logger.warning("ICE connection disconnected")

    # Keep the application running
    await asyncio.Event().wait()

async def start_audio_to_video_session(api_key, face_id):
    url = "https://api.simli.ai/startAudioToVideoSession"
    payload = {
        "faceId": face_id,
        "isJPG": False,
        "apiKey": api_key,
        "syncAudio": True,
    }

    headers = {"Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response_text = await response.text()
            if response.status == 200:
                res_json = await response.json()
                session_token = res_json.get("session_token")
                logger.debug(f"Received session_token: {session_token}")
                return session_token
            else:
                logger.error(f"Error starting audio-to-video session: {response.status} - {response_text}")
                return None

async def start_webrtc_session(offer_sdp, offer_type, api_key, session_token):
    url = "https://api.simli.ai/StartWebRTCSession"
    payload = {
        "sdp": offer_sdp,
        "type": offer_type,
        "apiKey": api_key,
        "session_token": session_token,
    }
    headers = {"Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response_text = await response.text()
            if response.status == 200:
                res_json = await response.json()
                answer_sdp = res_json.get("sdp")
                logger.debug(f"Received answer SDP:\n{answer_sdp}")
                return answer_sdp
            else:
                logger.error(f"Error starting WebRTC session: {response.status} - {response_text}")
                return None

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            agent_name="john", entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, port=8083
        )
    )
