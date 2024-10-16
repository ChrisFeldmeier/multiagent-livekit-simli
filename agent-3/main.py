import os
import asyncio
import logging
import aiohttp
import numpy as np
import time
import base64
import json
import struct
import ctypes
import wave
import livekit
from livekit.agents import JobContext, WorkerOptions, cli, JobProcess
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, silero, cartesia, openai
from livekit import rtc
#from utils import audio, shortuuid
from livekit.agents import tokenize, tts, utils
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamTrack
)
from av import VideoFrame, AudioFrame
from aiortc.contrib.media import MediaRelay
from aiortc import MediaStreamTrack
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack
from dotenv import load_dotenv
import cv2
from scipy.signal import resample
import asyncio
from typing import AsyncIterable, AsyncGenerator
from livekit.plugins.cartesia import TTS as CartesiaTTS  # Ensure this import is correct
import uuid

load_dotenv()

# Set global logging level to DEBUG for more comprehensive logs
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)  # Create a logger for this module

dc = None

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


class CustomSynthesizeStream(tts.SynthesizeStream):
    def __init__(self, opts, session):
        super().__init__()  # Ensure proper initialization
        self.opts = opts
        self.session = session
        self._closed = False  # Track the closed state

    def _check_not_closed(self):
        """Check if the stream is closed and raise an error if it is."""
        if self._closed:
            raise RuntimeError(f"{self.__class__.__module__}.{self.__class__.__name__} is closed")

    async def _main_task(self):
        """Implement the main task logic here."""
        self._check_not_closed()
        #logging.info("Running _main_task in CustomSynthesizeStream")
        await self.start_processing()

    async def start_processing(self):
        """Start processing audio frames from the event channel."""
        self._check_not_closed()
        #logging.info("Starting audio frame processing")
        consumer_task = asyncio.create_task(self._consume_event_ch())
        await consumer_task

    async def _consume_event_ch(self):
        """Consume and process audio frames from the event channel."""
        #logging.info(f"Next Start Processing audio")
        audio_event = await self._event_ch.get()
        #logging.info(f"Next END ")
        while not self._closed:
            audio_event = await self._event_ch.get()  # Access the event channel from the superclass
            try:
                # Process the audio event
                #logging.info(f"Processing audio event: {audio_event}")
                continue
                # Add custom processing logic here
            finally:
                self._event_ch.task_done()

    def modify_audio_frame(self, frame: bytes) -> bytes:
        """Modify the audio frame."""
        self._check_not_closed()
        #logging.info("Modifying audio frame")
        return frame  # Return the modified frame

    def close(self):
        """Close the stream and mark it as closed."""
        self._closed = True
        #logging.info("CustomSynthesizeStream is now closed")


class CustomTTS(CartesiaTTS):
    def __init__(self, *args, **kwargs):
        # Call the parent class constructor to ensure all attributes are initialized
        super().__init__(*args, **kwargs)
        self.dc = None

    def set_data_channel(self, dc):
        """Set the data channel for sending audio frames."""
        self.dc = dc
        #logging.info("Data channel set")

    def stream(self) -> CustomSynthesizeStream:
        #logging.info("CustomTTS stream called")

        
        stream = CustomSynthesizeStream(self._opts, self._ensure_session())
        #logging.info("CustomTTS stream called")
        #stream.start_processing()
        #stream.close()

        return stream


            # Send the audio data in chunks
            #for i in range(0, len(original_stream), 6000):
            #    dc.send(original_stream[i: i + 6000]) # send audio stream to simli
            #    #logging.info(f"Sent audio chunk {i // 6000 + 1}")
            #return original_stream


class CustomAudioStreamTrack(MediaStreamTrack):
                kind = "audio"

                def __init__(self, audio_stream):
                    super().__init__()  # Initialize the base class
                    self.audio_stream = audio_stream

                async def recv(self):
                    # Get the next audio frame from the audio stream
                    frame = await self.audio_stream.recv()
                    return frame

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

# New class to send TTS output over WebRTC as audio
class TTSAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, tts_audio_data, sample_rate=16000, channels=1):
        super().__init__()  # Initialize the base class
        self.audio_data = tts_audio_data  # Audio data from Cartesia TTS
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration = 0.1  # 100ms per frame
        self.samples_per_frame = int(self.sample_rate * self.frame_duration)
        self.current_position = 0

    async def recv(self):
        if self.current_position >= len(self.audio_data):
            return None  # End of audio data

        # Create a frame from the current position
        frame_data = self.audio_data[self.current_position:self.current_position + self.samples_per_frame]
        self.current_position += self.samples_per_frame

        # Convert the byte data into a WebRTC AudioFrame
        frame = AudioFrame.from_ndarray(np.frombuffer(frame_data, dtype=np.int16), format="s16")
        return frame

async def entrypoint(ctx: JobContext):
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are a voice assistant. Pretend we're having a human conversation, "
                    "no special formatting or headings, just natural speech. "
                    "Only say something if I call you with your name Albert."
                ),
            )
        ]
    )

    # TTS String
    def _before_tts_cb(assistant, text):
        if isinstance(text, str):
            return str.replace("livekit", "LiveKit")
        else:
            async def _iterate_str():
                async for chunk in text:
                    yield "" #chunk.replace("word_may_be_partial_here", "")  #need this to accpet TTS
            return _iterate_str()
   
        # return tokenize.utils.replace_words(
        #     text=plain_text, replacements={"{": "", "}": ""}
        # )

    #def _before_tts(agent: VoiceAssistant, text: str | AsyncIterable[str]):
    #    if isinstance(text, str):
    #        print("SPEAKING TEXT:")
    #        print(str)
    #        return str.replace("livekit", "LiveKit")
    #    else:
    #        async def _iterate_str():
    #            async for chunk in text:
    #                yield chunk.replace("word_may_be_partial_here", "")
    #        
    #        return _iterate_str()

    custom_tts = cartesia.TTS(voice="a0e99841-438c-4a64-b679-ae501e7d6091", sample_rate=16000)
    #custom_tts._ensure_session
    assistant = VoiceAssistant(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(
            base_url="https://api.cerebras.ai/v1",
            api_key=os.environ.get("CEREBRAS_API_KEY"),
            model="llama3.1-8b",
        ),
        tts=custom_tts,
        chat_ctx=initial_ctx,
        before_tts_cb=_before_tts_cb
    )



    async def get_audio_frames(tts: cartesia.TTS) -> AsyncGenerator[AudioFrame, None]:
        async for chunk in tts.stream():
            # Convert chunk to AudioFrame
            #logging.info("receive cartesia audio stream chunk")
            audio_frame = AudioFrame(
                data=chunk,
                sample_rate=tts.sample_rate,
                num_channels=1,  # Assuming mono
                samples_per_channel=len(chunk) // 2  # Assuming 16-bit audio
            )
            yield audio_frame
    #asyncio.create_task(get_audio_frames(custom_tts))


    def process_audio_stream(audio_stream, speaker_name):
        async def audio_processing_task():
            async for frame_event in audio_stream:
                try:
                    # Access the audio frame from the event
                    frame = frame_event.frame
                    # Process the audio frame

                    ### kann wsl. später entfernt werden, weil nicht genutzt
                    
                    print(f"New Audio frame received from {speaker_name}")
                    # Add your audio processing logic here
                except Exception as e:
                    logging.error(f"Error processing audio frame: {e}")
                    break
        # Start the audio processing task
        asyncio.create_task(audio_processing_task())
                # audio_stream is an async iterator that yields AudioFrame
                # here you should send the audio stream to simli 


    #await room.local_participant.publish_data("hello world")

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

        # Just a Test here sending a local WAV file to the avatar -> choose the file format
        # file_path = "input-audio-longer.wav" 
        # # Ensure the file exists
        # if not os.path.exists(file_path):
        #     logging.error(f"File {file_path} does not exist.")
        # else:
        #     # Test Speaking
        #     # Open the WAV file
        #     with wave.open(file_path, 'rb') as wav_file:
        #         # Check if the file is in PCM Int16 format
        #         if wav_file.getsampwidth() != 2 or wav_file.getcomptype() != 'NONE':
        #             logging.error("The WAV file is not in PCM Int16 format.")
        #         elif wav_file.getnchannels() != 1:
        #             logging.error("The WAV file is not mono.")
        #         else:
        #             # Read the frames from the WAV file
        #             audio_data = wav_file.readframes(wav_file.getnframes())

        #             # Since the file is already mono, directly convert to bytes
        #             mono_audio_data = audio_data

        #             # Send the audio data in chunks
        #             for i in range(0, len(mono_audio_data), 6000):
        #                 dc.send(mono_audio_data[i: i + 6000]) # send audio stream to simli
        #                 #logging.info(f"Sent audio chunk {i // 6000 + 1}")

        # logger.info("Now receiving audio & video")

    @dc.on("message")
    def on_message(message):
        logger.info(f"Message received on DataChannel: {message}")

    await ctx.connect()
    assistant.start(ctx.room)
    chat = rtc.ChatManager(ctx.room)
    room = ctx.room

    #logging.info("connected to room %s", room.name)
    #logging.info("participants: %s", room.remote_participants)

    #participant = room.remote_participants.get("Chris")
    
    #audio_stream = rtc.AudioStream.from_participant(
    #                participant=participant,
    #                track_source=rtc.TrackSource.MICROPHONE,
    #                sample_rate=18000,
    #                num_channels=1,
    #            )

    # if a speaker is changing
    @room.on("active_speakers_changed")
    def on_active_speakers_changed(speakers: list[rtc.Participant]):
        #logging.info("active speakers changed: %s", speakers)

    # what happens in the room
    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        #logging.info("Track subscribed: %s", publication.sid)
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            _video_stream = rtc.VideoStream(track)
            #logging.info("track video subscribed!")
            #logging.info("participant name "+participant.name)
            #logging.info("participant name "+participant.identity)
            
            # video_stream is an async iterator that yields VideoFrame
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            print("Subscribed to an Audio Track")
            
            # Access the audio stream
            audio_stream = rtc.AudioStream(track)
            #logging.info("Audio stream accessed!")
            
            # Identify the speaker
            speaker_name = participant.name
            speaker_identity = participant.identity
            #logging.info(f"Audio track from participant: {speaker_name} ({speaker_identity})")

            # if speaker_name == "Chris":
            #     async def process_audio_stream():
            #         async for frame_event in audio_stream:
            #             try:
            #                 # Access the audio frame from the event
            #                 frame = frame_event.frame
            #                 #logging.info(f"New Audio track frame received from {speaker_name}")

            #                 # Convert the audio frame to bytes (Assuming it's in PCM 16-bit format)
            #                 audio_data = frame.to_bytes()

            #                 # Send the audio frame data to Simli using the DataChannel
            #                 for i in range(0, len(audio_data), 6000):
            #                     if dc and dc.readyState == "open":
            #                         dc.send(audio_data[i: i + 6000])
            #                         #logging.info(f"Sent audio chunk {i // 6000 + 1} to Simli")
            #                     else:
            #                         logging.warning("DataChannel is not open")
            #             except Exception as e:
            #                 logging.error(f"Error processing audio frame: {e}")
            #                 break

            #     # Start the async task
            #     asyncio.create_task(process_audio_stream())

            return audio_stream
            
            # Process the audio stream
            #process_audio_stream(audio_stream, speaker_name)

    # Initialize the MediaRelay before use
    relay = MediaRelay()


    async def get_audio_frames(text: str, tts: cartesia.TTS) -> AsyncGenerator[AudioFrame, None]:
        async for chunk in tts.synthesize(text):
            # Convert chunk to AudioFrame
            #logging.info("FRAME: Cartesia")
            audio_frame = AudioFrame(
                data=chunk,
                sample_rate=tts.sample_rate,
                num_channels=1,  # Assuming mono
                samples_per_channel=len(chunk) // 2  # Assuming 16-bit audio
            )
            yield audio_frame

    async def answer_from_text(txt: str):
        #logging.info(txt)
        audio_data = None
        
        # Synthesize the text
        tts_synth = custom_tts.synthesize(txt)
        all_frames = []
        async for audio_frame in tts_synth:
            # If audio_data is not directly bytes, convert it
            audio_data = audio_frame.frame.data
            all_frames.append(audio_data)
        
        all_audio_data = b''.join(all_frames)
        # Send the audio data in chunks over the data channel
        for i in range(0, len(all_audio_data), 6000):
            chunk = all_audio_data[i: i + 6000]
            if dc and dc.readyState == "open":
                dc.send(chunk)
                #logging.info(f"Sent audio chunk {i // 6000 + 1} to data channel")
            else:
                logging.warning("DataChannel is not open")

        chat_ctx = assistant.chat_ctx.copy()
        chat_ctx.append(role="user", text=txt)
        #stream = assistant.llm.chat(chat_ctx=chat_ctx)
        #await assistant.say(stream)

    @chat.on("message_received")
    def on_chat_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(answer_from_text(msg.message))

    async def broadcast_event(local_participant, event_name, msg: ChatMessage):
        # Encode the event data into a format suitable for transmission
    
        #logging.info("MY MESSAGE COTNENT:")
        #logging.info(msg.content)
        
        tts_synth = custom_tts.synthesize(msg.content)
        all_frames = []
        async for audio_frame in tts_synth:
            # If audio_data is not directly bytes, convert it
            audio_data = audio_frame.frame.data
            all_frames.append(audio_data)
        
        all_audio_data = b''.join(all_frames)
        # Send the audio data in chunks over the data channel
        for i in range(0, len(all_audio_data), 6000):
            chunk = all_audio_data[i: i + 6000]
            if dc and dc.readyState == "open":
                dc.send(chunk)
                #logging.info(f"MESSAGE Sent audio chunk {i // 6000 + 1} to data channel")
            else:
                logging.warning("MESSAGE DataChannel is not open")

            #await local_participant.publish_data(
            #    payload=payload,
            #    reliable=True,  # Reliable transmission
            #    destination_identities=[],  # Broadcast to all participants
            ##    topic="speaking_state"
            #)
            #logging.info(f"MESSAGE Broadcasted event '{event_name}' with data: {msg}")
  

    def user_speech_committed(assistant, local_participant, user_msg: ChatMessage):
        asyncio.create_task(broadcast_event(local_participant, "user_speech_committed", user_msg))
        #logging.info(f"Event data: user_speech_committed")

    def agent_speech_committed(assistant, local_participant, msg: ChatMessage):
        asyncio.create_task(broadcast_event(local_participant, "agent_speech_committed", msg))
        #logging.info(f"Event data: agent_speech_committed")

    #assistant.on('user_speech_committed', lambda user_msg: user_speech_committed(assistant, ctx.room.local_participant, user_msg))
    assistant.on('agent_speech_committed', lambda msg: agent_speech_committed(assistant, ctx.room.local_participant, msg))
    
    # Function to send TTS audio to Simli
    async def send_tts_audio_to_simli(tts_text):
        # Generate TTS audio from Cartesia
        tts_audio_data = await assistant.tts.synthesize(tts_text)
        # Create an AudioSource for LiveKit
        SAMPLE_RATE = 16000  # or the rate provided by Cartesia
        NUM_CHANNELS = 1  # or the channel configuration provided by Cartesia
        tts_audio_track = TTSAudioTrack(tts_audio_data, sample_rate=SAMPLE_RATE, channels=NUM_CHANNELS)

        # Add transceiver and send audio
        #tts_audio_transceiver = pc.addTransceiver(tts_audio_track, direction="sendonly")
        #### no dc.send is the correct one

        # Publish the audio track to the LiveKit room
        try:
            livekit_audio_track = rtc.LocalAudioTrack.create_audio_track("tts-audio", tts_audio_track)
            await room.local_participant.publish_track(livekit_audio_track)
            logger.info("TTS audio track published")
        except Exception as e:
            logger.error(f"Error publishing TTS audio track: {e}")

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
                                frame = frame.to_rgb() #get direct the stream from frame

                                if frame:
                                    logger.info(f"Video frame received from Simli: {frame.width}x{frame.height}")

                                    # Assuming the frame is already in RGB format
                                    ndarray_rgb = frame.to_ndarray(format="rgb24")

                                    if ndarray_rgb is not None:
                                        logger.debug(f"Frame-NDArray Shape (RGB): {ndarray_rgb.shape}")

                                        # Initialize the LiveKit VideoFrame with the RGB data
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

            # Publish the video track in the LiveKit room
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


            SAMPLE_RATE = 48000
            NUM_CHANNELS = 1

            # Create an AudioSource for LiveKit
            audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
            print("AudioSource for LiveKit created")

        
            async def forward_audio(relayed_audio, audio_source):
                while True:
                    try:
                        frame = await relayed_audio.recv()
                        if frame:
                            print("Audio frame received from Simli")

                            # Log the frame details
                            print(f"Sample rate: {frame.sample_rate}")
                            print(f"Channels layout: {frame.layout.channels}")
                            print(f"Format: {frame.format.name}")
                            print(f"Time base: {frame.time_base}")
                            print(f"PTS (presentation timestamp): {frame.pts}")
                            print(f"Number of planes: {len(frame.planes)}")

                            if len(frame.planes) > 0:
                                audio_plane = frame.planes[0]

                                # Log buffer details
                                print(f"Buffer Pointer: {audio_plane.buffer_ptr}")
                                print(f"Buffer Size: {audio_plane.buffer_size}")

                                # Read the raw audio data from the buffer as bytearray
                                audio_data_ptr = ctypes.cast(audio_plane.buffer_ptr, ctypes.POINTER(ctypes.c_int16))
                                audio_data = np.frombuffer(ctypes.string_at(audio_data_ptr, audio_plane.buffer_size), dtype=np.int16)

                                # Convert stereo to mono by averaging the two channels
                                if len(frame.layout.channels) == 2:
                                    audio_data = audio_data.reshape(-1, 2)
                                    mono_audio_data = audio_data.mean(axis=1).astype(np.int16)
                                else:
                                    mono_audio_data = audio_data

                                # Create the LiveKit AudioFrame with the PCM data
                                samples_per_channel = len(mono_audio_data)
                                livekit_frame = rtc.AudioFrame.create(SAMPLE_RATE, NUM_CHANNELS, samples_per_channel)
                                np.copyto(np.frombuffer(livekit_frame.data, dtype=np.int16), mono_audio_data)

                                # Send the frame to the LiveKit audio source    
                                await audio_source.capture_frame(livekit_frame)
                                print("AudioFrame successfully captured in AudioSource")
                            else:
                                print("No audio planes found in the frame!")

                    except asyncio.CancelledError:
                        print("forward_audio task was cancelled")
                        break
                    except Exception as e:
                        print(f"Error processing audio frame: {e}")

                    await asyncio.sleep(0)  # Yield control to the event loop

            task_forward_audio = asyncio.create_task(forward_audio(relayed_audio, audio_source))

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
                    #options = rtc.TrackPublishOptions()
                    #options.source = rtc.TrackSource.SOURCE_MICROPHONE
                    publication = await room.local_participant.publish_track(livekit_audio_track) #,options
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

# Send TTS audio to Simli after connection setup
    async def test_send_tts():
        await send_tts_audio_to_simli("How are you? I am the Assistant Chris.")  # Test TTS audio with this phrase

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

    #await asyncio.sleep(2)
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
            agent_name="Albert", entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, port=8084
        )
    )