# Multi-agent meeting LiveKit server AND with Simli.com Avatars

## Run LiveKit server
These commands will install [LiveKit server](https://github.com/livekit/livekit) on your machine and run it in dev mode. Dev mode uses a specific API key and secret pair.
1. `brew install livekit`
2. `livekit-server -dev`

## Run LiveKit Meet
Usually you'd run the agent(s) first and then start a session and the agent(s) would automatically join. Turns out that isn't how it works for multi-agent at the moment. So what we're going to do is have the human join the meeting first, and then explicitly have the agents join the room.
1. `cd meet`
2. `pnpm i`
3. `cp .env.example .env.local`
4. `pnpm dev`
5. open `localhost:3000` in a browser and click 'Start Meeting'
6. note the room name in your browser address bar: `http://localhost:3000/rooms/<room-name>`

## Run first Agent (With Simli Avatar)

✅ Handling & Handshakes with Web RTC Implementation with Simli Avatar are working

❌ Avatar Video Picture is not working at the moment, got a green broken screen 🙁

🪄 most of the magic is in the agent-1/main.py file


![Description of the image](/error.jpeg)

### How `main.py` Works

The `main.py` script in the `agent-1` directory is designed to facilitate a WebRTC session using a voice assistant and video processing capabilities.

#### Function Descriptions with Simli API Integration

1. **`prewarm(proc: JobProcess)`**:
   - Loads the voice activity detection (VAD) model from the `silero` plugin and stores it in the process's user data for later use.

2. **`SilentAudioTrack` and `BlackVideoTrack` Classes**:
   - **Purpose**: Generate silent audio and black video frames, respectively.
   - **Usage**: These tracks are used as placeholders or default tracks in the WebRTC session.

3. **`entrypoint(ctx: JobContext)`**:
   - **Purpose**: Main function to set up and manage the WebRTC session.
   - **Steps**:
     - Initializes a `VoiceAssistant` with VAD, STT, and TTS capabilities.
     - Connects to the LiveKit room and starts the voice assistant.
     - **Simli API Setup**:
       - Retrieves Simli API keys (`SIMLI_API_KEY` and `SIMLI_FACE_ID`) from environment variables.
       - Starts an audio-to-video session with Simli to obtain a session token using `start_audio_to_video_session`.
     - Configures ICE servers and creates an `RTCPeerConnection`.
     - Sets up a `DataChannel` to send the session token.
     - Subscribes to incoming media tracks (audio and video) and relays them using `MediaRelay`.
     - Creates and sends a WebRTC offer to Simli, then sets the remote description with the received answer using `start_webrtc_session`.
     - Monitors ICE connection state changes and logs relevant information.

4. **`start_audio_to_video_session(api_key, face_id)`**:
   - **Purpose**: Initiates an audio-to-video session with Simli's API.
   - **Process**:
     - Sends a POST request to Simli's API endpoint with the `faceId`, `apiKey`, and other parameters.
     - **Returns**: A session token if successful, or logs an error if not.

5. **`start_webrtc_session(offer_sdp, offer_type, api_key, session_token)`**:
   - **Purpose**: Sends a WebRTC offer to Simli and receives an SDP answer.
   - **Process**:
     - Sends a POST request to Simli's WebRTC API endpoint with the SDP offer, type, API key, and session token.
     - **Returns**: The SDP answer if successful, or logs an error if not.

#### WebRTC Connection Details

- **ICE Servers**: Configured using Google's public STUN servers to facilitate NAT traversal.
- **RTCPeerConnection**: 
  - Created with the specified ICE servers.
  - Manages the connection and media tracks between the local and remote peers.
- **DataChannel**: 
  - Used to send the session token to the remote peer once the channel is open.
- **Track Handling**:
  - Subscribes to incoming tracks and relays them using `MediaRelay`.
  - Forwards video and audio frames to LiveKit's media sources.
- **SDP Offer/Answer**:
  - Creates an SDP offer and waits for ICE gathering to complete.
  - Sends the offer to Simli and sets the remote description with the received answer.
- **ICE Connection State Monitoring**:
  - Logs changes in the ICE connection state to provide insights into the connection status.

#### Simli API Setup Requirements
- **Environment Variables (in .env file) **:
  - `SIMLI_API_KEY`: Your API key for accessing Simli's services.
  - `SIMLI_FACE_ID`: The ID of the face to be used in the audio-to-video session.
- **API Endpoints**:
  - **Audio-to-Video Session**: `https://api.simli.ai/startAudioToVideoSession`
  - **WebRTC Session**: `https://api.simli.ai/StartWebRTCSession`
- **Network Configuration**:
  - Ensure that your environment can make outbound HTTP requests to Simli's API endpoints.

1. `cd agent-1`
2. `python -m venv .venv`
3. `source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. `cp .env.example .env`
6. add values for keys in `.env`
7. `python main.py connect --room <room-name>`

## Run second agent (At the moment only standard audio Agent)
1. `cd agent-2`
2. `python -m venv .venv`
3. `source .venv/bin/activate`
4. `pip install -r requirements.txt`
5. `cp ../agent-1/.env .`
7. `python main.py connect --room <room-name>`