// get DOM elements
var dataChannelLog = document.getElementById("data-channel"),
  iceConnectionLog = document.getElementById("ice-connection-state"),
  iceGatheringLog = document.getElementById("ice-gathering-state"),
  signalingLog = document.getElementById("signaling-state");

// peer connection
var pc = null;

// data channel
var dc = null,
  dcInterval = null;

function createPeerConnection() {
  var config = {
    sdpSemantics: "unified-plan",
  };

  config.iceServers = [{ urls: ["stun:stun.l.google.com:19302"] }];

  pc = new RTCPeerConnection(config);
  // register some listeners to help debugging
  pc.addEventListener(
    "icegatheringstatechange",
    () => {
      iceGatheringLog.textContent += " -> " + pc.iceGatheringState;
    },
    false
  );
  iceGatheringLog.textContent = pc.iceGatheringState;

  pc.addEventListener(
    "iceconnectionstatechange",
    () => {
      iceConnectionLog.textContent += " -> " + pc.iceConnectionState;
    },
    false
  );
  iceConnectionLog.textContent = pc.iceConnectionState;

  pc.addEventListener(
    "signalingstatechange",
    () => {
      signalingLog.textContent += " -> " + pc.signalingState;
    },
    false
  );
  signalingLog.textContent = pc.signalingState;

  // connect audio / video
  pc.addEventListener("track", (evt) => {
    if (evt.track.kind == "video")
      document.getElementById("video").srcObject = evt.streams[0];
    else document.getElementById("audio").srcObject = evt.streams[0];
  });

  pc.onicecandidate = (event) => {
    if (event.candidate === null) {
      console.log(JSON.stringify(pc.localDescription));
    } else {
      console.log(event.candidate);
      candidateCount += 1;
    }
  };

  return pc;
}

let candidateCount = 0;
let prevCandidateCount = -1;
function CheckIceCandidates() {
  if (
    pc.iceGatheringState === "complete" ||
    candidateCount === prevCandidateCount
  ) {
    console.log(pc.iceGatheringState, candidateCount);
    connectToRemotePeer();
  } else {
    prevCandidateCount = candidateCount;
    setTimeout(CheckIceCandidates, 250);
  }
}

function negotiate() {
  return pc
    .createOffer()
    .then((offer) => {
      return pc.setLocalDescription(offer);
    })
    .then(() => {
      prevCandidateCount = candidateCount;
      setTimeout(CheckIceCandidates, 250);
    });
}

function connectToRemotePeer() {
  var offer = pc.localDescription;
  var codec;
  document.getElementById("offer-sdp").textContent = offer.sdp;

  return fetch("https://api.simli.ai/StartWebRTCSession", {
    body: JSON.stringify({
      sdp: offer.sdp,
      type: offer.type,
    }),
    headers: {
      "Content-Type": "application/json",
    },
    method: "POST",
  })
    .then((response) => {
      return response.json();
    })
    .then((answer) => {
      document.getElementById("answer-sdp").textContent = answer.sdp;
      return pc.setRemoteDescription(answer);
    })
    .then((out) => {
      return out;
    })
    .catch((e) => {
      alert(e);
    });
}

function start() {
  document.getElementById("start").style.display = "none";

  pc = createPeerConnection();

  var time_start = null;

  const current_stamp = () => {
    if (time_start === null) {
      time_start = new Date().getTime();
      return 0;
    } else {
      return new Date().getTime() - time_start;
    }
  };

  var parameters = { ordered: true };
  dc = pc.createDataChannel("datachannel", parameters);

  dc.addEventListener("error", (err) => {
    console.error(err);
  });

  dc.addEventListener("close", () => {
    clearInterval(dcInterval);
    dataChannelLog.textContent += "- close\n";
  });

  dc.addEventListener("open", async () => {
    console.log(dc.id);
    const metadata = {
      faceId: "tmp9i8bbq7c", // Simli face ID
      isJPG: false,
      apiKey: "14kg6yd4wjxlm5se40ei1", // Simli API key
      syncAudio: true,
    };

    const response = await fetch(
      "https://api.simli.ai/startAudioToVideoSession",
      {
        method: "POST",
        body: JSON.stringify(metadata),
        headers: {
          "Content-Type": "application/json",
        },
      }
    );
    const resJSON = await response.json();
    dc.send(resJSON.session_token);

    dataChannelLog.textContent += "- open\n";
    dcInterval = setInterval(() => {
      var message = "ping " + current_stamp();
      dataChannelLog.textContent += "> " + message + "\n";
      dc.send(message);
    }, 1000);
    await setTimeout(()=>{}, 100);
    var message = new Uint8Array(16000)  // 0.5 second silence to start the audio (Optional)
    dc.send(message);
    initializeWebsocketDecoder()
  });
  
  dc.addEventListener("message", (evt) => {
    dataChannelLog.textContent += "< " + evt.data + "\n";

    if (evt.data.substring(0, 4) === "pong") {
      var elapsed_ms = current_stamp() - parseInt(evt.data.substring(5), 10);
      dataChannelLog.textContent += " RTT " + elapsed_ms + " ms\n";
    }
  });

  // Build media constraints.

  const constraints = {
    audio: true,
    video: true,
  };

  // Acquire media and start negotiation.

  document.getElementById("media").style.display = "block";
  navigator.mediaDevices.getUserMedia(constraints).then(
    (stream) => {
      stream.getTracks().forEach((track) => {
        pc.addTrack(track, stream);
      });
      return negotiate();
    },
    (err) => {
      alert("Could not acquire media: " + err);
    }
  );
  document.getElementById("stop").style.display = "inline-block";
}

function stop() {
  document.getElementById("stop").style.display = "none";

  // close data channel
  if (dc) {
    dc.close();
  }

  // close transceivers
  if (pc.getTransceivers) {
    pc.getTransceivers().forEach((transceiver) => {
      if (transceiver.stop) {
        transceiver.stop();
      }
    });
  }

  // close local audio / video
  pc.getSenders().forEach((sender) => {
    sender.track.stop();
  });

  // close peer connection
  setTimeout(() => {
    pc.close();
  }, 500);
}
