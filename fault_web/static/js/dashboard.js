(function () {

  // DOM elements - Swapped faultImage for faultVideo
  const faultVideo = document.getElementById("fault-video"); 
  const videoSource = document.getElementById("video-source");
  const faultText = document.getElementById("fault-text");
  const eventsBody = document.getElementById("events-body");
  const accuracyCard = document.getElementById("current-accuracy");
  const statusCard = document.getElementById("current-status");
  const connectionCard = document.getElementById("connection-status");

  // Changed path to your videos folder
  const VIDEO_PATH = "/static/videos/"; 

  /* =========================
      ACCURACY COLOR SCALE
  ========================== */
  function getAccuracyColor(acc) {
    // Note: Adjusted thresholds to handle 0.0-1.0 scale
    if (acc >= 0.90) return "#2ecc71";
    if (acc >= 0.75) return "#94e073";
    if (acc >= 0.50) return "#f1c40f";
    if (acc >= 0.25) return "#e67e22";
    return "#e74c3c";
  }

  /* =========================
      STATUS COLOR LOGIC
  ========================== */
  function getStatusClass(label) {
    const text = label.toLowerCase();
    if (text.includes("healthy")) return "status-healthy";
    if (text.includes("off")) return "status-off";
    return "status-fault"; 
  }

  /* =========================
      UPDATE MAIN CARDS
  ========================== */
  function updateStatus(label, accuracy) {
    const accDisplay = (accuracy * 100).toFixed(0) + "%";
    const accColor = getAccuracyColor(accuracy);

    accuracyCard.innerText = accDisplay;
    accuracyCard.style.color = accColor;

    statusCard.className = "status-value"; // Reset classes
    statusCard.innerText = label;
    statusCard.classList.add(getStatusClass(label));
  }

  /* =========================
      ADD EVENT ROW
  ========================== */
  function addRow(label, accuracy, timestamp) {
    const row = document.createElement("div");
    row.className = "event-row";

    const accPercent = (accuracy * 100).toFixed(0) + "%";
    const color = getAccuracyColor(accuracy);

    const labelClass = label.toLowerCase().includes("healthy") ? "label-healthy" : 
                       label.toLowerCase().includes("off") ? "label-off" : "label-fault";

    row.innerHTML = `
      <span style="display:flex; justify-content:center;">
        <div class="label ${labelClass}">${label}</div>
      </span>
      <span style="font-weight:bold;color:${color}">${accPercent}</span>
      <span>${timestamp}</span>
    `;

    eventsBody.prepend(row);
    if (eventsBody.children.length > 50) eventsBody.removeChild(eventsBody.lastChild);
  }

  /* =========================
      SOCKET CONNECTION
  ========================== */
  const socket = io("https://unhesitantly-unresuscitating-jazlyn.ngrok-free.dev", {
    transports: ["websocket"]
  });

  // Connection card defaults
  connectionCard.innerText = "Connecting...";
  connectionCard.style.color = "#f1c40f";

  socket.on("connect", () => {
    console.log("Connected to server");
    connectionCard.innerText = "Connected";
    connectionCard.style.color = "#20a457";
  });

  socket.on("disconnect", () => {
    console.log("Disconnected from server");
    connectionCard.innerText = "Disconnected";
    connectionCard.style.color = "#e74c3c";
  });

  socket.on("connect_error", () => {
    console.log("Connection error");
    connectionCard.innerText = "Error";
    connectionCard.style.color = "#e67e22";
  });

  /* =========================
      RECEIVE NEW RESULT
  ========================== */
  socket.on("new_result", (msg) => {
    if (!msg || !msg.prediction) return;

    const label = msg.prediction.label || "Unknown";
    const accuracy = msg.prediction.confidence ?? 0;
    const timestamp = msg.timestamp ? new Date(msg.timestamp).toLocaleString() : new Date().toLocaleString();

    updateStatus(label, accuracy);
    addRow(label, accuracy, timestamp);

    // VIDEO UPDATE LOGIC
    if (faultVideo && videoSource) {
      let newVideo = "";

      if (label.toLowerCase().includes("healthy")) {
        newVideo = VIDEO_PATH + "healthy.mp4";
        faultText.innerText = "Healthy Line";
      } 

      else if (label.toLowerCase().includes("off")) {
        newVideo = VIDEO_PATH + "off.mp4";
        faultText.innerText = "Off State";
      }

      else if (label.toLowerCase().includes("belt fault")) {
        newVideo = VIDEO_PATH + "belt.mp4";
        faultText.innerText = "Belt Fault Detected";
      } 
      else {
        newVideo = VIDEO_PATH + "waiting.mp4";
        faultText.innerText = "Waiting";
        faultText.innerText = label;
      }

      // Only change source if it's different to prevent flickering
    if (!videoSource.src.includes(newVideo)) {
        // 1. Fade out
        faultVideo.classList.add("video-fade");

        setTimeout(() => {
          videoSource.src = newVideo;
          faultVideo.load();
          
          // 2. Fade back in once ready
          faultVideo.oncanplay = () => {
            faultVideo.play();
            faultVideo.classList.remove("video-fade");
            faultVideo.oncanplay = null;
          };
        }, 500); 
      }
    }
  }); 

})();