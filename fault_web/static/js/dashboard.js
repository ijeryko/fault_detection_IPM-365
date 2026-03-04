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
    if (acc >= 90) return "#2ecc71";
    if (acc >= 75) return "#94e073";
    if (acc >= 50) return "#f1c40f";
    if (acc >= 25) return "#e67e22";
    return "#e74c3c";
  }

  /* =========================
      STATUS COLOR LOGIC
  ========================== */
  function getStatusClass(label) {
    const text = label.toLowerCase();
    if (text.includes("healthy")) return "status-healthy";
    if (text.includes("off")) return "status-off";
    if (text.includes("belt")) return "status-belt_fault";
    if (text.includes("rusty")) return "status-rusty";
    return "status-fault"; 
  }

  /* =========================
      UPDATE MAIN CARDS
  ========================== */
  function updateStatus(label, accuracy) {
    const accDisplay = (accuracy).toFixed(1) + "%";
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

    const accPercent = (accuracy).toFixed(1) + "%";
    const color = getAccuracyColor(accuracy);

    const labelClass = label.toLowerCase().includes("healthy") ? "label-healthy" : 
                       label.toLowerCase().includes("off") ? "label-off" : 
                       label.toLowerCase().includes("belt") ? "label-loose_belt" :
                       label.toLowerCase().includes("bearing_fault") ? "label-bearing_fault" :

                       "label-fault";

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
  const socket = io("https://yadiel-nonefficient-eda.ngrok-free.dev", {
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

  function handleNewResult(msg) {
    if (!msg || !msg.prediction) return;

    // 1. Get exact label from Python and convert to lowercase for comparison
    const rawLabel = msg.prediction.label || "Unknown";
    const label = rawLabel.toLowerCase();
    const accuracy = msg.prediction.confidence ?? 0;

    // IMPORTANT:
    // Your backend sometimes sends timestamp as ms int, sometimes string.
    // Handle both safely:
    let timestamp;
    if (msg.timestamp) {
      const d = new Date(msg.timestamp);
      timestamp = isNaN(d.getTime()) ? new Date().toLocaleString() : d.toLocaleString();
    } else {
      timestamp = new Date().toLocaleString();
    }

    updateStatus(rawLabel, accuracy);
    addRow(rawLabel, accuracy, timestamp);

    // VIDEO UPDATE LOGIC
    if (faultVideo && videoSource) {
        let newVideo = "";
        let displayTitle = "";

        if (label.includes("healthy")) {
            newVideo = VIDEO_PATH + "healthy.mp4";
            displayTitle = "Healthy Line";
        } 
        else if (label.includes("off")) {
            newVideo = VIDEO_PATH + "off.mp4";
            displayTitle = "Off State";
        }
        else if (label.includes("belt")) {
            newVideo = VIDEO_PATH + "belt.mp4";
            displayTitle = "Belt Fault Detected";
        } 
        else if (label.includes("waiting")) {
            newVideo = VIDEO_PATH + "waiting.mp4";
            displayTitle = "Waiting for Fault";
        }
        else {
            newVideo = VIDEO_PATH + "waiting.mp4";
            displayTitle = rawLabel;
        }

        if (faultText) faultText.innerText = displayTitle;

        if (!videoSource.src.toLowerCase().includes(newVideo.toLowerCase())) {
            faultVideo.classList.add("video-fade");

            setTimeout(() => {
                videoSource.src = newVideo;
                faultVideo.load();

                faultVideo.oncanplay = () => {
                    faultVideo.play().catch(e => console.log("Playback error:", e));
                    faultVideo.classList.remove("video-fade");
                    faultVideo.oncanplay = null;
                };
            }, 100);
        }
    }
}

  /* =========================
      RECEIVE NEW RESULT
  ========================== */
  /* =========================
    RECEIVE NEW RESULT
========================== */
  socket.on("new_result", (msg) => {
    console.log("Received new result:", msg);
      handleNewResult(msg);
      if (!msg || !msg.prediction) return;

      // 1. Get exact label from Python and convert to lowercase for comparison
      const rawLabel = msg.prediction.label || "Unknown";
      const label = rawLabel.toLowerCase();
      const accuracy = msg.prediction.confidence ?? 0;
      const timestamp = msg.timestamp ? new Date(msg.timestamp).toLocaleString() : new Date().toLocaleString();

      updateStatus(rawLabel, accuracy);
      addRow(rawLabel, accuracy, timestamp);

      // VIDEO UPDATE LOGIC
      if (faultVideo && videoSource) {
          let newVideo = "";
          let displayTitle = "";

          // 2. Clearer Priority Logic
          if (label.includes("healthy")) {
              newVideo = VIDEO_PATH + "healthy.mp4";
              displayTitle = "Healthy Line";
          } 
          else if (label.includes("off")) {
              newVideo = VIDEO_PATH + "off.mp4";
              displayTitle = "Off State";
          }
          else if (label.includes("belt")) { // Using "belt" covers "Belt_Fault" or "belt fault"
              newVideo = VIDEO_PATH + "belt.mp4";
              displayTitle = "Belt Fault Detected";
          } 

          else if (label.includes("waiting")) {
              newVideo = VIDEO_PATH + "waiting.mp4";
              displayTitle = "Waiting for Fault";
          }
          else {
              newVideo = VIDEO_PATH + "waiting.mp4";
              displayTitle = rawLabel; // Show the actual unknown label
          }

          if (faultText) {
              faultText.innerText = displayTitle;
          }

          // 3. Only change source if it's different to prevent flickering
          // We use .includes() because videoSource.src returns the FULL URL (http://...)
          if (!videoSource.src.toLowerCase().includes(newVideo.toLowerCase())) {
              console.log("Status changed! Switching video to:", newVideo); // Debugging line
              
              faultVideo.classList.add("video-fade");

              setTimeout(() => {
                  videoSource.src = newVideo;
                  faultVideo.load();
                  
                  faultVideo.oncanplay = () => {
                      faultVideo.play().catch(e => console.log("Playback error:", e));
                      faultVideo.classList.remove("video-fade");
                      faultVideo.oncanplay = null;
                  };
              }, 100); 
          }
      }
  });

  socket.on("init_events", (data) => {
    if (!data || !Array.isArray(data.events)) return;

    // data.events are in DEVICE payload format:
    // { device_id, timestamp, prediction:{label,confidence}, metrics... }
    // Convert each to the new_result format expected by handleNewResult.
    data.events.forEach((p) => {
      console.log("Processing event:", p);
      if (!p || !p.prediction) return;

      handleNewResult({
        prediction: p.prediction,
        timestamp: p.timestamp
      });
    });
  });

})();