(function () {
  const statusEl = document.getElementById("status");
  const logEl = document.getElementById("log");
  const latestLineEl = document.getElementById("latestLine");
  const latestMetaEl = document.getElementById("latestMeta");
  const clearBtn = document.getElementById("clearBtn");

  function prependLine(line) {
    // Keep newest at top
    const current = logEl.textContent || "";
    logEl.textContent = line + "\n" + current;
  }

  // Load initial snapshot (page refresh still shows history)
  if (Array.isArray(INITIAL_LINES)) {
    for (let i = INITIAL_LINES.length - 1; i >= 0; i--) {
      prependLine(INITIAL_LINES[i]);
    }
  }

  clearBtn.addEventListener("click", () => {
    logEl.textContent = "";
    latestLineEl.textContent = "—";
    latestMetaEl.textContent = "—";
  });

  const socket = io({
    transports: ["websocket"],
  });

  socket.on("connect", () => {
    statusEl.textContent = "Connected";
  });

  socket.on("disconnect", () => {
    statusEl.textContent = "Disconnected";
  });

  socket.on("server_hello", (msg) => {
    // optional
  });

  socket.on("new_result", (msg) => {
    const line = msg?.line || "—";
    latestLineEl.textContent = line;

    const deviceId = msg?.device_id || "unknown";
    const sps = (msg?.sps_est ?? 0).toFixed(1);
    const ts = msg?.ts_ms ? new Date(msg.ts_ms).toLocaleString() : "";
    latestMetaEl.textContent = `Device: ${deviceId} | SPS est: ${sps} | ${ts}`;

    prependLine(line);
  });
})();
