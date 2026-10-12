(() => {
  const $ = (id) => document.getElementById(id);

  const MAX_BYTES = 8 * 1024 * 1024;
  const OK_TYPES = ["image/jpeg", "image/png", "image/webp"];

  // The page only remembers the image the user picked; the AI work happens on the server.
  const state = { stream: null, blob: null, previewUrl: null, scanning: false, healthTimer: null };

  const ITEM_ICONS = { Battery: "🔋", Biological: "🍌", Cardboard: "📦", Clothes: "👕", Glass: "🍾",
    Metal: "🥫", Paper: "📄", Plastic: "🧴", Shoes: "👟", Trash: "🗑️" };
  const ROUTE_ICONS = { "Recycle": "♻", "Compost / organics": "🍌", "Special drop-off": "🔋",
    "Donate / reuse": "👕", "General waste": "🗑️", "Needs review": "🔎" };
  const RECYCLABLE_TEXT = {
    conditional: "GENERALLY RECYCLABLE · IF CLEAN AND ACCEPTED LOCALLY",
    no: "NOT FOR THE RECYCLING BIN",
  };

  // ---------------------------------------------------------------- helpers
  let toastTimer;
  function toast(message) {
    const el = $("toast");
    el.textContent = message;
    el.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("show"), 3500);
  }
  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  const percent = (fraction) => `${Math.round(fraction * 100)}%`;

  function showError(message) {
    const box = $("error-box");
    box.textContent = message;
    box.hidden = !message;
  }
  function setStatus(text) { $("camera-status").textContent = text; }

  // ----------------------------------------------------------- model status
  async function checkModel() {
    try {
      const response = await fetch("/api/health");
      if (!response.ok) throw new Error("health failed");
      const { model } = await response.json();
      const ready = model.state === "ready";
      $("model-dot").classList.toggle("ready", ready);
      $("model-status").textContent = ready
        ? "AI model ready · waste classifier (SigLIP2, 10 classes)"
        : model.message;
      if (ready || model.state === "error") stopHealthPolling();
    } catch (_) {
      $("model-dot").classList.remove("ready");
      $("model-status").textContent = "Cannot reach the ECHO SCAN server. Is it still running?";
    }
  }
  function startHealthPolling() {
    checkModel();
    state.healthTimer = setInterval(checkModel, 4000);
  }
  function stopHealthPolling() {
    clearInterval(state.healthTimer);
    state.healthTimer = null;
  }

  // ----------------------------------------------------------------- camera
  function stopCamera() {
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    $("camera").srcObject = null;
    $("camera").hidden = true;
    $("capture-btn").disabled = true;
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      toast("Camera needs a supported browser opened at http://127.0.0.1:5000. You can still upload a photo.");
      return;
    }
    try {
      stopCamera();
      state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } }, audio: false });
      clearImage();
      $("camera").hidden = false;
      $("camera-placeholder").hidden = true;
      $("camera").srcObject = state.stream;
      await $("camera").play();
      $("capture-btn").disabled = false;
      $("start-camera").innerHTML = "<span>◎</span> Restart camera";
      setStatus("CAMERA LIVE");
    } catch (_) {
      stopCamera();
      $("camera-placeholder").hidden = false;
      toast("Camera permission was denied or no camera was found. Try uploading a photo.");
    }
  }

  function capturePhoto() {
    const video = $("camera");
    if (!state.stream || !video.videoWidth) { toast("Wait for the camera preview, then capture."); return; }
    const canvas = $("capture-canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) { toast("Could not capture the photo. Please try again."); return; }
      stopCamera();
      setImage(blob, "camera-photo.jpg");
    }, "image/jpeg", 0.92);
  }

  // ----------------------------------------------------------------- image
  function clearImage() {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
    state.blob = null;
    $("preview-image").hidden = true;
    $("preview-image").removeAttribute("src");
    $("scan-btn").disabled = true;
  }

  function setImage(blob, label) {
    clearImage();
    state.blob = blob;
    state.previewUrl = URL.createObjectURL(blob);
    $("preview-image").src = state.previewUrl;
    $("preview-image").hidden = false;
    $("camera").hidden = true;
    $("camera-placeholder").hidden = true;
    $("scan-btn").disabled = false;
    setStatus("IMAGE READY · PRESS SCAN");
    showResult(null);
    showError("");
    $("scan-btn").dataset.label = label;
  }

  function onFileChosen(event) {
    const file = event.target.files && event.target.files[0];
    event.target.value = "";            // lets the same file be chosen again later
    if (!file) return;
    if (!OK_TYPES.includes(file.type)) { toast("Please choose a JPG, PNG or WebP image."); return; }
    if (file.size > MAX_BYTES) { toast("That image is larger than 8 MB. Choose a smaller photo."); return; }
    if (file.size === 0) { toast("That file is empty."); return; }
    stopCamera();
    setImage(file, file.name);
  }

  // ----------------------------------------------------------------- scanning
  function setScanning(on) {
    state.scanning = on;
    $("scan-overlay").hidden = !on;
    $("scan-btn").disabled = on || !state.blob;
    $("scan-btn-label").textContent = on ? "Scanning…" : "Scan with AI";
  }

  async function scan() {
    if (!state.blob || state.scanning) return;
    setScanning(true);
    showError("");
    setStatus("SCANNING…");
    try {
      const form = new FormData();
      form.append("image", state.blob, $("scan-btn").dataset.label || "photo.jpg");
      const response = await fetch("/api/classify", { method: "POST", body: form });
      let data = null;
      try { data = await response.json(); } catch (_) { /* not JSON */ }
      if (!response.ok) {
        const message = (data && data.error) || "The scan failed. Please try again.";
        showResult(null);
        showError(message);
        setStatus("SCAN FAILED · NO RESULT");
        if (response.status === 503) checkModel();
        return;
      }
      showResult(data);
      setStatus(data.status === "classified" ? "SCAN COMPLETE · VERIFY RESULT" : "SCAN COMPLETE · NOT SURE");
      await Promise.all([loadHistory(), loadStats()]);
    } catch (_) {
      showResult(null);
      showError("Could not reach the ECHO SCAN server. Check that it is still running, then try again.");
      setStatus("SCAN FAILED · NO RESULT");
    } finally {
      setScanning(false);
    }
  }

  function showResult(r) {
    $("empty-result").hidden = !!r;
    $("result-content").hidden = !r;
    if (!r) { $("result-status").textContent = "AWAITING SCAN"; $("confidence-bar").style.width = "0%"; return; }

    const classified = r.status === "classified";
    const good = classified && r.route === "Recycle";
    $("result-status").textContent = classified ? (good ? "RECYCLING STREAM" : "SPECIAL ROUTE") : "NOT SURE";
    $("result-hero").classList.toggle("nonrecyclable", classified && !good);
    $("result-hero").classList.toggle("unsure", !classified);
    $("result-symbol").textContent = classified ? (ITEM_ICONS[r.predicted_label] || "♻") : "🔎";
    $("result-overline").textContent = classified ? (RECYCLABLE_TEXT[r.recyclable] || "FOLLOW THIS DISPOSAL ROUTE") : "LOW CONFIDENCE · VERIFY MANUALLY";
    $("result-title").textContent = classified ? r.route : "Needs review";
    $("result-item").textContent = classified ? `Detected: ${r.item}` : r.item;
    $("confidence-value").textContent = `${percent(r.confidence)} · model score, not a guarantee`;
    $("confidence-bar").style.width = percent(r.confidence);
    $("result-message").textContent = r.message || "";
    $("result-message").hidden = !r.message;
    $("guidance-text").textContent = r.guidance;
    $("warning-text").textContent = r.warning;
    $("category-label").textContent = r.category;
    $("saved-note").textContent = "Saved to history";
    const alternatives = $("alternatives");
    alternatives.textContent = r.alternatives && r.alternatives.length
      ? "Other possibilities: " + r.alternatives.map((a) => `${a.label} (${percent(a.probability)})`).join(" · ")
      : "";
    alternatives.hidden = !alternatives.textContent;
  }

  function resetScan() {
    stopCamera();
    clearImage();
    showResult(null);
    showError("");
    $("camera-placeholder").hidden = false;
    $("start-camera").innerHTML = "<span>◎</span> Start camera";
    setStatus("NO IMAGE YET");
  }

  // ------------------------------------------------- history and dashboard
  async function loadStats() {
    try {
      const response = await fetch("/api/stats");
      if (!response.ok) throw new Error("stats failed");
      const s = await response.json();
      $("stat-total").textContent = s.total;
      $("stat-recyclable").textContent = s.recyclable;
      $("stat-other").textContent = s.other_routes;
      $("stat-review").textContent = s.needs_review;
      const max = Math.max(1, ...s.by_category.map((c) => c.count));
      $("breakdown").hidden = !s.by_category.length;
      $("breakdown-list").innerHTML = s.by_category.map((c) =>
        `<div class="breakdown-row"><span>${escapeHtml(c.category)}</span><div class="breakdown-track"><div class="breakdown-bar" data-width="${Math.round((c.count / max) * 100)}"></div></div><b>${c.count}</b></div>`
      ).join("");
      document.querySelectorAll(".breakdown-bar").forEach((bar) => { bar.style.width = bar.dataset.width + "%"; });
    } catch (_) { toast("Could not load statistics. Please refresh the page."); }
  }

  async function loadHistory() {
    const body = $("history-body");
    try {
      const response = await fetch("/api/history");
      if (!response.ok) throw new Error("history failed");
      const rows = await response.json();
      if (!rows.length) {
        body.innerHTML = '<tr><td colspan="5" class="table-empty">No scans yet. Your first scan will show up here.</td></tr>';
        return;
      }
      body.innerHTML = rows.map((r) => {
        const sure = r.status === "classified";
        const icon = sure ? (ITEM_ICONS[r.predicted_label] || "♻") : "🔎";
        const klass = !sure ? "review" : (r.route === "Recycle" ? "yes" : "no");
        const when = new Date(r.created_at);
        const date = isNaN(when) ? r.created_at : when.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
        return `<tr><td><div class="item-cell"><span class="item-mini">${icon}</span>${escapeHtml(r.item_name)}</div></td>` +
          `<td><span class="category-pill">${escapeHtml(r.category)}</span></td>` +
          `<td><span class="decision-pill ${klass}">${escapeHtml(r.route)}</span></td>` +
          `<td>${percent(r.confidence)}</td><td>${escapeHtml(date)}</td></tr>`;
      }).join("");
    } catch (_) {
      body.innerHTML = '<tr><td colspan="5" class="table-empty">Could not load history. Refresh the page.</td></tr>';
    }
  }

  async function clearHistory() {
    if (!confirm("Clear your whole scan history? This cannot be undone.")) return;
    try {
      const response = await fetch("/api/history", {
        method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true }),
      });
      if (!response.ok) throw new Error("clear failed");
      await Promise.all([loadHistory(), loadStats()]);
      toast("Your scan history was cleared.");
    } catch (_) { toast("Could not clear history. Is the server running?"); }
  }

  // ------------------------------------------------------------------ start
  $("start-camera").addEventListener("click", startCamera);
  $("capture-btn").addEventListener("click", capturePhoto);
  $("image-upload").addEventListener("change", onFileChosen);
  $("scan-btn").addEventListener("click", scan);
  $("reset-scan").addEventListener("click", resetScan);
  $("clear-history").addEventListener("click", clearHistory);
  window.addEventListener("beforeunload", stopCamera);

  loadHistory();
  loadStats();
  startHealthPolling();
})();
