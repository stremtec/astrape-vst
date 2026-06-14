const state = {
  status: null,
  settings: {},
  voicebanks: [],
  selectedVoicebank: "",
  selectedFile: null,
  stream: null,
  streamReady: false,
  inputContext: null,
  outputContext: null,
  mediaStream: null,
  worklet: null,
  nextOutputTime: 0,
  outputSampleRate: 44100,
  dryChunks: [],
  dryChunkOffset: 0,
  dryQueuedSamples: 0,
  gateEnvelope: 0,
  bypass: false,
  inputHistory: new Float32Array(512),
  outputHistory: new Float32Array(512),
  referenceBuffer: null,
  referencePlaying: false,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const titles = {
  convert: ["Live conversion", "44.1 kHz device I/O, causal 16 kHz content path"],
  voicebanks: ["VoiceBank library", "One continuous target recording, five seconds or longer"],
  audio: ["Audio I/O", "Device routing, callback cadence and gain staging"],
  diagnostics: ["Diagnostics", "FCPE profile, checkpoints and training state"],
};

function toast(message, kind = "") {
  const item = document.createElement("div");
  item.className = `toast ${kind}`;
  item.textContent = message;
  $("#toastRegion").appendChild(item);
  setTimeout(() => item.remove(), 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = payload.detail || payload.message || message;
    } catch {}
    throw new Error(message);
  }
  return response.json();
}

function setView(name) {
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  $("#viewTitle").textContent = titles[name][0];
  $("#viewSubtitle").textContent = titles[name][1];
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function formatSeconds(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} s` : "--";
}

function formatDb(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} dBFS` : "--";
}

function currentVoicebank() {
  return state.voicebanks.find((profile) => profile.id === state.selectedVoicebank) || null;
}

function profileInitial(name) {
  return (name || "V").slice(0, 1).toUpperCase();
}

function selectVoicebank(id, persist = true) {
  state.selectedVoicebank = id;
  state.settings.voicebank = id;
  renderVoicebanks();
  renderProfileDetail();
  if (persist) saveSettings();
}

function profileMarkup(profile, mode) {
  const warning = profile.quality_warnings && profile.quality_warnings.length;
  if (mode === "chip") {
    return `
      <button class="profile-chip ${profile.id === state.selectedVoicebank ? "active" : ""}" data-profile="${profile.id}">
        <span class="profile-avatar">${profileInitial(profile.id)}</span>
        <span><strong>${profile.id}</strong><small>${formatSeconds(profile.duration_seconds)} · ${warning ? "Check quality" : "Clean"}</small></span>
        <span class="profile-check">${profile.id === state.selectedVoicebank ? "✓" : ""}</span>
      </button>`;
  }
  return `
    <article class="profile-card ${profile.id === state.selectedVoicebank ? "active" : ""}" data-profile="${profile.id}">
      <div class="profile-card-head">
        <span class="profile-avatar">${profileInitial(profile.id)}</span>
        <strong>${profile.id}</strong>
      </div>
      <div class="profile-card-meta">
        <span>${formatSeconds(profile.duration_seconds)}</span>
        <span>${formatDb(profile.rms_dbfs)}</span>
        <span>${profile.active_speech_ratio == null ? "--" : `${Math.round(profile.active_speech_ratio * 100)}% active`}</span>
      </div>
      <footer>
        <span>${profile.embedding_model ? "Mio 128d" : "Unknown model"}</span>
        <span>${warning ? `${warning} warning` : "Ready"}</span>
      </footer>
    </article>`;
}

function renderVoicebanks() {
  $("#profileCount").textContent = `${state.voicebanks.length} profiles`;
  $("#profileSelector").innerHTML = state.voicebanks.length
    ? state.voicebanks.map((profile) => profileMarkup(profile, "chip")).join("")
    : '<div class="empty-inline">No VoiceBank profiles</div>';
  $("#profileGrid").innerHTML = state.voicebanks.length
    ? state.voicebanks.map((profile) => profileMarkup(profile, "card")).join("")
    : '<div class="empty-inline">Create the first VoiceBank profile</div>';
  $$("[data-profile]").forEach((item) => {
    item.addEventListener("click", () => selectVoicebank(item.dataset.profile));
  });
}

async function drawAudioWaveform(url) {
  const canvas = $("#referenceWaveform");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#101317";
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (!url) return;
  try {
    const response = await fetch(url);
    const data = await response.arrayBuffer();
    const audioContext = new AudioContext();
    const buffer = await audioContext.decodeAudioData(data.slice(0));
    state.referenceBuffer = buffer;
    const samples = buffer.getChannelData(0);
    const step = Math.max(1, Math.floor(samples.length / canvas.width));
    context.strokeStyle = "#63d18b";
    context.lineWidth = 1.2;
    context.beginPath();
    for (let x = 0; x < canvas.width; x += 1) {
      let peak = 0;
      const start = x * step;
      for (let index = start; index < Math.min(start + step, samples.length); index += 1) {
        peak = Math.max(peak, Math.abs(samples[index]));
      }
      const height = peak * canvas.height * 0.43;
      context.moveTo(x, canvas.height / 2 - height);
      context.lineTo(x, canvas.height / 2 + height);
    }
    context.stroke();
    await audioContext.close();
  } catch {
    state.referenceBuffer = null;
  }
}

function renderProfileDetail() {
  const profile = currentVoicebank();
  $("#detailName").textContent = profile ? profile.id : "No profile";
  $("#playReferenceButton").disabled = !profile?.preview_url;
  $("#deleteProfileButton").disabled = !profile;
  $("#analyzeF0Button").disabled = !profile?.has_source;
  const metricNodes = $$("#profileMetrics > div strong");
  const values = profile
    ? [
        formatSeconds(profile.duration_seconds),
        formatDb(profile.rms_dbfs),
        profile.active_speech_ratio == null ? "--" : `${(profile.active_speech_ratio * 100).toFixed(1)}%`,
        profile.clipping_fraction == null ? "--" : `${(profile.clipping_fraction * 100).toFixed(3)}%`,
      ]
    : ["--", "--", "--", "--"];
  metricNodes.forEach((node, index) => { node.textContent = values[index]; });
  const warnings = profile?.quality_warnings || [];
  $("#qualityLine").className = `quality-line ${profile ? (warnings.length ? "warning" : "good") : ""}`;
  $("#qualityLine").textContent = profile
    ? warnings.length ? warnings.join(", ") : "Reference quality checks passed"
    : "Select a VoiceBank profile";
  drawAudioWaveform(profile?.preview_url);
}

async function refreshVoicebanks() {
  state.voicebanks = await api("/api/voicebanks");
  if (!state.voicebanks.some((profile) => profile.id === state.selectedVoicebank)) {
    state.selectedVoicebank = state.voicebanks[0]?.id || "";
    state.settings.voicebank = state.selectedVoicebank;
  }
  renderVoicebanks();
  renderProfileDetail();
}

function applyStatus() {
  const status = state.status;
  const decoder = status.decoder;
  const training = status.training;
  $("#runtimeDot").className = `status-dot ${decoder.ready ? "ready" : "warning"}`;
  $("#runtimeLabel").textContent = decoder.ready ? "Runtime ready" : "Decoder training pending";
  $("#trainingLabel").textContent = training.running
    ? `E${String(training.epoch ?? "--").padStart(3, "0")} ${training.phase || "training"}`
    : "Training stopped";
  $("#contentState").textContent = status.content_checkpoint ? "Checkpoint ready" : "Missing";
  $("#waveState").textContent = decoder.ready ? "Checkpoint ready" : "Training pending";
  $("#contentCheckpoint").textContent = status.content_checkpoint || "Missing";
  $("#waveCheckpoint").textContent = decoder.checkpoint || decoder.reason;
  $("#trainingPhase").textContent = training.phase
    ? `E${String(training.epoch).padStart(3, "0")} ${training.phase}${training.step ? ` ${training.step}/${training.steps}` : ""}`
    : "Not running";
  $("#validationCosine").textContent = training.frame_cosine == null ? "Pending epoch validation" : training.frame_cosine.toFixed(4);
  $("#trainingLog").textContent = training.line || "No log";
  $("#f0Capability").textContent = decoder.supports_f0_conditioning ? `Enabled · ${decoder.f0_model || "FCPE"}` : "Pending conditioned checkpoint";
  $("#formantCapability").textContent = decoder.supports_formant_conditioning ? "Enabled" : "Pending conditioned checkpoint";
  $("#inputRateBadge").textContent = "Input device native";
  $("#outputRateBadge").textContent = decoder.sample_rate ? `Output ${(decoder.sample_rate / 1000).toFixed(1)}k` : "Output 44.1k";
  $("#pipelineLatency").textContent = decoder.ready ? "~47 ms/frame" : "-- ms";

  const pitchSupported = Boolean(decoder.supports_f0_conditioning);
  const formantSupported = Boolean(decoder.supports_formant_conditioning);
  $("#pitchControl").disabled = !pitchSupported;
  $("#formantControl").disabled = !formantSupported;
  $("#conditioningPill").className = `capability-pill ${pitchSupported || formantSupported ? "ready" : ""}`;
  $("#conditioningPill").textContent = pitchSupported || formantSupported ? "Conditioning ready" : "Checkpoint gated";
  $("#f0Status").className = `engine-status ${status.f0.installed ? "ready" : ""}`;
  $("#f0Status").textContent = status.f0.live_conditioning ? "Live conditioning" : "Analysis ready";
  $("#f0RuntimeState").textContent = status.f0.live_conditioning ? "FCPE live" : "FCPE profile";
  $("#startButton").disabled = !(decoder.ready && state.selectedVoicebank);
  $("#bypassButton").disabled = !state.streamReady;
  if (!state.streamReady) {
    $("#streamStatus").textContent = decoder.ready
      ? state.selectedVoicebank ? "Ready to start" : "Select a VoiceBank"
      : decoder.reason;
  }
}

async function refreshStatus() {
  try {
    state.status = await api("/api/status");
    applyStatus();
  } catch (error) {
    $("#runtimeDot").className = "status-dot error";
    $("#runtimeLabel").textContent = "Server unavailable";
    toast(error.message, "error");
  }
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  state.selectedVoicebank = state.settings.voicebank || "";
  $$("[data-setting]").forEach((control) => {
    const key = control.dataset.setting;
    if (!(key in state.settings)) return;
    if (control.type === "checkbox") control.checked = Boolean(state.settings[key]);
    else control.value = state.settings[key];
  });
  updateControlOutputs();
  updateChunkButtons();
}

let saveTimer = null;
function saveSettings() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    try {
      await api("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.settings),
      });
    } catch (error) {
      toast(error.message, "error");
    }
  }, 220);
}

function updateControlOutputs() {
  $("#pitchValue").textContent = `${Number($("#pitchControl").value).toFixed(1)} st`;
  $("#formantValue").textContent = `${Number($("#formantControl").value).toFixed(1)} st`;
  $("#wetValue").textContent = `${Math.round(Number($("#wetControl").value) * 100)}%`;
  $("#outputGainValue").textContent = `${Number($("#outputGainControl").value).toFixed(1)} dB`;
  $("#inputGainValue").textContent = `${Number($("#inputGainControl").value).toFixed(1)} dB`;
  $("#gateValue").textContent = `${Number($("#gateControl").value).toFixed(0)} dB`;
}

function updateChunkButtons() {
  $$("#chunkSize button").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.chunk) === Number(state.settings.chunk_ms || 5));
  });
}

function updateSinkSupport() {
  $("#sinkSupport").textContent = "setSinkId" in AudioContext.prototype
    ? "Selectable"
    : "System default";
}

async function refreshDevices() {
  try {
    const permission = await navigator.mediaDevices.getUserMedia({ audio: true });
    permission.getTracks().forEach((track) => track.stop());
    const devices = await navigator.mediaDevices.enumerateDevices();
    const inputs = devices.filter((device) => device.kind === "audioinput");
    const outputs = devices.filter((device) => device.kind === "audiooutput");
    const input = $("#inputDevice");
    const output = $("#outputDevice");
    input.innerHTML = '<option value="">Default input</option>';
    output.innerHTML = '<option value="">Default output</option>';
    inputs.forEach((device, index) => input.add(new Option(device.label || `Input ${index + 1}`, device.deviceId)));
    outputs.forEach((device, index) => output.add(new Option(device.label || `Output ${index + 1}`, device.deviceId)));
    input.value = state.settings.input_device || "";
    output.value = state.settings.output_device || "";
    updateSinkSupport();
    toast("Audio devices refreshed", "success");
  } catch (error) {
    toast(`Device access failed: ${error.message}`, "error");
  }
}

function setSelectedFile(file) {
  state.selectedFile = file;
  $("#selectedFileName").textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB` : "No file selected";
  if (file && !$("#profileName").value) {
    $("#profileName").value = file.name.replace(/\.[^.]+$/, "");
  }
  $("#buildButton").disabled = !(file && $("#profileName").value.trim());
}

async function buildVoicebank() {
  const name = $("#profileName").value.trim();
  if (!state.selectedFile || !name) return;
  const form = new FormData();
  form.append("file", state.selectedFile);
  form.append("name", name);
  form.append("device", state.settings.compute_device || "cpu");
  $("#buildButton").disabled = true;
  $("#buildState").textContent = "Uploading reference";
  $("#buildProgress i").style.width = "8%";
  try {
    const result = await api("/api/voicebanks", { method: "POST", body: form });
    pollJob(result.job_id);
  } catch (error) {
    $("#buildState").textContent = error.message;
    $("#buildProgress i").style.width = "0";
    $("#buildButton").disabled = false;
    toast(error.message, "error");
  }
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    $("#buildState").textContent = job.message;
    $("#buildProgress i").style.width = `${Math.round(job.progress * 100)}%`;
    if (job.status === "complete") {
      toast("VoiceBank profile created", "success");
      setSelectedFile(null);
      $("#profileName").value = "";
      await refreshVoicebanks();
      selectVoicebank(job.name);
      setView("voicebanks");
      return;
    }
    if (job.status === "failed") {
      $("#buildButton").disabled = false;
      toast(job.message, "error");
      return;
    }
    setTimeout(() => pollJob(jobId), 700);
  } catch (error) {
    toast(error.message, "error");
  }
}

async function deleteSelectedProfile() {
  const profile = currentVoicebank();
  if (!profile) return;
  if (!window.confirm(`Delete VoiceBank "${profile.id}"?`)) return;
  try {
    await api(`/api/voicebanks/${encodeURIComponent(profile.id)}`, { method: "DELETE" });
    toast("VoiceBank deleted", "success");
    await refreshVoicebanks();
    await refreshStatus();
  } catch (error) {
    toast(error.message, "error");
  }
}

function drawF0(result) {
  const canvas = $("#f0Chart");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#101317";
  context.fillRect(0, 0, canvas.width, canvas.height);
  const curve = result.curve || [];
  const nonzero = curve.map((point) => point.hz).filter((value) => value > 0);
  const maxHz = Math.max(400, ...nonzero);
  context.strokeStyle = "#2d343c";
  context.lineWidth = 1;
  for (let row = 1; row < 4; row += 1) {
    const y = canvas.height * row / 4;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(canvas.width, y);
    context.stroke();
  }
  context.strokeStyle = "#efb85d";
  context.lineWidth = 2;
  context.beginPath();
  let drawing = false;
  curve.forEach((point, index) => {
    if (!point.hz) {
      drawing = false;
      return;
    }
    const x = curve.length > 1 ? index / (curve.length - 1) * canvas.width : 0;
    const y = canvas.height - Math.min(1, point.hz / maxHz) * canvas.height * 0.88 - 12;
    if (drawing) context.lineTo(x, y);
    else context.moveTo(x, y);
    drawing = true;
  });
  context.stroke();
}

async function analyzeF0() {
  const profile = currentVoicebank();
  if (!profile) return;
  $("#analyzeF0Button").disabled = true;
  $("#analyzeF0Button").textContent = "Analyzing";
  const threshold = state.settings.f0_threshold ?? 0.006;
  const min = state.settings.f0_min ?? 50;
  const max = state.settings.f0_max ?? 1100;
  try {
    const result = await api(`/api/voicebanks/${encodeURIComponent(profile.id)}/f0?threshold=${threshold}&f0_min=${min}&f0_max=${max}`);
    drawF0(result);
    const stats = result.statistics;
    const values = [
      stats.median_hz == null ? "-- Hz" : `${stats.median_hz.toFixed(1)} Hz`,
      stats.p05_hz == null ? "--" : `${stats.p05_hz.toFixed(0)}–${stats.p95_hz.toFixed(0)} Hz`,
      `${(result.voiced_ratio * 100).toFixed(1)}%`,
      `${result.hop_ms.toFixed(1)} ms`,
    ];
    $$("#f0Metrics strong").forEach((node, index) => { node.textContent = values[index]; });
    setView("diagnostics");
    toast("FCPE analysis complete", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    $("#analyzeF0Button").disabled = false;
    $("#analyzeF0Button").textContent = "Analyze";
  }
}

function shiftHistory(history, samples) {
  if (!samples.length) return;
  const take = Math.min(history.length, samples.length);
  history.copyWithin(0, take);
  history.set(samples.subarray(samples.length - take), history.length - take);
}

function rmsDb(samples) {
  if (!samples.length) return -Infinity;
  let sum = 0;
  for (const sample of samples) sum += sample * sample;
  return 20 * Math.log10(Math.max(1e-8, Math.sqrt(sum / samples.length)));
}

function updateMeter(kind, samples) {
  const db = rmsDb(samples);
  const width = Math.max(0, Math.min(100, (db + 60) / 60 * 100));
  $(`#${kind}Meter`).style.width = `${width}%`;
  $(`#${kind}Db`).textContent = Number.isFinite(db) ? `${db.toFixed(0)} dB` : "-inf";
}

function enqueueDry(samples) {
  state.dryChunks.push(samples);
  state.dryQueuedSamples += samples.length;
  const maximum = 16000 * 2;
  while (state.dryQueuedSamples > maximum && state.dryChunks.length) {
    const excess = state.dryQueuedSamples - maximum;
    const available = state.dryChunks[0].length - state.dryChunkOffset;
    const discard = Math.min(excess, available);
    state.dryChunkOffset += discard;
    state.dryQueuedSamples -= discard;
    if (state.dryChunkOffset === state.dryChunks[0].length) {
      state.dryChunks.shift();
      state.dryChunkOffset = 0;
    }
  }
}

function takeDry(sampleCount) {
  const output = new Float32Array(sampleCount);
  let written = 0;
  while (written < sampleCount && state.dryChunks.length) {
    const chunk = state.dryChunks[0];
    const available = chunk.length - state.dryChunkOffset;
    const take = Math.min(sampleCount - written, available);
    output.set(
      chunk.subarray(state.dryChunkOffset, state.dryChunkOffset + take),
      written,
    );
    state.dryChunkOffset += take;
    state.dryQueuedSamples -= take;
    written += take;
    if (state.dryChunkOffset === chunk.length) {
      state.dryChunks.shift();
      state.dryChunkOffset = 0;
    }
  }
  return output;
}

function applyNoiseGate(samples) {
  const threshold = Number(state.settings.noise_gate_db ?? -55);
  const target = rmsDb(samples) >= threshold ? 1 : 0;
  const attack = Math.exp(-1 / (0.003 * 16000));
  const release = Math.exp(-1 / (0.035 * 16000));
  for (let index = 0; index < samples.length; index += 1) {
    const coefficient = target > state.gateEnvelope ? attack : release;
    state.gateEnvelope = target + coefficient * (state.gateEnvelope - target);
    samples[index] *= state.gateEnvelope;
  }
}

function drawLive() {
  const canvas = $("#liveWaveform");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#101317";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "#2d343c";
  context.beginPath();
  context.moveTo(0, canvas.height / 2);
  context.lineTo(canvas.width, canvas.height / 2);
  context.stroke();
  const draw = (samples, color, scale) => {
    context.strokeStyle = color;
    context.lineWidth = 1.2;
    context.beginPath();
    samples.forEach((sample, index) => {
      const x = index / (samples.length - 1) * canvas.width;
      const y = canvas.height / 2 - sample * canvas.height * scale;
      if (index) context.lineTo(x, y);
      else context.moveTo(x, y);
    });
    context.stroke();
  };
  draw(state.inputHistory, "#63d18b", 0.31);
  draw(state.outputHistory, "#68a9ff", 0.31);
  requestAnimationFrame(drawLive);
}

async function scheduleOutput(samples) {
  if (!state.outputContext) return;
  const outputRate = state.outputSampleRate || 44100;
  const outputGain = 10 ** ((state.settings.output_gain_db || 0) / 20);
  const mix = Math.max(0, Math.min(1, Number(state.settings.wet ?? 1)));
  const wetGainValue = state.bypass ? 0 : Math.sin(mix * Math.PI / 2) * outputGain;
  const dryGainValue = state.bypass ? outputGain : Math.cos(mix * Math.PI / 2) * outputGain;
  const convertedBuffer = state.outputContext.createBuffer(1, samples.length, outputRate);
  convertedBuffer.copyToChannel(samples, 0);
  const convertedSource = state.outputContext.createBufferSource();
  const convertedGain = state.outputContext.createGain();
  convertedSource.buffer = convertedBuffer;
  convertedGain.gain.value = wetGainValue;
  convertedSource.connect(convertedGain).connect(state.outputContext.destination);
  const drySampleCount = Math.round(convertedBuffer.duration * 16000);
  const drySamples = takeDry(drySampleCount);
  const dryBuffer = state.outputContext.createBuffer(1, drySamples.length, 16000);
  dryBuffer.copyToChannel(drySamples, 0);
  const drySource = state.outputContext.createBufferSource();
  const dryGain = state.outputContext.createGain();
  drySource.buffer = dryBuffer;
  dryGain.gain.value = dryGainValue;
  drySource.connect(dryGain).connect(state.outputContext.destination);
  const now = state.outputContext.currentTime;
  state.nextOutputTime = Math.max(now + 0.005, state.nextOutputTime);
  convertedSource.start(state.nextOutputTime);
  drySource.start(state.nextOutputTime);
  state.nextOutputTime += convertedBuffer.duration;
  const monitored = wetGainValue > 0 ? samples : drySamples;
  shiftHistory(state.outputHistory, monitored);
  updateMeter("output", monitored);
}

async function startStream() {
  if (state.stream) {
    await stopStream();
    return;
  }
  if (!state.status?.decoder.ready) {
    toast(state.status?.decoder.reason || "Decoder is not ready", "error");
    return;
  }
  try {
    const constraints = state.settings.input_device
      ? { audio: { deviceId: { exact: state.settings.input_device }, channelCount: 1 } }
      : { audio: { channelCount: 1 } };
    state.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
    state.inputContext = new AudioContext({ latencyHint: "interactive" });
    state.outputContext = new AudioContext({ sampleRate: 44100, latencyHint: "interactive" });
    if (state.settings.output_device && state.outputContext.setSinkId) {
      await state.outputContext.setSinkId(state.settings.output_device);
    }
    await state.inputContext.audioWorklet.addModule("/static/capture-worklet.js");
    const chunkSamples = Math.max(40, Math.round(16000 * (state.settings.chunk_ms || 5) / 1000));
    state.worklet = new AudioWorkletNode(state.inputContext, "astrape-capture", {
      processorOptions: { targetRate: 16000, chunkSamples },
    });
    const source = state.inputContext.createMediaStreamSource(state.mediaStream);
    const mute = state.inputContext.createGain();
    mute.gain.value = 0;
    source.connect(state.worklet).connect(mute).connect(state.inputContext.destination);

    const protocol = location.protocol === "https:" ? "wss" : "ws";
    state.stream = new WebSocket(`${protocol}://${location.host}/api/stream`);
    state.stream.binaryType = "arraybuffer";
    state.stream.onopen = () => {
      state.stream.send(JSON.stringify({
        ...state.settings,
        voicebank: state.selectedVoicebank,
      }));
      $("#streamStatus").textContent = "Loading models";
    };
    state.stream.onmessage = async (event) => {
      if (event.data instanceof ArrayBuffer) {
        await scheduleOutput(new Float32Array(event.data));
        return;
      }
      const message = JSON.parse(event.data);
      if (message.type === "ready") {
        state.streamReady = true;
        state.outputSampleRate = Number(message.output_sample_rate) || 44100;
        $("#outputRateBadge").textContent = `Output ${(state.outputSampleRate / 1000).toFixed(1)}k`;
        $("#streamDot").className = "status-dot ready";
        $("#streamStatus").textContent = "Live";
        $("#startButton span:last-child").textContent = "Stop conversion";
        $("#bypassButton").disabled = false;
        state.worklet.port.onmessage = ({ data }) => {
          const input = new Float32Array(data);
          const gain = 10 ** ((state.settings.input_gain_db || 0) / 20);
          for (let index = 0; index < input.length; index += 1) input[index] *= gain;
          applyNoiseGate(input);
          shiftHistory(state.inputHistory, input);
          updateMeter("input", input);
          enqueueDry(input);
          if (state.stream?.readyState === WebSocket.OPEN) state.stream.send(input.buffer);
        };
      } else if (message.type === "error") {
        toast(message.message, "error");
        stopStream();
      }
    };
    state.stream.onerror = () => toast("Streaming connection failed", "error");
    state.stream.onclose = () => {
      if (state.stream) stopStream();
    };
  } catch (error) {
    toast(error.message, "error");
    await stopStream();
  }
}

async function stopStream() {
  const socket = state.stream;
  state.stream = null;
  state.streamReady = false;
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "flush" }));
    setTimeout(() => socket.close(), 120);
  }
  state.worklet?.disconnect();
  state.worklet = null;
  state.mediaStream?.getTracks().forEach((track) => track.stop());
  state.mediaStream = null;
  await state.inputContext?.close().catch(() => {});
  await state.outputContext?.close().catch(() => {});
  state.inputContext = null;
  state.outputContext = null;
  state.nextOutputTime = 0;
  state.outputSampleRate = 44100;
  state.dryChunks = [];
  state.dryChunkOffset = 0;
  state.dryQueuedSamples = 0;
  state.gateEnvelope = 0;
  state.bypass = false;
  $("#streamDot").className = "status-dot";
  $("#streamStatus").textContent = state.status?.decoder.ready ? "Ready to start" : "Waiting for decoder checkpoint";
  $("#startButton span:last-child").textContent = "Start conversion";
  $("#bypassButton").disabled = true;
  $("#bypassButton").setAttribute("aria-pressed", "false");
  $("#bypassButton").textContent = "Bypass";
}

function bindEvents() {
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => setView(item.dataset.view)));
  $$("[data-open-view]").forEach((item) => item.addEventListener("click", () => setView(item.dataset.openView)));
  $("#refreshButton").addEventListener("click", async () => {
    await Promise.all([refreshStatus(), refreshVoicebanks()]);
    toast("Status refreshed", "success");
  });
  $("#refreshDevicesButton").addEventListener("click", refreshDevices);
  $("#dropZone").addEventListener("click", () => $("#voiceFileInput").click());
  $("#voiceFileInput").addEventListener("change", (event) => setSelectedFile(event.target.files[0]));
  ["dragenter", "dragover"].forEach((name) => $("#dropZone").addEventListener(name, (event) => {
    event.preventDefault();
    $("#dropZone").classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((name) => $("#dropZone").addEventListener(name, (event) => {
    event.preventDefault();
    $("#dropZone").classList.remove("dragging");
  }));
  $("#dropZone").addEventListener("drop", (event) => setSelectedFile(event.dataTransfer.files[0]));
  $("#profileName").addEventListener("input", () => {
    $("#buildButton").disabled = !(state.selectedFile && $("#profileName").value.trim());
  });
  $("#buildButton").addEventListener("click", buildVoicebank);
  $("#deleteProfileButton").addEventListener("click", deleteSelectedProfile);
  $("#analyzeF0Button").addEventListener("click", analyzeF0);
  $("#playReferenceButton").addEventListener("click", () => {
    const profile = currentVoicebank();
    if (!profile?.preview_url) return;
    const player = $("#referencePlayer");
    if (player.paused) {
      player.src = profile.preview_url;
      player.play();
      $("#playReferenceButton").textContent = "■";
    } else {
      player.pause();
      player.currentTime = 0;
      $("#playReferenceButton").textContent = "▶";
    }
  });
  $("#referencePlayer").addEventListener("ended", () => { $("#playReferenceButton").textContent = "▶"; });
  $("#startButton").addEventListener("click", startStream);
  $("#bypassButton").addEventListener("click", () => {
    state.bypass = !state.bypass;
    $("#bypassButton").setAttribute("aria-pressed", String(state.bypass));
    $("#bypassButton").textContent = state.bypass ? "Bypassed" : "Bypass";
  });
  $$("[data-setting]").forEach((control) => {
    control.addEventListener("input", () => {
      const key = control.dataset.setting;
      state.settings[key] = control.type === "checkbox"
        ? control.checked
        : control.type === "range" || control.type === "number"
          ? Number(control.value)
          : control.value;
      updateControlOutputs();
      saveSettings();
    });
  });
  $$("#chunkSize button").forEach((button) => button.addEventListener("click", () => {
    state.settings.chunk_ms = Number(button.dataset.chunk);
    updateChunkButtons();
    saveSettings();
  }));
}

async function boot() {
  bindEvents();
  drawLive();
  updateSinkSupport();
  await loadSettings();
  await Promise.all([refreshStatus(), refreshVoicebanks()]);
  applyStatus();
  setInterval(refreshStatus, 10000);
}

boot().catch((error) => toast(error.message, "error"));
