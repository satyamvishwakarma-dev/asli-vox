const micBtn = document.getElementById('micBtn');
const micBtnText = document.getElementById('micBtnText');
const systemDot = document.getElementById('systemDot');
const systemStatusText = document.getElementById('systemStatusText');

const verdictBanner = document.getElementById('verdictBanner');
const verdictTitle = document.getElementById('verdictTitle');
const verdictSub = document.getElementById('verdictSub');

const probBar = document.getElementById('probBar');
const probPercent = document.getElementById('probPercent');
const windowSizeText = document.getElementById('windowSizeText');
const latencyVal = document.getElementById('latencyVal');
const frameCounter = document.getElementById('frameCounter');
const vadStatus = document.getElementById('vadStatus');

const heatmapImg = document.getElementById('heatmapImg');
const heatmapPlaceholder = document.getElementById('heatmapPlaceholder');
const radarSweep = document.getElementById('radarSweep');

const canvas = document.getElementById('waveformCanvas');
const ctx = canvas.getContext('2d');

let socket = null;
let audioContext = null;
let processor = null;
let stream = null;
let framesProcessed = 0;
let lastTimestamp = 0;
let filePlaybackInterval = null;

function resizeCanvas() {
    if (!canvas || !canvas.parentElement) return;
    canvas.width = canvas.parentElement.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.parentElement.clientHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawOscilloscope(dataArray) {
    if (!canvas || !canvas.parentElement) return;
    const width = canvas.parentElement.clientWidth;
    const height = canvas.parentElement.clientHeight;

    ctx.clearRect(0, 0, width, height);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = '#0aeaff';
    ctx.shadowBlur = 10;
    ctx.shadowColor = 'rgba(10, 234, 255, 0.8)';
    ctx.beginPath();

    const sliceWidth = width / dataArray.length;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i];
        const y = ((v + 1) / 2) * height;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        x += sliceWidth;
    }

    ctx.lineTo(width, height / 2);
    ctx.stroke();
    ctx.shadowBlur = 0;
}

function encodeFloat32ToBase64(float32Array) {
    const uint8Array = new Uint8Array(float32Array.buffer, float32Array.byteOffset, float32Array.byteLength);
    let binary = '';
    const len = uint8Array.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(uint8Array[i]);
    }
    return window.btoa(binary);
}

function ensureSocketConnection() {
    return new Promise((resolve, reject) => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            resolve(socket);
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

        socket.onopen = () => {
            if (systemDot) systemDot.classList.add('active');
            if (systemStatusText) systemStatusText.innerText = "STREAM ONLINE";
            if (vadStatus) vadStatus.innerText = "Active";
            if (radarSweep) radarSweep.style.display = "block";
            resolve(socket);
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const now = performance.now();
            if (lastTimestamp && latencyVal) {
                latencyVal.innerText = `${Math.round(now - lastTimestamp)} ms`;
            }

            framesProcessed++;
            if (frameCounter) frameCounter.innerText = framesProcessed;
            if (windowSizeText) windowSizeText.innerText = `${data.buffered_seconds}s`;

            // Debug log every response
            console.log("[AsliVox Telemetry]", data);
            updateTelemetry(data);
        };

        socket.onerror = (err) => reject(err);
        socket.onclose = () => stopPipeline();
    });
}

if (micBtn) {
    micBtn.addEventListener('click', async () => {
        if (micBtn.classList.contains('active')) {
            await stopPipeline();
        } else {
            await startPipeline();
        }
    });
}

// Clean linear downsampler to 16kHz
function downsampleTo16k(inputData, inputSampleRate) {
    if (inputSampleRate === 16000) return inputData;
    const ratio = inputSampleRate / 16000;
    const newLength = Math.round(inputData.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetInput = 0;
    while (offsetResult < result.length) {
        const nextOffsetInput = Math.round((offsetResult + 1) * ratio);
        let accum = 0, count = 0;
        for (let i = offsetInput; i < nextOffsetInput && i < inputData.length; i++) {
            accum += inputData[i];
            count++;
        }
        result[offsetResult] = count > 0 ? accum / count : inputData[offsetInput];
        offsetResult++;
        offsetInput = nextOffsetInput;
    }
    return result;
}
// Clean linear downsampler to 16kHz
function downsampleTo16k(inputData, inputSampleRate) {
    if (inputSampleRate === 16000) return inputData;
    const ratio = inputSampleRate / 16000;
    const newLength = Math.round(inputData.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetInput = 0;
    while (offsetResult < result.length) {
        const nextOffsetInput = Math.round((offsetResult + 1) * ratio);
        let accum = 0, count = 0;
        for (let i = offsetInput; i < nextOffsetInput && i < inputData.length; i++) {
            accum += inputData[i];
            count++;
        }
        result[offsetResult] = count > 0 ? accum / count : inputData[offsetInput];
        offsetResult++;
        offsetInput = nextOffsetInput;
    }
    return result;
}

function encodeFloat32ToBase64(float32Array) {
    const uint8Array = new Uint8Array(float32Array.buffer, float32Array.byteOffset, float32Array.byteLength);
    let binary = '';
    const len = uint8Array.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(uint8Array[i]);
    }
    return window.btoa(binary);
}

async function startPipeline() {
    try {
        await ensureSocketConnection();

        // Pin directly to window to prevent V8 Garbage Collection drops
        window._asliStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: false,
                autoGainControl: false,
                noiseSuppression: false
            },
            video: false
        });

        window._asliAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (window._asliAudioCtx.state === 'suspended') {
            await window._asliAudioCtx.resume();
        }

        const nativeSR = window._asliAudioCtx.sampleRate;
        console.log(`[AsliVox] Capturing audio at native hardware rate: ${nativeSR} Hz`);

        if (micBtn) {
            micBtn.classList.add('active');
            if (micBtnText) micBtnText.innerText = "Halt interceptor";
        }

        window._asliSource = window._asliAudioCtx.createMediaStreamSource(window._asliStream);
        
        // 4096 samples buffer
        window._asliProcessor = window._asliAudioCtx.createScriptProcessor(4096, 1, 1);

        window._asliProcessor.onaudioprocess = (e) => {
            const rawChannel = e.inputBuffer.getChannelData(0);
            drawOscilloscope(rawChannel);

            const downsampled16k = downsampleTo16k(rawChannel, nativeSR);

            if (socket && socket.readyState === WebSocket.OPEN) {
                lastTimestamp = performance.now();
                socket.send(JSON.stringify({ audio: encodeFloat32ToBase64(downsampled16k) }));
            }
        };

        window._asliSource.connect(window._asliProcessor);

        // A live gain path is strictly required by Chromium to keep the processor thread active
        window._asliMuteGain = window._asliAudioCtx.createGain();
        window._asliMuteGain.gain.value = 0.0;
        window._asliProcessor.connect(window._asliMuteGain);
        window._asliMuteGain.connect(window._asliAudioCtx.destination);

    } catch (err) {
        alert("Audio initialization error: " + err.message);
        await stopPipeline();
    }
}

async function stopPipeline() {
    if (filePlaybackInterval) {
        clearInterval(filePlaybackInterval);
        filePlaybackInterval = null;
    }

    if (window._asliProcessor) {
        window._asliProcessor.disconnect();
        window._asliProcessor.onaudioprocess = null;
        window._asliProcessor = null;
    }

    if (window._asliSource) {
        window._asliSource.disconnect();
        window._asliSource = null;
    }

    if (window._asliStream) {
        window._asliStream.getTracks().forEach(track => track.stop());
        window._asliStream = null;
    }

    if (window._asliAudioCtx) {
        await window._asliAudioCtx.close();
        window._asliAudioCtx = null;
    }

    if (micBtn) {
        micBtn.classList.remove('active');
        if (micBtnText) micBtnText.innerText = "Engage sensors";
    }
}

function updateTelemetry(data) {
    if (vadStatus) vadStatus.innerText = "STREAMING";

    const riskScore = data.confidence;
    const percentStr = (riskScore * 100).toFixed(1);
    if (probPercent) probPercent.innerText = `${percentStr}%`;
    if (probBar) probBar.style.width = `${percentStr}%`;

    if (data.verdict === 'spoof') {
        if (verdictBanner) {
            verdictBanner.className = 'verdict-banner spoof';
            if (verdictTitle) verdictTitle.innerText = "SYNTHETIC SPOOF DETECTED";
            if (verdictSub) verdictSub.innerText = `CRITICAL ANOMALY: LLR Margin ${data.margin} (Threshold: ${data.threshold})`;
        }
        if (probBar) {
            probBar.style.backgroundColor = "var(--alert-red)";
            probBar.style.boxShadow = "0 0 12px var(--alert-red)";
        }

        if (data.heatmap && heatmapImg && heatmapPlaceholder) {
            heatmapImg.src = "data:image/png;base64," + data.heatmap;
            heatmapImg.style.display = "block";
            heatmapPlaceholder.style.display = "none";
        }
    } else {
        if (verdictBanner) {
            verdictBanner.className = 'verdict-banner bonafide';
            if (verdictTitle) verdictTitle.innerText = "BONAFIDE VOICE PATTERN";
            if (verdictSub) verdictSub.innerText = `Human vocal harmonics verified (LLR Margin: ${data.margin} < ${data.threshold})`;
        }
        if (probBar) {
            probBar.style.backgroundColor = "var(--verified-green)";
            probBar.style.boxShadow = "0 0 12px var(--verified-green)";
        }

        if (heatmapImg && heatmapPlaceholder) {
            heatmapImg.style.display = "none";
            heatmapPlaceholder.style.display = "flex";
        }
    }
}
const audioFileInput = document.getElementById('audioFileInput');

if (audioFileInput) {
    audioFileInput.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        if (verdictBanner) {
            verdictBanner.className = 'verdict-banner';
            if (verdictTitle) verdictTitle.innerText = "ANALYZING FILE...";
            if (verdictSub) verdictSub.innerText = `Evaluating acoustic resonance for ${file.name}`;
        }

        try {
            const res = await fetch('/api/verify-file', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            updateTelemetry(data);
        } catch (err) {
            alert("Analysis failed: " + err.message);
        }
    });
}