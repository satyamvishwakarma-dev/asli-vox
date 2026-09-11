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

function encodeFloat32ToBase64(channelData) {
    const buffer = new Uint8Array(channelData.buffer);
    let binary = '';
    for (let i = 0; i < buffer.byteLength; i++) {
        binary += String.fromCharCode(buffer[i]);
    }
    return btoa(binary);
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

async function startPipeline() {
    try {
        await ensureSocketConnection();

        stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            },
            video: false
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        if (micBtn) {
            micBtn.classList.add('active');
            if (micBtnText) micBtnText.innerText = "Halt interceptor";
        }

        const source = audioContext.createMediaStreamSource(stream);
        // 4096 samples = ~256ms chunk @ 16kHz
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (e) => {
            const channelData = e.inputBuffer.getChannelData(0);
            drawOscilloscope(channelData);

            if (socket && socket.readyState === WebSocket.OPEN) {
                lastTimestamp = performance.now();
                socket.send(JSON.stringify({ audio: encodeFloat32ToBase64(channelData) }));
            }
        };

        source.connect(processor);

        const muteGain = audioContext.createGain();
        muteGain.gain.value = 0.0;
        processor.connect(muteGain);
        muteGain.connect(audioContext.destination);

    } catch (err) {
        alert("Audio initialization error: " + err.message);
        await stopPipeline();
    }
}

async function stopPipeline() {
    if (processor) {
        try { processor.disconnect(); } catch (e) {}
        processor = null;
    }

    if (stream) {
        try { stream.getTracks().forEach(track => track.stop()); } catch (e) {}
        stream = null;
    }

    if (audioContext) {
        if (audioContext.state !== 'closed') {
            try { await audioContext.close(); } catch (e) {}
        }
        audioContext = null;
    }

    if (socket) {
        socket.onclose = null;
        if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
            socket.close();
        }
        socket = null;
    }

    if (micBtn) {
        micBtn.classList.remove('active');
        if (micBtnText) micBtnText.innerText = "Engage sensors";
    }

    if (systemDot) systemDot.classList.remove('active');
    if (systemStatusText) systemStatusText.innerText = "CORE OFFLINE";
    if (vadStatus) vadStatus.innerText = "Standby";
    if (radarSweep) radarSweep.style.display = "none";
    if (latencyVal) latencyVal.innerText = "-- ms";

    if (verdictBanner) {
        verdictBanner.className = 'verdict-banner';
        if (verdictTitle) verdictTitle.innerText = "System ready";
        if (verdictSub) verdictSub.innerText = "Awaiting acoustic payload";
    }

    if (probBar) {
        probBar.style.width = "0%";
        probBar.style.backgroundColor = "var(--verified-green)";
        probBar.style.boxShadow = "0 0 10px var(--verified-green)";
    }
    if (probPercent) probPercent.innerText = "0.00%";
}

function updateTelemetry(data) {
    if (vadStatus) vadStatus.innerText = data.vad || "ACTIVE";

    if (data.verdict === 'silent') {
        if (verdictBanner) {
            verdictBanner.className = 'verdict-banner';
            if (verdictTitle) verdictTitle.innerText = "Listening / Silence";
            if (verdictSub) verdictSub.innerText = "Waiting for audible speech...";
        }
        if (probBar) probBar.style.width = "0%";
        if (probPercent) probPercent.innerText = "0.00%";
        return;
    }

    const spoofProb = data.confidence;
    const percentStr = (spoofProb * 100).toFixed(2);
    if (probPercent) probPercent.innerText = `${percentStr}%`;
    if (probBar) probBar.style.width = `${percentStr}%`;

    if (data.verdict === 'spoof') {
        if (verdictBanner) {
            verdictBanner.className = 'verdict-banner spoof';
            if (verdictTitle) verdictTitle.innerText = "Synthetic spoof detected";
            if (verdictSub) verdictSub.innerText = `Spoof probability: ${percentStr}% (Threshold: ${data.threshold})`;
        }
        if (probBar) {
            probBar.style.backgroundColor = "var(--alert-red)";
            probBar.style.boxShadow = "0 0 10px var(--alert-red)";
        }

        if (data.heatmap && heatmapImg && heatmapPlaceholder) {
            heatmapImg.src = "data:image/png;base64," + data.heatmap;
            heatmapImg.style.display = "block";
            heatmapPlaceholder.style.display = "none";
        }
    } else {
        if (verdictBanner) {
            verdictBanner.className = 'verdict-banner bonafide';
            if (verdictTitle) verdictTitle.innerText = "Bonafide voice pattern";
            if (verdictSub) verdictSub.innerText = `Authentic human speech confirmed (${(100 - spoofProb * 100).toFixed(2)}% confidence)`;
        }
        if (probBar) {
            probBar.style.backgroundColor = "var(--verified-green)";
            probBar.style.boxShadow = "0 0 10px var(--verified-green)";
        }

        if (heatmapImg && heatmapPlaceholder) {
            heatmapImg.style.display = "none";
            heatmapPlaceholder.style.display = "flex";
        }
    }
}