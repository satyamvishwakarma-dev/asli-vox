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

// Setup High-DPI canvas
function resizeCanvas() {
    canvas.width = canvas.parentElement.clientWidth * window.devicePixelRatio;
    canvas.height = canvas.parentElement.clientHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

function drawOscilloscope(dataArray) {
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
        const v = dataArray[i]; // Amplitude: -1.0 to 1.0
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

micBtn.addEventListener('click', async () => {
    if (micBtn.classList.contains('active')) {
        stopPipeline();
    } else {
        await startPipeline();
    }
});

async function startPipeline() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        audioContext = new AudioContext({ sampleRate: 16000 });

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

        socket.onopen = () => {
            systemDot.classList.add('active');
            systemStatusText.innerText = "STREAM ONLINE";
            micBtn.classList.add('active');
            micBtnText.innerText = "Halt interceptor";
            vadStatus.innerText = "Active";
            radarSweep.style.display = "block";
        };

        socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const now = performance.now();
            if (lastTimestamp) {
                const latency = Math.round(now - lastTimestamp);
                latencyVal.innerText = `${latency} ms`;
            }

            framesProcessed++;
            frameCounter.innerText = framesProcessed;
            windowSizeText.innerText = `${data.buffered_seconds}s`;

            updateTelemetry(data);
        };

        socket.onclose = () => {
            stopPipeline();
        };

        const source = audioContext.createMediaStreamSource(stream);
        // ScriptProcessor: 4096 samples = 256ms chunk @ 16kHz
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (e) => {
            const channelData = e.inputBuffer.getChannelData(0);
            drawOscilloscope(channelData);

            if (socket && socket.readyState === WebSocket.OPEN) {
                lastTimestamp = performance.now();
                // Send raw PCM Float32 buffer as base64 string
                const buffer = new Uint8Array(channelData.buffer);
                let binary = '';
                for (let i = 0; i < buffer.byteLength; i++) {
                    binary += String.fromCharCode(buffer[i]);
                }
                socket.send(JSON.stringify({ audio: btoa(binary) }));
            }
        };

        source.connect(processor);
        processor.connect(audioContext.destination);

    } catch (err) {
        alert("Failed to initialize audio input: " + err.message);
        stopPipeline();
    }
}

function stopPipeline() {
    if (processor) processor.disconnect();
    if (audioContext) audioContext.close();
    if (stream) stream.getTracks().forEach(track => track.stop());
    if (socket) socket.close();

    micBtn.classList.remove('active');
    micBtnText.innerText = "Engage sensors";
    systemDot.classList.remove('active');
    systemStatusText.innerText = "CORE OFFLINE";
    vadStatus.innerText = "Standby";
    radarSweep.style.display = "none";
    latencyVal.innerText = "-- ms";

    verdictBanner.className = 'verdict-banner';
    verdictTitle.innerText = "System ready";
    verdictSub.innerText = "Awaiting acoustic payload";
    probBar.style.width = "0%";
    probBar.style.backgroundColor = "var(--verified-green)";
    probBar.style.boxShadow = "0 0 10px var(--verified-green)";
    probPercent.innerText = "0.00%";
}

function updateTelemetry(data) {
    const spoofProb = data.confidence;
    const percentStr = (spoofProb * 100).toFixed(2);
    probPercent.innerText = `${percentStr}%`;
    probBar.style.width = `${percentStr}%`;

    if (data.verdict === 'spoof') {
        verdictBanner.className = 'verdict-banner spoof';
        verdictTitle.innerText = "Synthetic spoof detected";
        verdictSub.innerText = `Critical anomaly: risk index exceeds operating barrier (${data.threshold})`;
        probBar.style.backgroundColor = "var(--alert-red)";
        probBar.style.boxShadow = "0 0 10px var(--alert-red)";

        if (data.heatmap) {
            heatmapImg.src = "data:image/png;base64," + data.heatmap;
            heatmapImg.style.display = "block";
            heatmapPlaceholder.style.display = "none";
        }
    } else {
        verdictBanner.className = 'verdict-banner bonafide';
        verdictTitle.innerText = "Bonafide voice pattern";
        verdictSub.innerText = "Acoustic envelope matches authentic human vocal resonance.";
        probBar.style.backgroundColor = "var(--verified-green)";
        probBar.style.boxShadow = "0 0 10px var(--verified-green)";

        heatmapImg.style.display = "none";
        heatmapPlaceholder.style.display = "flex";
    }
}
