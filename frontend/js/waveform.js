// waveform.js – Live Audio Waveform Visualizer using Web Audio API
// -----------------------------------------------------------------------------

export function initWaveform() {
    const canvas = document.getElementById('waveform');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let audioCtx = null;
    let analyser = null;
    let source = null;
    let animId = null;
    let isRunning = false;

    function resizeCanvas() {
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * (window.devicePixelRatio || 1);
        canvas.height = rect.height * (window.devicePixelRatio || 1);
        ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
    }

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Render idle state
    function drawIdle() {
        const w = canvas.getBoundingClientRect().width;
        const h = canvas.getBoundingClientRect().height;
        ctx.clearRect(0, 0, w, h);

        // Subtle grid / center line
        ctx.beginPath();
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        ctx.moveTo(0, h / 2);
        ctx.lineTo(w, h / 2);
        ctx.stroke();

        // Ambient gentle wave
        const time = Date.now() * 0.002;
        ctx.beginPath();
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 2;
        for (let x = 0; x < w; x += 4) {
            const y = (h / 2) + Math.sin(x * 0.03 + time) * 3;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        if (!isRunning) {
            animId = requestAnimationFrame(drawIdle);
        }
    }

    drawIdle();

    window.__waveform = {
        start: function(stream) {
            if (isRunning) return;
            isRunning = true;
            if (animId) cancelAnimationFrame(animId);

            try {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 1024;
                analyser.smoothingTimeConstant = 0.8;

                source = audioCtx.createMediaStreamSource(stream);
                source.connect(analyser);

                const bufferLength = analyser.fftSize;
                const dataArray = new Uint8Array(bufferLength);

                function render() {
                    if (!isRunning) return;
                    animId = requestAnimationFrame(render);

                    analyser.getByteTimeDomainData(dataArray);

                    const w = canvas.getBoundingClientRect().width;
                    const h = canvas.getBoundingClientRect().height;

                    ctx.fillStyle = '#090d16';
                    ctx.fillRect(0, 0, w, h);

                    // Grid line
                    ctx.beginPath();
                    ctx.strokeStyle = '#1a2234';
                    ctx.lineWidth = 1;
                    ctx.moveTo(0, h / 2);
                    ctx.lineTo(w, h / 2);
                    ctx.stroke();

                    // Waveform line
                    ctx.beginPath();
                    ctx.lineWidth = 2.5;

                    // Neon gradient stroke
                    const grad = ctx.createLinearGradient(0, 0, w, 0);
                    grad.addColorStop(0, '#00e5ff');
                    grad.addColorStop(0.5, '#00ff99');
                    grad.addColorStop(1, '#00e5ff');
                    ctx.strokeStyle = grad;

                    const sliceWidth = w / bufferLength;
                    let x = 0;

                    for (let i = 0; i < bufferLength; i++) {
                        const v = dataArray[i] / 128.0; // [0, 2]
                        const y = (v * h) / 2;

                        if (i === 0) ctx.moveTo(x, y);
                        else ctx.lineTo(x, y);

                        x += sliceWidth;
                    }

                    ctx.lineTo(w, h / 2);
                    ctx.stroke();
                }

                render();
            } catch (err) {
                console.error('Waveform audio context error:', err);
                isRunning = false;
                drawIdle();
            }
        },

        stop: function() {
            isRunning = false;
            if (animId) cancelAnimationFrame(animId);
            if (source) {
                try { source.disconnect(); } catch (e) {}
                source = null;
            }
            if (audioCtx && audioCtx.state !== 'closed') {
                try { audioCtx.close(); } catch (e) {}
                audioCtx = null;
            }
            drawIdle();
        }
    };
}