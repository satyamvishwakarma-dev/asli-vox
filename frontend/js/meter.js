// meter.js – Circular Authenticity Meter (green-to-red gradient by confidence)
// -----------------------------------------------------------------------------

export function initMeter() {
    const meterContainer = document.getElementById('meter');
    const meterText = document.getElementById('meter-text');
    if (!meterContainer) return;

    // Reset container and create SVG
    meterContainer.innerHTML = '';

    const size = 160;
    const strokeWidth = 12;
    const radius = (size - strokeWidth) / 2;
    const center = size / 2;
    const circumference = 2 * Math.PI * radius;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', size);
    svg.setAttribute('height', size);
    svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
    svg.style.transform = 'rotate(-90deg)'; // Start stroke from top

    // Defs for gradients
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    defs.innerHTML = `
        <linearGradient id="meterGreen" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#00e5ff" />
            <stop offset="100%" stop-color="#00ff99" />
        </linearGradient>
        <linearGradient id="meterRed" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#ff9900" />
            <stop offset="100%" stop-color="#ff3366" />
        </linearGradient>
    `;
    svg.appendChild(defs);

    // Background circle track
    const track = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    track.setAttribute('cx', center);
    track.setAttribute('cy', center);
    track.setAttribute('r', radius);
    track.setAttribute('fill', 'none');
    track.setAttribute('stroke', '#1c2438');
    track.setAttribute('stroke-width', strokeWidth);
    svg.appendChild(track);

    // Foreground animated indicator circle
    const progress = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    progress.setAttribute('cx', center);
    progress.setAttribute('cy', center);
    progress.setAttribute('r', radius);
    progress.setAttribute('fill', 'none');
    progress.setAttribute('stroke', 'url(#meterGreen)');
    progress.setAttribute('stroke-width', strokeWidth);
    progress.setAttribute('stroke-linecap', 'round');
    progress.style.strokeDasharray = `${circumference}`;
    progress.style.strokeDashoffset = `${circumference * 0.5}`; // 50% initial
    progress.style.transition = 'stroke-dashoffset 0.6s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.4s ease';
    svg.appendChild(progress);

    meterContainer.appendChild(svg);

    // Inner score display
    const labelDiv = document.createElement('div');
    labelDiv.className = 'meter-inner-label';
    labelDiv.style.position = 'absolute';
    labelDiv.style.top = '50%';
    labelDiv.style.left = '50%';
    labelDiv.style.transform = 'translate(-50%, -50%)';
    labelDiv.style.textAlign = 'center';
    labelDiv.style.pointerEvents = 'none';
    labelDiv.innerHTML = `
        <div id="meter-score" style="font-size: 1.75rem; font-weight: 800; font-family: monospace; color: #fff;">--%</div>
        <div id="meter-status" style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; color: #8fa0b5;">Ready</div>
    `;
    meterContainer.style.position = 'relative';
    meterContainer.appendChild(labelDiv);

    window.__meter = {
        setConfidence: function(confidence, verdict = 'bonafide') {
            const scoreEl = document.getElementById('meter-score');
            const statusEl = document.getElementById('meter-status');

            // Authenticity percentage:
            // If bonafide: authenticity = confidence (e.g. 0.95 -> 95%)
            // If spoof: authenticity = 1.0 - confidence (e.g. 0.90 spoof -> 10% authenticity)
            let authenticity = verdict === 'bonafide' ? confidence : (1.0 - confidence);
            authenticity = Math.max(0.02, Math.min(0.98, authenticity));

            const pctInt = Math.round(authenticity * 100);
            const offset = circumference * (1 - authenticity);
            progress.style.strokeDashoffset = `${offset}`;

            if (scoreEl) scoreEl.textContent = `${pctInt}%`;

            if (verdict === 'bonafide') {
                progress.setAttribute('stroke', 'url(#meterGreen)');
                if (statusEl) {
                    statusEl.textContent = 'Authentic';
                    statusEl.style.color = '#00ff99';
                }
                if (scoreEl) scoreEl.style.color = '#00ff99';
                if (meterText) meterText.textContent = `Authenticity: ${pctInt}% (Genuine Human)`;
            } else {
                progress.setAttribute('stroke', 'url(#meterRed)');
                if (statusEl) {
                    statusEl.textContent = 'Spoof / AI';
                    statusEl.style.color = '#ff3366';
                }
                if (scoreEl) scoreEl.style.color = '#ff3366';
                if (meterText) meterText.textContent = `Authenticity: ${pctInt}% (AI Spoof Detected)`;
            }
        },

        reset: function() {
            progress.style.strokeDashoffset = `${circumference * 0.5}`;
            progress.setAttribute('stroke', '#4a5568');
            const scoreEl = document.getElementById('meter-score');
            const statusEl = document.getElementById('meter-status');
            if (scoreEl) {
                scoreEl.textContent = '--%';
                scoreEl.style.color = '#fff';
            }
            if (statusEl) {
                statusEl.textContent = 'Idle';
                statusEl.style.color = '#8fa0b5';
            }
            if (meterText) meterText.textContent = 'Authenticity Meter';
        }
    };
}