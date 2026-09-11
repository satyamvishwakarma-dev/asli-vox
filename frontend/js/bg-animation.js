// bg-animation.js – Animated Background: Waveforms, Particles & Grid
// -----------------------------------------------------------------------------

export function initBgAnimation() {
    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let animId = null;
    let isVisible = true;

    // Resize canvas to fill viewport
    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    // Pause when tab hidden
    document.addEventListener('visibilitychange', () => {
        isVisible = !document.hidden;
        if (isVisible && !animId) animate();
    });

    // -------------------------------------------------------------------------
    // WAVEFORM LAYER – Slow sine waves
    // -------------------------------------------------------------------------
    const waves = [
        { amplitude: 30, frequency: 0.008, speed: 0.015, offset: 0, color: 'rgba(0, 229, 255, 0.06)' },
        { amplitude: 20, frequency: 0.012, speed: 0.02, offset: 0, color: 'rgba(0, 229, 255, 0.04)' },
        { amplitude: 25, frequency: 0.006, speed: 0.01, offset: 0, color: 'rgba(0, 255, 153, 0.035)' },
        { amplitude: 15, frequency: 0.015, speed: 0.025, offset: 0, color: 'rgba(0, 229, 255, 0.03)' },
    ];

    function drawWaves(time) {
        const w = canvas.width;
        const h = canvas.height;

        waves.forEach((wave, idx) => {
            wave.offset += wave.speed;

            ctx.beginPath();
            ctx.strokeStyle = wave.color;
            ctx.lineWidth = 1.5;

            const baseY = h * (0.25 + idx * 0.18);

            for (let x = 0; x < w; x += 3) {
                const y = baseY + Math.sin(x * wave.frequency + wave.offset) * wave.amplitude;
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        });
    }

    // -------------------------------------------------------------------------
    // PARTICLE LAYER – Floating dots
    // -------------------------------------------------------------------------
    const particles = [];
    const PARTICLE_COUNT = 60;

    function createParticle() {
        return {
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            size: Math.random() * 2 + 0.5,
            speedX: (Math.random() - 0.5) * 0.4,
            speedY: -(Math.random() * 0.3 + 0.15),
            opacity: Math.random() * 0.4 + 0.1,
            color: Math.random() > 0.6
                ? `rgba(0, 255, 153, VAL)`
                : `rgba(0, 229, 255, VAL)`,
            pulse: Math.random() * Math.PI * 2,
            pulseSpeed: Math.random() * 0.02 + 0.01,
        };
    }

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push(createParticle());
    }

    function drawParticles() {
        const w = canvas.width;
        const h = canvas.height;

        particles.forEach(p => {
            p.x += p.speedX;
            p.y += p.speedY;
            p.pulse += p.pulseSpeed;

            // Wrap around
            if (p.y < -10) { p.y = h + 10; p.x = Math.random() * w; }
            if (p.x < -10) p.x = w + 10;
            if (p.x > w + 10) p.x = -10;

            const alpha = p.opacity * (0.6 + 0.4 * Math.sin(p.pulse));
            const fillColor = p.color.replace('VAL', alpha.toFixed(3));

            ctx.beginPath();
            ctx.fillStyle = fillColor;
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    // -------------------------------------------------------------------------
    // GRID LAYER – Faint cyber grid
    // -------------------------------------------------------------------------
    const GRID_SPACING = 80;
    let gridPulse = 0;

    function drawGrid() {
        const w = canvas.width;
        const h = canvas.height;

        gridPulse += 0.008;
        const alpha = 0.03 + 0.02 * Math.sin(gridPulse);

        ctx.strokeStyle = `rgba(0, 229, 255, ${alpha.toFixed(3)})`;
        ctx.lineWidth = 0.5;

        // Vertical lines
        for (let x = 0; x < w; x += GRID_SPACING) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }

        // Horizontal lines
        for (let y = 0; y < h; y += GRID_SPACING) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // Grid intersection dots
        const dotAlpha = alpha * 3;
        ctx.fillStyle = `rgba(0, 229, 255, ${dotAlpha.toFixed(3)})`;
        for (let x = 0; x < w; x += GRID_SPACING) {
            for (let y = 0; y < h; y += GRID_SPACING) {
                ctx.beginPath();
                ctx.arc(x, y, 1, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }

    // -------------------------------------------------------------------------
    // HORIZONTAL SCAN LINE – Moving detection line
    // -------------------------------------------------------------------------
    let scanY = 0;
    const scanSpeed = 0.6;

    function drawScanLine() {
        const h = canvas.height;
        scanY += scanSpeed;
        if (scanY > h) scanY = 0;

        const gradient = ctx.createLinearGradient(0, scanY - 30, 0, scanY + 30);
        gradient.addColorStop(0, 'rgba(0, 229, 255, 0)');
        gradient.addColorStop(0.5, 'rgba(0, 229, 255, 0.06)');
        gradient.addColorStop(1, 'rgba(0, 229, 255, 0)');

        ctx.fillStyle = gradient;
        ctx.fillRect(0, scanY - 30, canvas.width, 60);
    }

    // -------------------------------------------------------------------------
    // MAIN ANIMATION LOOP
    // -------------------------------------------------------------------------
    function animate() {
        if (!isVisible) {
            animId = null;
            return;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        drawGrid();
        drawWaves(Date.now());
        drawParticles();
        drawScanLine();

        animId = requestAnimationFrame(animate);
    }

    animate();
}
