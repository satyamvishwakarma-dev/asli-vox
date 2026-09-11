(function () {
    const canvas = document.getElementById('bgField');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let width, height, dpr;
    let t = 0;

    // Global intensity — app.js can nudge this via window.__bgIntensity
    // (0 = idle ambient drift, 1 = active listening, driven higher briefly on spoof alerts)
    window.__bgIntensity = window.__bgIntensity || 0;
    let displayedIntensity = 0;

    function resize() {
        dpr = Math.min(window.devicePixelRatio || 1, 2);
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    window.addEventListener('resize', resize);
    resize();

    // Ribbon field: layered sine ribbons drifting slowly, like an idle spectrogram
    const RIBBONS = 5;
    const ribbonConfigs = Array.from({ length: RIBBONS }, (_, i) => ({
        offsetY: 0.18 + (i / (RIBBONS - 1)) * 0.64,
        amp: 26 + i * 6,
        freq: 0.0016 + i * 0.0003,
        speed: 0.0006 + i * 0.00018,
        phase: i * 1.7,
        hueMix: i % 2 === 0 ? 'cyan' : 'violet',
        width: 1.1 + (i % 3) * 0.4,
    }));

    // Particle field: faint drifting motes, like acoustic energy
    const PARTICLE_COUNT = 46;
    const particles = Array.from({ length: PARTICLE_COUNT }, () => ({
        x: Math.random(),
        y: Math.random(),
        r: 0.6 + Math.random() * 1.6,
        vx: (Math.random() - 0.5) * 0.00012,
        vy: (Math.random() - 0.5) * 0.00012,
        phase: Math.random() * Math.PI * 2,
        speed: 0.4 + Math.random() * 0.8,
    }));

    function colorFor(name, alpha) {
        return name === 'cyan'
            ? `rgba(10, 234, 255, ${alpha})`
            : `rgba(139, 107, 255, ${alpha})`;
    }

    function draw() {
        t += 1;

        // ease displayed intensity toward target for smooth transitions
        const target = Math.max(0, Math.min(1, window.__bgIntensity || 0));
        displayedIntensity += (target - displayedIntensity) * 0.02;

        ctx.clearRect(0, 0, width, height);

        // Ribbons
        ribbonConfigs.forEach((r) => {
            const baseY = height * r.offsetY;
            const ampBoost = r.amp * (1 + displayedIntensity * 0.9);
            const alpha = 0.05 + displayedIntensity * 0.06;

            ctx.beginPath();
            ctx.lineWidth = r.width;
            ctx.strokeStyle = colorFor(r.hueMix, alpha + 0.03);
            ctx.shadowBlur = 14 + displayedIntensity * 10;
            ctx.shadowColor = colorFor(r.hueMix, 0.5);

            for (let x = 0; x <= width; x += 8) {
                const y =
                    baseY +
                    Math.sin(x * r.freq + t * r.speed + r.phase) * ampBoost +
                    Math.sin(x * r.freq * 2.3 - t * r.speed * 1.4 + r.phase) * (ampBoost * 0.28);
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        });
        ctx.shadowBlur = 0;

        // Particles
        particles.forEach((p) => {
            p.x += p.vx * p.speed * (1 + displayedIntensity);
            p.y += p.vy * p.speed * (1 + displayedIntensity);
            if (p.x < -0.02) p.x = 1.02;
            if (p.x > 1.02) p.x = -0.02;
            if (p.y < -0.02) p.y = 1.02;
            if (p.y > 1.02) p.y = -0.02;

            const twinkle = 0.4 + 0.6 * Math.abs(Math.sin(t * 0.01 * p.speed + p.phase));
            const alpha = (0.15 + displayedIntensity * 0.35) * twinkle;

            ctx.beginPath();
            ctx.fillStyle = `rgba(200, 220, 255, ${alpha})`;
            ctx.arc(p.x * width, p.y * height, p.r + displayedIntensity * 0.8, 0, Math.PI * 2);
            ctx.fill();
        });

        requestAnimationFrame(draw);
    }

    // Respect reduced-motion preference: render one calm static frame only
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
        draw();
    } else {
        requestAnimationFrame(draw);
    }

    // Let app.js hook in without any edits: observe status text / verdict changes
    // and modulate ambient intensity automatically.
    const statusEl = document.getElementById('systemStatusText');
    const verdictEl = document.getElementById('verdictBanner');

    if (statusEl) {
        const obs = new MutationObserver(() => {
            const listening = statusEl.textContent.trim().toUpperCase().includes('ONLINE');
            window.__bgIntensity = listening ? 0.55 : 0;
        });
        obs.observe(statusEl, { childList: true, characterData: true, subtree: true });
    }

    if (verdictEl) {
        const obs2 = new MutationObserver(() => {
            if (verdictEl.classList.contains('spoof')) {
                window.__bgIntensity = 1;
            } else if (verdictEl.classList.contains('bonafide')) {
                window.__bgIntensity = 0.55;
            }
        });
        obs2.observe(verdictEl, { attributes: true, attributeFilter: ['class'] });
    }
})();
