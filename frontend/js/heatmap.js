// heatmap.js – Render the Grad-CAM spectrogram heatmap returned by the server
// -----------------------------------------------------------------------------

export function initHeatmap() {
    const img = document.getElementById('heatmap');
    const caption = document.getElementById('heatmap-caption');
    if (!img) return;

    window.__heatmap = {
        update: function(b64Data, verdict = 'spoof') {
            if (!b64Data) {
                img.style.opacity = '0.3';
                return;
            }

            // Ensure valid data URI prefix
            const src = b64Data.startsWith('data:')
                ? b64Data
                : `data:image/png;base64,${b64Data}`;

            img.src = src;
            img.style.opacity = '1';
            img.style.display = 'block';

            if (caption) {
                if (verdict === 'spoof') {
                    caption.textContent = 'Grad-CAM Spoof Heatmap (Red/Yellow = High Artifact Density)';
                    caption.style.color = '#ff4466';
                } else {
                    caption.textContent = 'Spectrogram Acoustic Representation (Natural Harmonics)';
                    caption.style.color = '#00ff99';
                }
            }
        },

        reset: function() {
            img.style.opacity = '0.3';
            if (caption) {
                caption.textContent = 'Grad-CAM Spoof Heatmap';
                caption.style.color = '#888';
            }
        },
    };
}