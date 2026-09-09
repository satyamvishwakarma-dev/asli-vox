// app.js – Main application controller wiring UI, Audio, WebSocket & REST
// -----------------------------------------------------------------------------

import { sendRecord, sendChunk, sendFinish, sendFile, fetchHistory } from './api.js';

let isRecording = false;
let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];

export function initApp() {
    const recordBtn = document.getElementById('recordBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const verdictSection = document.getElementById('verdict');
    const verdictLabel = document.getElementById('verdict-label');
    const verdictConfidence = document.getElementById('verdict-confidence');
    const historyList = document.getElementById('history-list');

    // Create hidden file input for audio uploads
    let fileInput = document.getElementById('fileInput');
    if (!fileInput) {
        fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.id = 'fileInput';
        fileInput.accept = 'audio/*,.wav,.mp3,.ogg,.flac';
        fileInput.style.display = 'none';
        document.body.appendChild(fileInput);
    }

    // Enable record button once app is initialized
    if (recordBtn) {
        recordBtn.disabled = false;
        recordBtn.addEventListener('click', toggleRecording);
    }

    // Upload button triggers hidden file input
    if (uploadBtn) {
        uploadBtn.addEventListener('click', () => {
            fileInput.value = '';
            fileInput.click();
        });
    }

    // File input change listener
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        showStatus(`Uploading and analyzing ${file.name}...`);
        if (uploadBtn) uploadBtn.disabled = true;

        try {
            await sendFile(file);
        } catch (err) {
            console.error('Upload detection failed:', err);
            alert(`Analysis failed: ${err.message || err}`);
        } finally {
            if (uploadBtn) uploadBtn.disabled = false;
        }
    });

    // Handle WebSocket connection state
    document.addEventListener('asliVox:wsReady', (e) => {
        const status = e.detail?.status;
        console.log(`WebSocket status changed: ${status}`);
    });

    // Listen for detection results from WebSocket or REST
    document.addEventListener('asliVox:result', (e) => {
        const result = e.detail;
        if (!result) return;
        renderResult(result);
    });

    // Initial session history fetch
    loadHistory();

    // -------------------------------------------------------------------------
    // Microphone Recording Logic
    // -------------------------------------------------------------------------
    async function toggleRecording() {
        if (!isRecording) {
            await startRecording();
        } else {
            await stopRecording();
        }
    }

    async function startRecording() {
        try {
            recordedChunks = [];
            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false,
                }
            });

            // Start waveform visualization
            if (window.__waveform && typeof window.__waveform.start === 'function') {
                window.__waveform.start(mediaStream);
            }

            // Prefer audio/webm or audio/wav supported by browser
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus') ? 'audio/ogg;codecs=opus' : '');

            const options = mimeType ? { mimeType } : {};
            mediaRecorder = new MediaRecorder(mediaStream, options);

            mediaRecorder.ondataavailable = (ev) => {
                if (ev.data && ev.data.size > 0) {
                    recordedChunks.push(ev.data);
                    // Stream chunk to WebSocket
                    ev.data.arrayBuffer().then(buffer => {
                        sendChunk(buffer);
                    });
                }
            };

            mediaRecorder.onstop = () => {
                // When recording stops, package all chunks and send finish marker
                if (recordedChunks.length > 0) {
                    const fullBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/wav' });
                    sendRecord(fullBlob);
                } else {
                    sendFinish();
                }
            };

            // Request slices every 400ms for live streaming detection
            mediaRecorder.start(400);
            isRecording = true;

            // Update button UI
            recordBtn.textContent = '⏹ Stop Recording';
            recordBtn.classList.add('recording');
            showStatus('Listening to microphone in real-time...');

        } catch (err) {
            console.error('Microphone access denied or error:', err);
            alert(`Microphone error: ${err.message || 'Permission denied. Please grant microphone access.'}`);
            isRecording = false;
        }
    }

    async function stopRecording() {
        isRecording = false;

        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }

        if (window.__waveform && typeof window.__waveform.stop === 'function') {
            window.__waveform.stop();
        }

        if (recordBtn) {
            recordBtn.textContent = '🎙️ Record (Mic)';
            recordBtn.classList.remove('recording');
        }

        showStatus('Processing audio capture...');
    }

    // -------------------------------------------------------------------------
    // UI Rendering
    // -------------------------------------------------------------------------
    function renderResult(result) {
        const { verdict, confidence, heatmap_base64, timestamp, audio_duration_sec, details } = result;

        // 1. Update Authenticity Meter
        if (window.__meter && typeof window.__meter.setConfidence === 'function') {
            window.__meter.setConfidence(confidence, verdict);
        }

        // 2. Update Grad-CAM Heatmap
        if (window.__heatmap && typeof window.__heatmap.update === 'function') {
            window.__heatmap.update(heatmap_base64, verdict);
        }

        // 3. Update Verdict Readout
        if (verdictSection) {
            verdictSection.style.display = 'block';
            verdictSection.className = `verdict-banner ${verdict}`;

            if (verdictLabel) {
                verdictLabel.textContent = verdict === 'bonafide'
                    ? '✅ GENUINE VOICE (BONAFIDE)'
                    : '🚨 AI DEEPFAKE DETECTED (SPOOF)';
                verdictLabel.className = `verdict-title ${verdict}`;
            }

            if (verdictConfidence) {
                const pct = (confidence * 100).toFixed(1);
                const durStr = audio_duration_sec ? ` • Duration: ${audio_duration_sec}s` : '';
                const modelStr = details?.model_source ? ` • Engine: ${details.model_source}` : '';
                verdictConfidence.textContent = `Confidence: ${pct}%${durStr}${modelStr}`;
            }
        }

        // 4. Prepend to Session History Strip
        addHistoryItemToDOM({
            id: details?.id || Math.random().toString(36).substring(7),
            verdict,
            confidence,
            timestamp: timestamp || new Date().toISOString(),
            source: isRecording ? 'live' : 'upload',
            audio_duration_sec,
        });
    }

    function addHistoryItemToDOM(item) {
        if (!historyList) return;

        // Remove placeholder empty state if present
        const emptyMsg = historyList.querySelector('.empty-history');
        if (emptyMsg) emptyMsg.remove();

        const row = document.createElement('div');
        row.className = `history-item ${item.verdict}`;

        const timeStr = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : 'Now';
        const isBonafide = item.verdict === 'bonafide';
        const pct = Math.round(item.confidence * 100);
        const badgeColor = isBonafide ? '#00ff99' : '#ff3366';
        const icon = isBonafide ? '🟢' : '🔴';
        const sourceIcon = item.source === 'live' ? '🎙️' : '📁';

        row.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center; padding: 0.4rem 0.6rem;">
                <span>${icon} <strong>${item.verdict.toUpperCase()}</strong> (${pct}%)</span>
                <span style="font-size:0.75rem; color:#8fa0b5;">${sourceIcon} ${timeStr}</span>
            </div>
        `;

        historyList.prepend(row);

        // Keep maximum 10 items in DOM
        while (historyList.children.length > 10) {
            historyList.removeChild(historyList.lastChild);
        }
    }

    async function loadHistory() {
        if (!historyList) return;
        const history = await fetchHistory();
        if (history && history.length > 0) {
            historyList.innerHTML = '';
            history.forEach(item => addHistoryItemToDOM(item));
        } else {
            historyList.innerHTML = '<div class="empty-history" style="padding:0.5rem; color:#64748b;">No recent scans yet. Record speech or upload a .wav file.</div>';
        }
    }

    function showStatus(text) {
        const statusEl = document.getElementById('meter-status');
        if (statusEl) {
            statusEl.textContent = text;
        }
    }
}
