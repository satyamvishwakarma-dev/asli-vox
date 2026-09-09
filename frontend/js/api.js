// api.js – WebSocket live streaming client & REST file upload helper
// -----------------------------------------------------------------------------

const WS_PROTOCOL = location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL = `${WS_PROTOCOL}//${location.host}/ws/detect`;
const REST_DETECT_URL = `${location.origin}/detect`;
const REST_HISTORY_URL = `${location.origin}/history`;

let ws = null;
let reconnectTimer = null;
const resultCallbacks = [];

export function onResult(callback) {
    if (typeof callback === 'function') {
        resultCallbacks.push(callback);
    }
}

function notifyResult(data) {
    window.__lastResult = data;
    // Dispatch standard CustomEvent
    const event = new CustomEvent('asliVox:result', { detail: data });
    document.dispatchEvent(event);

    // Call registered callbacks
    resultCallbacks.forEach(cb => {
        try { cb(data); } catch (e) { console.error('Callback error:', e); }
    });
}

export function initAPI() {
    connectWebSocket();
}

function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return;
    }

    try {
        ws = new WebSocket(WS_URL);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
            console.log('✅ WebSocket connected to AsliVox engine:', WS_URL);
            if (reconnectTimer) {
                clearTimeout(reconnectTimer);
                reconnectTimer = null;
            }
            document.dispatchEvent(new CustomEvent('asliVox:wsReady', { detail: { status: 'open' } }));
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                notifyResult(data);
            } catch (err) {
                console.error('Failed to parse WebSocket JSON payload:', err, event.data);
            }
        };

        ws.onerror = (err) => {
            console.warn('WebSocket connection error:', err);
        };

        ws.onclose = () => {
            console.log('WebSocket closed. Scheduling reconnect in 2s...');
            document.dispatchEvent(new CustomEvent('asliVox:wsReady', { detail: { status: 'closed' } }));
            if (!reconnectTimer) {
                reconnectTimer = setTimeout(connectWebSocket, 2000);
            }
        };
    } catch (e) {
        console.error('WebSocket initialization error:', e);
    }
}

// Send binary audio chunk (e.g. PCM samples or audio blob slice)
export function sendChunk(arrayBufferOrBlob) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket();
        return;
    }
    ws.send(arrayBufferOrBlob);
}

// Send finish control message when recording ends
export function sendFinish() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'finish' }));
    }
}

// Send completed recording blob (backward compatible with original API)
export function sendRecord(blob) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket();
    }

    if (blob instanceof Blob) {
        blob.arrayBuffer().then(buf => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(buf);
                sendFinish();
            }
        });
    } else {
        sendChunk(blob);
        sendFinish();
    }
}

// REST single-shot file upload
export async function sendFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(REST_DETECT_URL, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Upload detection failed: ${response.status} ${errorText}`);
    }

    const result = await response.json();
    notifyResult(result);
    return result;
}

// Fetch session history from REST endpoint
export async function fetchHistory() {
    try {
        const response = await fetch(REST_HISTORY_URL);
        if (response.ok) {
            return await response.json();
        }
    } catch (e) {
        console.warn('Failed to fetch session history:', e);
    }
    return [];
}

// Global hook for easy debugging in browser console
window.__api = {
    initAPI,
    sendRecord,
    sendChunk,
    sendFinish,
    sendFile,
    fetchHistory,
    onResult,
};