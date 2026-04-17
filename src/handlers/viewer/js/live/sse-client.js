export function createSseClient({ url, onMessage, onStatusChange = () => {} }) {
  let es = null;
  let attempts = 0;
  let reconnectTimer = null;
  let stopped = false;

  function connect() {
    if (stopped) return;
    onStatusChange('connecting');
    es = new EventSource(url);
    es.addEventListener('open', () => {
      attempts = 0;
      onStatusChange('connected');
    });
    es.addEventListener('message', (evt) => {
      try {
        onMessage(JSON.parse(evt.data));
      } catch (e) {
        onMessage({ raw: evt.data });
      }
    });
    es.addEventListener('error', () => {
      onStatusChange('disconnected');
      es.close();
      scheduleReconnect();
    });
  }

  function scheduleReconnect() {
    if (stopped) return;
    const delay = Math.min(30_000, 1000 * Math.pow(2, attempts));
    attempts++;
    reconnectTimer = setTimeout(connect, delay);
  }

  function close() {
    stopped = true;
    if (es) es.close();
    if (reconnectTimer) clearTimeout(reconnectTimer);
    onStatusChange('closed');
  }

  return { connect, close };
}
