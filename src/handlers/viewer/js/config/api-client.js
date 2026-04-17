export function createApiClient({ baseUrl, apiKey }) {
  baseUrl = baseUrl.replace(/\/$/, '');

  function authHeaders(extra = null) {
    const h = { 'X-API-Key': apiKey };
    if (extra) Object.assign(h, extra);
    return h;
  }

  async function fetchGexf(userId, etag = null) {
    const url = `${baseUrl}/api/graph/${encodeURIComponent(userId)}/export?format=gexf`;
    return fetch(url, { headers: authHeaders(etag ? { 'If-None-Match': etag } : null) });
  }

  function sseUrl(userId) {
    return `${baseUrl}/api/events/sse?user_id=${encodeURIComponent(userId)}&api_key=${encodeURIComponent(apiKey)}`;
  }

  async function fetchMemoryContent(memoryId) {
    const url = `${baseUrl}/api/memories/${encodeURIComponent(memoryId)}`;
    return fetch(url, { headers: authHeaders() });
  }

  return { fetchGexf, sseUrl, fetchMemoryContent };
}
