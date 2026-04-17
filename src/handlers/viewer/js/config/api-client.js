export function createApiClient({ baseUrl, apiKey }) {
  baseUrl = baseUrl.replace(/\/$/, '');

  async function fetchGexf(userId, etag = null) {
    const headers = { 'X-API-Key': apiKey };
    if (etag) headers['If-None-Match'] = etag;
    const url = `${baseUrl}/api/graph/${encodeURIComponent(userId)}/export?format=gexf`;
    return fetch(url, { headers });
  }

  function sseUrl(userId) {
    const u = `${baseUrl}/api/events/sse?user_id=${encodeURIComponent(userId)}&api_key=${encodeURIComponent(apiKey)}`;
    return u;
  }

  async function fetchMemoryContent(memoryId) {
    const url = `${baseUrl}/api/memories/${encodeURIComponent(memoryId)}`;
    return fetch(url, { headers: { 'X-API-Key': apiKey } });
  }

  return { fetchGexf, sseUrl, fetchMemoryContent };
}
