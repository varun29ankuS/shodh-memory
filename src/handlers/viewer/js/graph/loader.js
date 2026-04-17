export function createLoader({ apiClient, gexfParser, GraphClass }) {
  async function parseFromText(text) {
    return gexfParser(GraphClass, text);
  }

  async function fetchFromApi(userId, prevEtag = null) {
    const resp = await apiClient.fetchGexf(userId, prevEtag);
    if (resp.status === 304) return { unchanged: true };
    if (!resp.ok) throw new Error(`fetch failed: ${resp.status}`);
    const etag = resp.headers.get('etag');
    const text = await resp.text();
    const graph = await parseFromText(text);
    const meta = extractMeta(text);
    return { graph, etag, meta };
  }

  function extractMeta(xml) {
    const m = xml.match(/<server_time>([^<]+)<\/server_time>/);
    return { server_time: m ? m[1] : null };
  }

  return { parseFromText, fetchFromApi };
}
