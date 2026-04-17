import { describe, it, expect, beforeEach, afterEach, spyOn } from "bun:test";
import { createApiClient } from '../../js/config/api-client.js';

describe('createApiClient', () => {
  let fetchSpy;
  beforeEach(() => { fetchSpy = spyOn(globalThis, 'fetch'); });
  afterEach(() => { fetchSpy.mockRestore(); });

  it('attaches X-API-Key header to fetch calls', async () => {
    fetchSpy.mockResolvedValue(new Response('ok', { status: 200 }));
    const api = createApiClient({ baseUrl: 'http://localhost:3000', apiKey: 'abc' });
    await api.fetchGexf('user1');
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe('http://localhost:3000/api/graph/user1/export?format=gexf');
    expect(init.headers['X-API-Key']).toBe('abc');
  });

  it('forwards If-None-Match when etag is provided', async () => {
    fetchSpy.mockResolvedValue(new Response('', { status: 304 }));
    const api = createApiClient({ baseUrl: '', apiKey: 'k' });
    await api.fetchGexf('u', 'W/"abc123"');
    expect(fetchSpy.mock.calls[0][1].headers['If-None-Match']).toBe('W/"abc123"');
  });

  it('builds SSE URL with api_key query param', () => {
    const api = createApiClient({ baseUrl: 'http://localhost:3000', apiKey: 'sse-k' });
    const url = api.sseUrl('u');
    expect(url).toBe('http://localhost:3000/api/events/sse?user_id=u&api_key=sse-k');
  });

  it('fetchMemoryContent includes user_id query param', async () => {
    fetchSpy.mockResolvedValue(new Response('{}', { status: 200 }));
    const api = createApiClient({ baseUrl: 'http://localhost:3000', apiKey: 'k' });
    await api.fetchMemoryContent('mem-1', 'user-42');
    expect(fetchSpy.mock.calls[0][0]).toBe(
      'http://localhost:3000/api/memories/mem-1?user_id=user-42'
    );
    expect(fetchSpy.mock.calls[0][1].headers['X-API-Key']).toBe('k');
  });
});
