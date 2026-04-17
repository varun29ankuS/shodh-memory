import { describe, it, expect, jest, mock, beforeEach, afterEach } from "bun:test";
import { createSseClient } from '../../js/live/sse-client.js';

class MockEventSource {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.listeners = {};
    MockEventSource.instances.push(this);
    setTimeout(() => { this.readyState = 1; this._fire('open', {}); }, 0);
  }
  addEventListener(evt, fn) { (this.listeners[evt] ||= []).push(fn); }
  close() { this.readyState = 2; }
  _fire(evt, data) { (this.listeners[evt] || []).forEach(f => f(data)); }
}

describe('sseClient', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    MockEventSource.instances = [];
    globalThis.EventSource = MockEventSource;
  });
  afterEach(() => jest.useRealTimers());

  it('connects to the given URL and calls onMessage for each event', () => {
    const onMessage = mock();
    const client = createSseClient({
      url: 'http://x/sse?user_id=u&api_key=k',
      onMessage,
    });
    client.connect();
    jest.runOnlyPendingTimers();
    const es = MockEventSource.instances[0];
    es._fire('message', { data: '{"event_type":"TODO_CREATED"}' });
    expect(onMessage).toHaveBeenCalledTimes(1);
  });

  it('reconnects with exponential backoff on error', () => {
    const client = createSseClient({ url: 'http://x/sse', onMessage: mock() });
    client.connect();
    jest.runOnlyPendingTimers();
    MockEventSource.instances[0]._fire('error', {});
    expect(MockEventSource.instances.length).toBe(1);
    jest.advanceTimersByTime(1000);
    expect(MockEventSource.instances.length).toBe(2);
    MockEventSource.instances[1]._fire('error', {});
    jest.advanceTimersByTime(2000);
    expect(MockEventSource.instances.length).toBe(3);
  });

  it('caps backoff at 30s', () => {
    const client = createSseClient({ url: 'http://x/sse', onMessage: mock() });
    client.connect();
    jest.runOnlyPendingTimers();
    for (let i = 0; i < 6; i++) {
      const cur = MockEventSource.instances[MockEventSource.instances.length - 1];
      cur._fire('error', {});
      jest.advanceTimersByTime(30_000);
    }
    expect(MockEventSource.instances.length).toBeGreaterThanOrEqual(6);
  });
});
