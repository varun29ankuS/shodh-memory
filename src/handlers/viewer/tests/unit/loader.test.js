import { describe, it, expect, mock } from "bun:test";
import { createLoader } from '../../js/graph/loader.js';

const SAMPLE_GEXF = `<?xml version="1.0" encoding="UTF-8"?>
<gexf xmlns="http://gexf.net/1.3" version="1.3">
  <graph defaultedgetype="directed">
    <nodes><node id="n1" label="A"/></nodes>
    <edges/>
  </graph>
</gexf>`;

describe('loader', () => {
  it('parses GEXF text into a graphology Graph', async () => {
    const loader = createLoader({ apiClient: null, gexfParser: (Graph, xml) => {
      const g = new Graph();
      g.addNode('n1', { label: 'A' });
      return g;
    }, GraphClass: class { constructor(){ this.nodes={}; this.addNode = (i,a)=>this.nodes[i]=a; } } });
    const g = await loader.parseFromText(SAMPLE_GEXF);
    expect(Object.keys(g.nodes)).toContain('n1');
  });

  it('fetches GEXF from API and returns {graph, etag, meta}', async () => {
    const apiClient = {
      fetchGexf: mock().mockResolvedValue(new Response(SAMPLE_GEXF, {
        status: 200,
        headers: { etag: 'W/"xyz"', 'content-type': 'application/gexf+xml' },
      })),
    };
    const loader = createLoader({ apiClient,
      gexfParser: () => ({ nodes: { n1: { label: 'A' } } }),
      GraphClass: class {} });
    const result = await loader.fetchFromApi('user1');
    expect(apiClient.fetchGexf).toHaveBeenCalledWith('user1', null);
    expect(result.etag).toBe('W/"xyz"');
    expect(result.graph).toBeDefined();
  });

  it('returns unchanged:true on 304', async () => {
    const apiClient = {
      fetchGexf: mock().mockResolvedValue(new Response(null, { status: 304 })),
    };
    const loader = createLoader({ apiClient, gexfParser: null, GraphClass: null });
    const result = await loader.fetchFromApi('user1', 'W/"abc"');
    expect(result.unchanged).toBe(true);
  });
});
