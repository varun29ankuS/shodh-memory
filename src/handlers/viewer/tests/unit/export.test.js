import { describe, it, expect, mock } from "bun:test";
import { exportIdsAsText, exportVisibleAsGexf } from '../../js/domain/export.js';

describe('exportIdsAsText', () => {
  it('joins IDs with newlines', () => {
    const result = exportIdsAsText(['a', 'b', 'c']);
    expect(result).toBe('a\nb\nc\n');
  });
});

describe('exportVisibleAsGexf', () => {
  it('delegates to gexfWrite with a filtered subgraph', () => {
    const gexfWrite = mock(() => '<gexf/>');
    const SubgraphClass = class {
      constructor() { this.nodes = []; this.edges = []; }
      addNode(id, a) { this.nodes.push([id, a]); }
      addEdge(s, t, a) { this.edges.push([s, t, a]); }
    };
    const graph = {
      nodes: () => ['a', 'b', 'hidden'],
      edges: () => ['ab'],
      getNodeAttributes: (i) => ({ id: i }),
      getEdgeAttributes: () => ({}),
      source: () => 'a', target: () => 'b',
    };
    const isVisible = (id) => id !== 'hidden';
    const result = exportVisibleAsGexf(graph, SubgraphClass, gexfWrite, isVisible);
    expect(result).toBe('<gexf/>');
    expect(gexfWrite).toHaveBeenCalledTimes(1);
  });
});
