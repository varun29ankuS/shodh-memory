export function exportIdsAsText(ids) {
  return ids.map(i => String(i)).join('\n') + '\n';
}

export function exportVisibleAsGexf(graph, SubgraphClass, gexfWrite, isNodeVisible) {
  const sub = new SubgraphClass();
  for (const id of graph.nodes()) {
    if (isNodeVisible(id)) sub.addNode(id, graph.getNodeAttributes(id));
  }
  for (const id of graph.edges()) {
    const s = graph.source(id), t = graph.target(id);
    if (isNodeVisible(s) && isNodeVisible(t)) {
      sub.addEdge(s, t, graph.getEdgeAttributes(id));
    }
  }
  return gexfWrite(sub);
}

export function download(text, filename, mimeType) {
  const blob = new Blob([text], { type: mimeType });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
