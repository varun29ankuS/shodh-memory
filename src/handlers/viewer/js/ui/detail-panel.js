export function createDetailPanel({ container, apiClient, userId }) {
  async function show(graph, id) {
    const attrs = graph.getNodeAttributes(id);
    container.innerHTML = `
      <h3>${escape(attrs.label || id)}</h3>
      <div class="meta">${escape(attrs.type)} · ${escape(attrs.tier || '')}</div>
      <dl>
        ${kv('importance', attrs.importance)}
        ${kv('activation', attrs.activation)}
        ${kv('access_count', attrs.access_count)}
        ${kv('last_accessed', attrs.last_accessed)}
      </dl>
      <div class="content">${escape(attrs.content || '')}</div>
    `;

    if (!attrs.content && attrs.type === 'memory') {
      try {
        const resp = await apiClient.fetchMemoryContent(id, userId);
        if (resp.ok) {
          const body = await resp.json();
          const full = body.content || body.experience?.content;
          if (full) {
            container.querySelector('.content').textContent = full;
            graph.mergeNodeAttributes(id, { content: full });
          }
        } else {
          container.querySelector('.content').textContent = 'Full content unavailable.';
        }
      } catch (e) {
        container.querySelector('.content').textContent = 'Full content unavailable.';
      }
    }
  }

  function hide() { container.innerHTML = ''; }
  return { show, hide };
}

function kv(label, value) {
  if (value == null) return '';
  return `<dt>${escape(label)}</dt><dd>${escape(String(value))}</dd>`;
}
function escape(s) {
  return String(s || '').replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}
