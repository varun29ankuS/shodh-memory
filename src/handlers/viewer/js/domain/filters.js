export const matchesFilters = {
  node(attrs, f) {
    if (!f.activeTypes.has(attrs.type)) return false;
    if (attrs.type === 'memory') {
      if (!f.activeTiers.has(attrs.tier)) return false;
    }
    if ((attrs.activation || 0) < (f.minActivation || 0)) return false;
    if (f.recencyWindowMs != null && attrs.last_accessed) {
      const age = Date.now() - Date.parse(attrs.last_accessed);
      if (age > f.recencyWindowMs) return false;
    }
    return true;
  },

  edge(attrs, f) {
    if (!f.activeTiers.has(attrs.tier)) return false;
    if ((attrs.weight || 0) < (f.minWeight || 0)) return false;
    if (!f.activeLtp.has(attrs.ltp_status || 'None')) return false;
    return true;
  },
};
