#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
source ../.env

echo "=== Purge & Load ==="
python3 scripts/load_sessions.py sessions/ --purge --api-key "$SHODH_API_KEY"

echo ""
echo "=== Exporting GEXF ==="
curl --max-time 120 -s "http://localhost:3033/api/graph/autonomites-pipeline/export?format=gexf" \
  -H "X-API-Key: $SHODH_API_KEY" -o graph.gexf
echo "graph.gexf: $(wc -c < graph.gexf | tr -d ' ') bytes"
