set -u
L1="SHODH_GRAPH_EXTRACTED_PREDICATES=1 SHODH_GRAPH_PREDICATE_WEIGHTS=1 SHODH_GRAPH_IDF_EDGES=1 SHODH_SPREAD_FIX=1 SHODH_CAUSAL_ORIGIN=1 SHODH_CAUSAL_ORIGIN_DEBUG=1"
gh workflow run locomo-recall.yml --ref feat/eval-graph-fidelity -f mode=lineage -f extra_env="$L1" >/dev/null 2>&1
sleep 15
R=$(gh run list --workflow=locomo-recall.yml -L 1 --json databaseId -q '.[0].databaseId')
for i in $(seq 1 220); do S=$(gh run view "$R" --json status -q .status 2>/dev/null); [ "$S" = "completed" ] && break; sleep 25; done
echo "=== lineage FUNNEL ($R: $(gh run view "$R" --json conclusion -q .conclusion)) ==="
gh run view "$R" --log 2>/dev/null | grep -oE "CAUSAL_FUNNEL qents=[0-9]+ cue=(true|false) origins=[0-9]+" | sort | uniq -c | sort -rn | head -20
echo "--- P@1 table ---"
gh run view "$R" --log 2>/dev/null | grep -E "\| (vamana_only|\+spreading|full) \|" | sed 's/^[^|]*| /| /' | sort -u
echo DONE
