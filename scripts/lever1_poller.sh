set -u
L1="SHODH_GRAPH_EXTRACTED_PREDICATES=1 SHODH_GRAPH_PREDICATE_WEIGHTS=1 SHODH_GRAPH_IDF_EDGES=1 SHODH_SPREAD_FIX=1 SHODH_CAUSAL_ORIGIN=1"
cap(){ gh run view "$1" --log 2>/dev/null | grep -E "\| (vamana_only|\+spreading|\+bm25|\+rerank|\+facts|full) \|" | sed 's/^[^|]*| /| /' | sort -u; }
waitfor(){ for i in $(seq 1 220); do S=$(gh run view "$1" --json status -q .status 2>/dev/null); [ "$S" = "completed" ] && return 0; sleep 25; done; }
disp(){ gh workflow run locomo-recall.yml --ref feat/eval-graph-fidelity -f mode="$1" -f extra_env="$2" >/dev/null 2>&1; sleep 15; gh run list --workflow=locomo-recall.yml -L 1 --json databaseId -q '.[0].databaseId'; }
run(){ R=$(disp "$1" "$2"); waitfor "$R"; echo "=== $3 ($R: $(gh run view "$R" --json conclusion -q .conclusion 2>/dev/null)) ==="; cap "$R"; echo; }
run lineage ""      "lineage BASELINE (re-confirm P@1=0)"
run lineage "$L1"   "lineage LEVER1+CAUSAL-ORIGIN"
run multi_hop ""    "multihop OFF"
run multi_hop "$L1" "multihop LEVER1-ON"
echo "ALL DONE"
