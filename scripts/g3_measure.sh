set -u
G3="SHODH_ACTR_FUSION=1 SHODH_GRAPH_EXTRACTED_PREDICATES=1 SHODH_GRAPH_PREDICATE_WEIGHTS=1 SHODH_GRAPH_IDF_EDGES=1 SHODH_SPREAD_FIX=1 SHODH_CAUSAL_ORIGIN=1"
cap(){ gh run view "$1" --log 2>/dev/null | grep -E "\| (vamana_only|\+spreading|\+bm25|\+rerank|\+facts|full) \|" | sed 's/^[^|]*| /| /' | sort -u; }
waitfor(){ for i in $(seq 1 240); do S=$(gh run view "$1" --json status -q .status 2>/dev/null); [ "$S" = "completed" ] && return 0; sleep 25; done; }
disp(){ gh workflow run locomo-recall.yml --ref feat/eval-graph-fidelity -f mode="$1" -f extra_env="$2" >/dev/null 2>&1; sleep 15; gh run list --workflow=locomo-recall.yml -L 1 --json databaseId -q '.[0].databaseId'; }
run(){ R=$(disp "$1" "$2"); waitfor "$R"; echo "=== $3 ($R: $(gh run view "$R" --json conclusion -q .conclusion)) ==="; cap "$R"; echo; }
run ontology "$G3"  "ontology G3-ON"
run multi_hop "$G3" "multihop G3-ON"
run lineage "$G3"   "lineage G3-ON"
run recall "SHODH_ACTR_FUSION=1" "LoCoMo recall GUARD (G3-only)"
echo ALL DONE
