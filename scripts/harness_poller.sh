set -u
cap(){ gh run view "$1" --log 2>/dev/null | grep -E "\| (vamana_only|\+spreading|\+bm25|\+rerank|\+facts|full|[0-9])" | sed 's/^[^|]*| /| /' | grep -E "\|.*[0-9]" | sort -u | head -12; }
waitfor(){ for i in $(seq 1 200); do S=$(gh run view "$1" --json status -q .status 2>/dev/null); [ "$S" = "completed" ] && return 0; sleep 25; done; }
disp(){ gh workflow run locomo-recall.yml --ref feat/eval-graph-fidelity -f mode="$1" >/dev/null 2>&1; sleep 15; gh run list --workflow=locomo-recall.yml -L 1 --json databaseId -q '.[0].databaseId'; }
for M in temporal ontology lineage selective_forgetting; do
  R=$(disp "$M"); waitfor "$R"; C=$(gh run view "$R" --json conclusion -q .conclusion 2>/dev/null)
  echo "=== $M ($R: $C) ==="; cap "$R"; echo
done
echo "ALL DONE"
