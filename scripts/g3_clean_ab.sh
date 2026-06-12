set -u
cap(){ gh run view "$1" --log 2>/dev/null | grep -E "\| (vamana_only|\+spreading|\+bm25|\+rerank|\+facts|full) \||recall@10|MRR|p@1" | sed 's/^[^|]*| /| /' | sort -u | head -16; }
waitfor(){ for i in $(seq 1 240); do S=$(gh run view "$1" --json status -q .status 2>/dev/null); [ "$S" = "completed" ] && return 0; sleep 25; done; }
disp(){ gh workflow run locomo-recall.yml --ref feat/eval-graph-fidelity -f mode="$1" -f extra_env="$2" -f suite="$3" >/dev/null 2>&1; sleep 15; gh run list --workflow=locomo-recall.yml -L 1 --json databaseId -q '.[0].databaseId'; }
run(){ R=$(disp "$1" "$2" "$3"); waitfor "$R"; echo "=== $4 ($R: $(gh run view "$R" --json conclusion -q .conclusion)) ==="; cap "$R"; echo; }
run ontology  ""                    locomo "ontology  OFF (baseline, same commit)"
run ontology  "SHODH_ACTR_FUSION=1" locomo "ontology  ON  (ACTR_FUSION only)"
run multi_hop ""                    locomo "multihop  OFF (baseline)"
run multi_hop "SHODH_ACTR_FUSION=1" locomo "multihop  ON  (ACTR_FUSION only)"
run lineage   ""                    locomo "lineage   OFF (baseline)"
run lineage   "SHODH_ACTR_FUSION=1" locomo "lineage   ON  (ACTR_FUSION only)"
run recall    ""                    smoke  "recall smoke OFF (regression baseline)"
run recall    "SHODH_ACTR_FUSION=1" smoke  "recall smoke ON  (single-hop guard)"
echo ALL DONE
