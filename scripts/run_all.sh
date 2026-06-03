#!/usr/bin/env bash
# End-to-end pipeline (no reprocessing). Primary = cerebellar volume change + association.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="src:${PYTHONPATH:-}"
CFG="config/cohort.yaml"
DRY=${1:-}

run() { echo "+ $*"; [ "$DRY" = "--dry-run" ] || "$@"; }

echo "== Phase 0: cohort + QC =="
run python3 -m etfvbm.io --config "$CFG"
run python3 -m etfvbm.qc --config "$CFG"

echo "== Phase 2: PRIMARY cerebellar longitudinal =="
run python3 -m etfvbm.cerebellum --config "$CFG"

echo "== Atrophy maps (exploratory; 3mo primary) =="
run python3 -m etfvbm.atrophy_maps --config "$CFG" --session ses-post3mo

echo "== Association (needs clinical.csv filled) =="
run python3 -m etfvbm.association --config "$CFG" --outcome tremor_improvement --timepoint ses-post3mo || \
  echo "  (association skipped — fill config/clinical.csv)"

echo "Done. Outputs in derivatives/."
