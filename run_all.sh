#!/usr/bin/env bash
# Reproduce the full analysis. Run from the repository root.
set -euo pipefail
for s in 01_prepare_data 02_geox_run_design 03_enumerate_and_score \
         04_geox_metrics_and_pseudo_test 05_robustness 06_make_figures; do
  echo "== $s"; python src/$s.py | tee outputs/interim/$s.log
done
# Optional: rebuild the Word report (needs Node.js and `npm install docx`)
# node report/build_report.js
