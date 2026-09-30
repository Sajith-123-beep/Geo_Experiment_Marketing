# Geo Experiment Design: +40% Paid-Social Test (US States)

Design of a US state-level geo experiment for a fictional online retailer (16 states, synthetic data).
The question: should the business test a 40% daily increase in paid-social spend from 1 to 28 June 2026,
and where? The primary outcome is finance net revenue, and the planning effect is +2.5% on treated states.

**Recommendation: PROCEED** with treatment in **Ohio, Pennsylvania and Missouri** (16.0% of audited revenue,
$274,106 extra spend) against an 11-state BAU comparison pool (CA, CO, GA, IN, KY, MI, NC, NV, SC, TN, WI).
Estimated MDE is 1.3-1.8% of treated revenue. Fallback: WI + TN + KY ($224k).

The full write-up is in [`report/Geo_Experiment_Report.docx`](report/Geo_Experiment_Report.docx)
(four sections: raw data analysis, hypothesis, experiment design, risks).

## Method in brief
1. **Clean the data** (`01`): remove 20 duplicate rows, normalise state labels, impute 12 unusable Georgia
   revenue days, repair Nevada's negative spend and orders.
2. **Run GeoX `run_design`** (`02`): HEAVY_UP, TBR, +40% budget, 28 days. Its top pick (KY+PA+TN+WI) needs
   $330k and breaks the $300k cap.
3. **Enumerate every feasible 2-4 state treatment set** (`03`) under the business rules: 12-25% revenue share,
   $300k cap, `can_increase_spend`, delivery groups (NC/SC), audience overlap, calendar conflicts
   (FL promo, TX expansion, CA contract). Each set is scored with a leave-one-window-out placebo test using the
   same non-negative-slope TBR regression as GeoX. 277 designs were feasible; 126 could not detect 2.5%.
4. **GeoX metrics and pseudo-test** (`04`): GeoX `get_r2` / `get_mde` on the shortlist, plus `geox.analyze` on
   4-31 May with 0%, +1.5% and +2.5% lift injected.
5. **Robustness** (`05`): control-pool stress tests (drop CA, GA, NV, CO; add TX, FL; exclude the spring-sale week).
6. **Figures** (`06`) for the report.

## Repository layout
```
data/Data.xlsx              input workbook (4 sheets: Daily metrics, Markets, Audience overlap, Operations calendar)
src/common.py               shared paths
src/01_prepare_data.py      cleaning and imputation
src/02_geox_run_design.py   official GeoX design search
src/03_enumerate_and_score.py  constraint enumeration + placebo MDE (uses placebo_tbr.py)
src/04_geox_metrics_and_pseudo_test.py  GeoX R2/MDE and analyze() pseudo-tests
src/05_robustness.py        stress tests and power curves
src/06_make_figures.py      report charts
src/placebo_tbr.py          shared placebo / TBR helper (imported by 05 and 06)
report/                     final Word report and the Node script that builds it
outputs/figures/            charts used in the report
run_all.sh                  runs the whole pipeline
```

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh            # run from the repo root
```
Tested with Python 3.12 and meridian-geox 1.0.1. Intermediate files are written to `outputs/interim/`
(pickles are git-ignored; logs are kept). To rebuild the Word report: `npm install docx && node report/build_report.js`.

## Caveats
- **GeoLift (R) was not run.** Only Google's GeoX was used. Cross-checking in GeoLift is a launch condition
  in the report.
- The per-design placebo MDE and the constraint enumeration are custom code built on GeoX's approach,
  not standard GeoX output.
- GeoX 1.0.1 `analyze()` returned degenerate confidence-interval and p-value fields in the pseudo-test,
  so only its point estimates are used.
- Data are synthetic and there are no test-period outcomes. MDE figures are planning estimates, not promised results.

## References
- https://github.com/facebookincubator/GeoLift
- https://developers.google.com/meridian/geox/intro-to-design
- https://github.com/google/meridian-geox
