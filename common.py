"""Shared paths for the pipeline. Run every script from the repository root."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_XLSX = ROOT / "data" / "Data.xlsx"
INTERIM = ROOT / "outputs" / "interim"
FIG = ROOT / "outputs" / "figures"
for p in (INTERIM, FIG):
    p.mkdir(parents=True, exist_ok=True)
