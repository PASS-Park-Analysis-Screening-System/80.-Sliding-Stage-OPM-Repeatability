"""Regression validation against the lab reference Tool (PMS Q&A 4249).

Loads the verification dataset (data/25mm, Sample15~19, Height) through the
*actual* analysis pipeline (analyze_recipe) and compares the 4 metrics per
position against the documented reference Tool values
(AFPRepeatabilityAnalysisBatch, 2021-06).

Usage:
    python scripts/validate_against_reference.py
    python scripts/validate_against_reference.py --outlier percentile --value 1.0
    python scripts/validate_against_reference.py --data data/25mm --tol 6

Notes:
    * Rep.Max / Rep.1σ / OPM 1σ should match within a few %.
    * OPM Max carries a known ~4-5% INTRINSIC gap: the reference values exceed
      the raw peak-to-valley of the provided TIFFs, so they are not reproducible
      by any leveling. This is reported but not counted as a failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root: `python scripts/validate_against_reference.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.data_loader import load_recipe  # noqa: E402
from src.core.analyzer import analyze_recipe  # noqa: E402

# Reference Tool values — data/25mm, Sample15~19, Height (PMS Q&A 4249 report).
# Keyed by position -> (rep_max, rep_1sigma, opm_max, opm_1sigma); None = unknown.
REFERENCE = {
    "1_LT": (7.596, 1.736, 104.788, 67.801),
    "2_CT": (None,  None,   73.805, None),
    "3_RT": (9.834, None,   None,   None),   # outlier-excluded reference
    "5_CM": (None,  4.232,  None,   None),
    "7_LB": (None,  1.735,  50.811, 29.766),
    "8_CB": (None,  4.307,  None,   53.435),
    "9_RB": (None,  None,   58.164, 27.100),
}
METRICS = ["rep_max", "rep_1sigma", "opm_max", "opm_1sigma"]
# OPM Max has a documented intrinsic ~4-5% gap (reference > raw P-V) -> informational only.
INTRINSIC = {"opm_max"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/25mm", help="recipe folder (default: data/25mm)")
    ap.add_argument("--outlier", default="percentile",
                    choices=["none", "percentile", "pixels"], help="outlier mode")
    ap.add_argument("--value", type=float, default=1.0, help="outlier threshold")
    ap.add_argument("--tol", type=float, default=6.0, help="pass tolerance in %% (non-intrinsic)")
    args = ap.parse_args()

    recipe = load_recipe(args.data, signal_source="Height", load_profiles=True)
    result = analyze_recipe(
        recipe, window_size=5, equipment_type="iso",
        outlier_mode=args.outlier, outlier_value=args.value,
    )

    print(f"Dataset: {args.data}  repeats={recipe.repeat_count}  "
          f"outlier={args.outlier}:{args.value}  tol=±{args.tol}%\n")
    header = f"{'pos':6} {'metric':11} {'computed':>10} {'reference':>10} {'%err':>8}  status"
    print(header)
    print("-" * len(header))

    failures = 0
    checked = 0
    for pos, refs in REFERENCE.items():
        pr = result.all_positions.get(pos)
        if pr is None:
            continue
        for metric, ref in zip(METRICS, refs):
            if ref is None:
                continue
            val = getattr(pr, metric)
            pct = (val - ref) / ref * 100.0
            checked += 1
            if metric in INTRINSIC:
                status = "INTRINSIC(±~5% expected)"
            elif abs(pct) <= args.tol:
                status = "OK"
            else:
                status = "FAIL"
                failures += 1
            print(f"{pos:6} {metric:11} {val:10.3f} {ref:10.3f} {pct:+7.2f}%  {status}")

    print("-" * len(header))
    print(f"Checked {checked} reference values; "
          f"{failures} failure(s) outside ±{args.tol}% (OPM Max excluded as intrinsic).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
