"""Measurement System Analysis (Gauge R&R) for the Sliding Stage OPM tool.

The characteristic studied is the **per-repeat OPM** (peak-to-valley height of the
Order-1 edge-leveled profile, `PositionResult.opm_values`). In gauge-study terms:

    part   = wafer position (1_LT .. 9_RB) — each has its own true OPM signature
    trial  = repeat measurement (the stage slides away and re-measures)
    appraiser = NONE — a single automated tool, measured once per condition

Because there is only one (automated) appraiser within a single recipe dataset, the
reproducibility / appraiser-variation (AV) term is **not estimable here**. This is
therefore an honest **repeatability-focused Type-1 gauge study**: %GRR == %EV. A full
Gauge R&R with reproducibility would treat separate datasets (Lot / PM cycle / day) as
appraisers — a future Tier-2 extension, not computed here.

Variance components (one-way), then AIAG-style ratios:

    EV (repeatability 1σ) = sqrt( Σ SS_i / Σ df_i )      # df-weighted pooled within-part
                                                          # stdev (== sqrt(mean s_i²) if balanced)
    PV (part variation 1σ) = sqrt(max(0, Var(part means) - EV²/n0))   # bias-corrected; n0 =
                                                          # one-way effective trials (== n if balanced)
    TV = sqrt(EV² + PV²)
    %EV = EV/TV, %PV = PV/TV, %GRR = %EV
    ndc = floor(1.41 · PV / GRR), capped at NDC_CAP; EV≈0 with PV>0 = many categories (excellent)
    %GRR_tol = (study_sigma · GRR) / tolerance           # tolerance = OPM-Max spec limit, a
                                                          # ONE-SIDED upper acceptance limit (lower
                                                          # bound 0), not a bilateral USL-LSL band.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .analyzer import AnalysisResult
from .data_loader import POSITION_LABELS

# AIAG study-variation multiplier: 6 = 99.73% (4th ed default); 5.15 = 99% (legacy).
STUDY_SIGMA = 6.0

# Cap for ndc: values above this are meaningless (AIAG treats ndc>=5 as adequate),
# and prevents float-noise EV from surfacing an absurd billions-magnitude integer.
NDC_CAP = 99

_NOTE_EV_ONLY = ("측정자(AV/재현성) 축 없음 — 단일 자동 장비 기준 "
                 "반복성(EV) 중심 Type-1 Gage. %GRR = %EV.")


@dataclass
class MSAResult:
    """Gauge-study result for one analysis (raw OPM characteristic)."""
    characteristic: str = "OPM (nm)"
    n_parts: int = 0
    n_trials: int = 0
    ev: float = 0.0                 # repeatability 1σ (nm)
    pv: float = 0.0                 # part variation 1σ (nm)
    tv: float = 0.0                 # total 1σ (nm)
    pct_ev: float = 0.0             # %EV (= %GRR here)
    pct_pv: float = 0.0
    pct_grr: float = 0.0            # = pct_ev (AV not applicable)
    ndc: int = 0
    tolerance: Optional[float] = None       # OPM-Max spec limit used for %GRR-to-tol
    pct_grr_tol: Optional[float] = None
    study_sigma: float = STUDY_SIGMA
    verdict: str = "N/A"
    judged_by: str = "tv"                   # which ratio drove the verdict: "tolerance" | "tv"
    note: str = ""
    part_means: dict = field(default_factory=dict)   # position -> mean OPM (nm)
    part_stdevs: dict = field(default_factory=dict)   # position -> repeat stdev (nm)


def _verdict(pct: float) -> str:
    if pct < 10.0:
        return "우수 (수용)"
    if pct <= 30.0:
        return "조건부 수용"
    return "부적합"


def compute_msa(result: AnalysisResult, study_sigma: float = STUDY_SIGMA) -> MSAResult:
    """Compute the repeatability-focused gauge study from an analysis result.

    Uses ALL repeats (``result.all_positions``) on the raw per-repeat OPM. The
    tolerance for %GRR-to-tolerance is the effective OPM-Max spec limit of the
    analysis (``result.spec_opm_limit``), so it honors any preset override.
    """
    source = result.all_positions
    tolerance = result.spec_opm_limit

    means: dict[str, float] = {}
    stdevs: dict[str, float] = {}
    ss_list: list[float] = []    # per-position sum of squared deviations
    dof_list: list[int] = []     # per-position (n_i - 1)
    counts: list[int] = []
    for pos in POSITION_LABELS:
        if pos not in source:
            continue
        vals = np.asarray(source[pos].opm_values, dtype=float)
        if vals.size < 2:
            continue
        means[pos] = float(vals.mean())
        s = float(vals.std(ddof=1))
        stdevs[pos] = s
        ss_list.append(s * s * (vals.size - 1))
        dof_list.append(vals.size - 1)
        counts.append(int(vals.size))

    if len(means) < 2 or sum(dof_list) <= 0:
        return MSAResult(tolerance=tolerance, study_sigma=study_sigma,
                         verdict="N/A",
                         note="데이터 부족 — MSA에는 위치 ≥2, repeat ≥2가 필요합니다.")

    # EV: df-weighted pooled within-part stdev (== sqrt(mean s_i²) when balanced).
    ev = float(np.sqrt(sum(ss_list) / sum(dof_list)))

    # n0: one-way effective trials/part for the PV bias term (== n when balanced).
    p = len(counts)
    total_n = sum(counts)
    n0 = ((total_n - sum(c * c for c in counts) / total_n) / (p - 1)) if p > 1 else float(counts[0])

    part_means = np.array(list(means.values()), dtype=float)
    var_pm = float(part_means.var(ddof=1))
    pv = float(np.sqrt(max(0.0, var_pm - (ev * ev) / n0))) if n0 > 0 else float(np.sqrt(var_pm))
    tv = float(np.sqrt(ev * ev + pv * pv))

    pct_ev = 100.0 * ev / tv if tv > 0 else 0.0
    pct_pv = 100.0 * pv / tv if tv > 0 else 0.0

    # ndc: capped; EV≈0 with PV>0 means MANY distinct categories (best case), not 0.
    if pv <= 0.0:
        ndc = 0
    elif ev > 0.0:
        ndc = min(int(1.41 * pv / ev), NDC_CAP)
    else:
        ndc = NDC_CAP

    pct_grr_tol = (100.0 * study_sigma * ev / tolerance
                   if tolerance and tolerance > 0 else None)
    judged_by = "tolerance" if pct_grr_tol is not None else "tv"
    judge = pct_grr_tol if pct_grr_tol is not None else pct_ev

    note = _NOTE_EV_ONLY
    if pct_grr_tol is not None:
        note += " 공차는 OPM 상한(단측, 하한 0) 기준."

    return MSAResult(
        characteristic="OPM (nm)",
        n_parts=len(means), n_trials=int(round(float(np.mean(counts)))),
        ev=ev, pv=pv, tv=tv,
        pct_ev=pct_ev, pct_pv=pct_pv, pct_grr=pct_ev, ndc=ndc,
        tolerance=tolerance, pct_grr_tol=pct_grr_tol, study_sigma=study_sigma,
        verdict=_verdict(judge), judged_by=judged_by, note=note,
        part_means=means, part_stdevs=stdevs)
