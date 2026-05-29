"""Cross-process comparison for Sliding Stage OPM Repeatability.

Compares analysis results from two different datasets (e.g., module assembly
vs final product assembly) to identify differences in OPM performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analyzer import AnalysisResult, POSITION_GROUPS
from .data_loader import POSITION_LABELS


@dataclass
class PositionDelta:
    """Comparison result for a single position."""
    position: str
    current_rep_max: float
    reference_rep_max: float
    delta_rep_max: float
    current_rep_1sigma: float
    reference_rep_1sigma: float
    delta_rep_1sigma: float
    current_opm_max: float
    reference_opm_max: float
    delta_opm_max: float
    pct_change_opm_max: float  # percentage change


@dataclass
class CompareResult:
    """Aggregate comparison between current and reference analysis."""
    current_label: str
    reference_label: str
    current_range: str
    reference_range: str
    position_deltas: dict[str, PositionDelta]
    group_summary: dict[str, dict]  # group_name -> {metric: (curr, ref, delta)}

    @property
    def max_delta_opm(self) -> float:
        if not self.position_deltas:
            return 0.0
        return max(abs(d.delta_opm_max) for d in self.position_deltas.values())


def compare_results(current: AnalysisResult,
                    reference: AnalysisResult,
                    use_best_window: bool = True) -> CompareResult:
    """Compare two analysis results position by position.

    Args:
        current: Current analysis result.
        reference: Reference analysis result.
        use_best_window: Use best window positions if available.

    Returns:
        CompareResult with per-position deltas and group summary.
    """
    curr_src = (current.best_window.positions
                if use_best_window and current.best_window
                else current.all_positions)
    ref_src = (reference.best_window.positions
               if use_best_window and reference.best_window
               else reference.all_positions)

    position_deltas = {}
    for pos in POSITION_LABELS:
        curr_pos = curr_src.get(pos)
        ref_pos = ref_src.get(pos)
        if curr_pos is None or ref_pos is None:
            continue

        delta_opm = curr_pos.opm_max - ref_pos.opm_max
        pct = (delta_opm / ref_pos.opm_max * 100) if ref_pos.opm_max != 0 else 0.0

        position_deltas[pos] = PositionDelta(
            position=pos,
            current_rep_max=curr_pos.rep_max,
            reference_rep_max=ref_pos.rep_max,
            delta_rep_max=curr_pos.rep_max - ref_pos.rep_max,
            current_rep_1sigma=curr_pos.rep_1sigma,
            reference_rep_1sigma=ref_pos.rep_1sigma,
            delta_rep_1sigma=curr_pos.rep_1sigma - ref_pos.rep_1sigma,
            current_opm_max=curr_pos.opm_max,
            reference_opm_max=ref_pos.opm_max,
            delta_opm_max=delta_opm,
            pct_change_opm_max=pct,
        )

    # Group summary
    group_summary = {}
    for group_name, group_positions in POSITION_GROUPS.items():
        curr_opms = [position_deltas[p].current_opm_max
                     for p in group_positions if p in position_deltas]
        ref_opms = [position_deltas[p].reference_opm_max
                    for p in group_positions if p in position_deltas]
        if curr_opms and ref_opms:
            curr_mean = float(np.mean(curr_opms))
            ref_mean = float(np.mean(ref_opms))
            group_summary[group_name] = {
                "current_mean_opm": curr_mean,
                "reference_mean_opm": ref_mean,
                "delta_mean_opm": curr_mean - ref_mean,
                "current_max_opm": float(max(curr_opms)),
                "reference_max_opm": float(max(ref_opms)),
            }

    return CompareResult(
        current_label="Current",
        reference_label="Reference",
        current_range=current.range_label,
        reference_range=reference.range_label,
        position_deltas=position_deltas,
        group_summary=group_summary,
    )


def get_compare_table(result: CompareResult) -> list[dict]:
    """Generate comparison table rows for UI display.

    Returns list of dicts with: Position, Curr OPM, Ref OPM, Δ OPM, Δ%, Curr Rep.Max, Ref Rep.Max, Δ Rep.Max
    """
    rows = []
    for pos in POSITION_LABELS:
        if pos not in result.position_deltas:
            continue
        d = result.position_deltas[pos]
        rows.append({
            "Position": pos,
            "Curr OPM Max": round(d.current_opm_max, 3),
            "Ref OPM Max": round(d.reference_opm_max, 3),
            "Δ OPM (nm)": round(d.delta_opm_max, 3),
            "Δ OPM (%)": f"{d.pct_change_opm_max:+.1f}%",
            "Curr Rep.Max": round(d.current_rep_max, 3),
            "Ref Rep.Max": round(d.reference_rep_max, 3),
            "Δ Rep.Max": round(d.delta_rep_max, 3),
        })

    # Group summary rows
    for group_name in ["Center", "Side", "Edge"]:
        if group_name in result.group_summary:
            g = result.group_summary[group_name]
            pct = (g["delta_mean_opm"] / g["reference_mean_opm"] * 100
                   if g["reference_mean_opm"] != 0 else 0)
            rows.append({
                "Position": f"[{group_name}]",
                "Curr OPM Max": round(g["current_max_opm"], 3),
                "Ref OPM Max": round(g["reference_max_opm"], 3),
                "Δ OPM (nm)": round(g["current_max_opm"] - g["reference_max_opm"], 3),
                "Δ OPM (%)": f"{pct:+.1f}%",
                "Curr Rep.Max": "-",
                "Ref Rep.Max": "-",
                "Δ Rep.Max": "-",
            })

    return rows
