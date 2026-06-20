"""Formal one-click PDF inspection report (matplotlib PdfPages — no new deps).

Two A4 pages: (1) header + verdict + dual-spec summary + per-position table +
MSA/QC one-liners + sign-off block; (2) wafer map + MSA per-position chart.

Korean text needs a CJK-capable font; matplotlib's default cannot render Hangul.
We temporarily switch font.family to an available Korean font (Malgun Gothic on
Windows) for the duration of the build, and restore it afterwards.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ..core.analyzer import AnalysisResult, get_dual_summary_table

A4 = (8.27, 11.69)
_FG = "#1e1e2e"
_MUTED = "#6c7086"
_PASS = "#2e7d32"
_FAIL = "#c62828"
_LINE = "#9399b2"


def _korean_font() -> Optional[str]:
    for name in ("Malgun Gothic", "Gulim", "Batang", "NanumGothic",
                 "Noto Sans CJK KR", "AppleGothic"):
        try:
            if font_manager.findfont(name, fallback_to_default=False):
                return name
        except Exception:
            continue
    return None


def _cell(row: dict, key: str) -> str:
    """Format a dual (raw/robust) metric cell: 'raw' or 'raw (rob)' when they differ."""
    raw = row.get(key)
    rob = row.get(f"{key} (rob)")
    if rob is None or rob == raw or raw == "-":
        return f"{raw}"
    return f"{raw} ({rob})"


def _verdict_text(passed: Optional[bool]) -> tuple[str, str]:
    if passed is None:
        return "판정 불가", _MUTED
    return ("PASS", _PASS) if passed else ("FAIL", _FAIL)


def _page_summary(result: AnalysisResult, robust_result, msa_result,
                  qc_result, meta: dict) -> Figure:
    fig = Figure(figsize=A4)
    fig.patch.set_facecolor("#ffffff")

    fig.text(0.5, 0.965, "Sliding Stage OPM Repeatability — 검수 리포트",
             ha="center", fontsize=15, fontweight="bold", color=_FG)
    fig.add_artist(Line2D([0.05, 0.95], [0.95, 0.95], color=_LINE, lw=1))

    # Header block (two columns of key:value)
    rows = [
        ("장비 ID", meta.get("equipment_id", "—")),
        ("Recipe / Range", f"{result.range_label}  ({meta.get('signal', '')})"),
        ("측정일시", meta.get("measured", "—")),
        ("분석일시", meta.get("analyzed", "—")),
        ("작성자", meta.get("author", "—")),
        ("Lot / Sample", meta.get("lot", "—")),
    ]
    y = 0.925
    for i, (k, v) in enumerate(rows):
        col = i % 2
        x = 0.07 + col * 0.47
        yy = y - (i // 2) * 0.028
        fig.text(x, yy, f"{k}:", fontsize=9, color=_MUTED)
        fig.text(x + 0.13, yy, str(v), fontsize=9, color=_FG)

    # Verdict badge
    vtxt, vcol = _verdict_text(result.overall_pass)
    fig.text(0.5, 0.80, f"종합 판정:  {vtxt}", ha="center",
             fontsize=20, fontweight="bold", color=vcol)

    # Provenance: which repeats the judged values came from (reproducibility key)
    if result.best_window is not None:
        prov = (f"판정 repeat: Best-5 [{result.best_window.repeat_range}] / "
                f"전체 {result.total_repeats} (min mean Rep.1σ window)")
    else:
        prov = f"판정 repeat: 전체 {result.total_repeats}"
    fig.text(0.5, 0.773, prov, ha="center", fontsize=8.5, color=_MUTED)

    # Aggregation rule (the scalar reduction) depends on equipment type
    if result.equipment_type == "dw":
        rep_basis = opm_basis = "Center 5_CM"
    else:
        rep_basis, opm_basis = "Total RMS", "Total Max"

    # Dual-spec lines (raw headline; robust in parentheses)
    lines = []
    if result.spec_limit is not None:
        rv = "—" if result.spec_value is None else f"{result.spec_value:.3f}"
        s = (f"Rep. 1σ ({rep_basis}):  {rv} / {result.spec_limit:g} nm  →  "
             f"{_verdict_text(result.spec_pass)[0]}")
        if robust_result is not None and robust_result.spec_value is not None:
            s += (f"   (robust {robust_result.spec_value:.3f} → "
                  f"{_verdict_text(robust_result.spec_pass)[0]})")
        lines.append(s)
    if result.spec_opm_limit is not None:
        ov = "—" if result.spec_opm_value is None else f"{result.spec_opm_value:.3f}"
        s = (f"OPM Max ({opm_basis}):  {ov} / {result.spec_opm_limit:g} nm  →  "
             f"{_verdict_text(result.spec_opm_pass)[0]}")
        if robust_result is not None and robust_result.spec_opm_value is not None:
            s += (f"   (robust {robust_result.spec_opm_value:.3f} → "
                  f"{_verdict_text(robust_result.spec_opm_pass)[0]})")
        lines.append(s)
    for i, s in enumerate(lines):
        fig.text(0.5, 0.745 - i * 0.022, s, ha="center", fontsize=9.5, color=_FG)

    # Per-position dual summary table
    table_rows = get_dual_summary_table(result, robust_result)
    keys = ["Rep. Max (nm)", "Rep. 1σ (nm)", "OPM Max (nm)", "OPM 1σ (nm)"]
    cols = ["Range", "Position", "Rep.Max (nm)", "Rep.1σ (nm)", "OPM Max (nm)", "OPM 1σ (nm)"]
    cells = [[r["Range"], r["Position"]] + [_cell(r, k) for k in keys] for r in table_rows]
    ax_t = fig.add_axes([0.05, 0.30, 0.90, 0.36])
    ax_t.axis("off")
    if cells:
        tbl = ax_t.table(cellText=cells, colLabels=cols, loc="upper center",
                         cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.25)
        for (rr, cc), cell in tbl.get_celld().items():
            cell.set_edgecolor(_LINE)
            if rr == 0:
                cell.set_facecolor("#e6e9ef")
                cell.set_text_props(fontweight="bold")
    fig.text(0.07, 0.288, "괄호 값 = robust(이상점 제외) · 단위 nm",
             fontsize=7.5, color=_MUTED)

    # MSA + QC one-liners
    yy = 0.252
    if msa_result is not None and getattr(msa_result, "verdict", "N/A") != "N/A":
        m = msa_result
        tol = "—" if m.pct_grr_tol is None else f"{m.pct_grr_tol:.1f}%"
        driver = "공차" if getattr(m, "judged_by", "tv") == "tolerance" else "TV"
        fig.text(0.07, yy, "MSA (반복성 EV Type-1, 재현성 미포함):", fontsize=9,
                 fontweight="bold", color=_FG)
        fig.text(0.07, yy - 0.022,
                 f"%GRR(TV) {m.pct_grr:.1f}% · %GRR(공차) {tol} · ndc {m.ndc} "
                 f"→ {m.verdict}  (판정 기준: %GRR {driver})",
                 fontsize=9, color=_FG)
        yy -= 0.05
    if qc_result is not None:
        warn = sum(1 for c in qc_result.checks if c.status != "PASS")
        fig.text(0.07, yy, f"QC Check: 전체 {qc_result.overall_status} "
                 f"(주의/실패 {warn}/{len(qc_result.checks)})", fontsize=9, color=_FG)
        yy -= 0.03

    # Sign-off block
    fig.add_artist(Line2D([0.05, 0.95], [0.15, 0.15], color=_LINE, lw=1))
    for i, role in enumerate(("작성", "검토", "승인")):
        x = 0.10 + i * 0.30
        fig.text(x + 0.07, 0.125, role, ha="center", fontsize=10,
                 fontweight="bold", color=_FG)
        fig.add_artist(Line2D([x, x + 0.14], [0.075, 0.075], color=_FG, lw=0.8))
        fig.text(x + 0.07, 0.055, "(서명 / 날짜)", ha="center", fontsize=7, color=_MUTED)

    fig.text(0.05, 0.025,
             f"Tool: {meta.get('tool_version', 'OPM Analyzer')} · "
             f"Algorithm: algorithm_spec v2.1 (PMS Q&A 4249) · 생성 {meta.get('analyzed', '')}",
             fontsize=6.5, color=_MUTED)
    return fig


def _page_visuals(result: AnalysisResult, msa_result) -> Figure:
    from .plot_manager import create_wafer_map_figure

    fig = Figure(figsize=A4)
    fig.patch.set_facecolor("#ffffff")
    fig.text(0.5, 0.965, "검수 리포트 — 시각 자료", ha="center",
             fontsize=13, fontweight="bold", color=_FG)

    # Wafer map (rendered from the existing dark figure, embedded as an image)
    wfig = create_wafer_map_figure(result, metric="opm_max", figsize=(7, 6))
    canvas = FigureCanvasAgg(wfig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    plt.close(wfig)
    ax_w = fig.add_axes([0.08, 0.52, 0.84, 0.40])
    ax_w.imshow(img)
    ax_w.axis("off")

    # MSA per-position bar chart (drawn directly from msa_result)
    ax_m = fig.add_axes([0.10, 0.08, 0.82, 0.36])
    if msa_result is not None and getattr(msa_result, "part_means", None):
        positions = list(msa_result.part_means.keys())
        means = [msa_result.part_means[p] for p in positions]
        errs = [msa_result.part_stdevs.get(p, 0.0) for p in positions]
        ax_m.bar(range(len(positions)), means, yerr=errs, capsize=3,
                 color="#4c6ef5", ecolor="#495057")
        ax_m.set_xticks(range(len(positions)))
        ax_m.set_xticklabels(positions, rotation=45, ha="right", fontsize=8)
        ax_m.set_ylabel("OPM (nm)")
        ax_m.set_title("MSA — Mean OPM ± repeat σ per position", fontsize=10)
    else:
        ax_m.axis("off")
        ax_m.text(0.5, 0.5, "MSA 데이터 없음", ha="center", va="center",
                  color=_MUTED, transform=ax_m.transAxes)
    return fig


def build_inspection_report(result: AnalysisResult, output_path: str | Path, *,
                            robust_result=None, msa_result=None,
                            qc_result=None, meta: Optional[dict] = None) -> None:
    """Write the two-page PDF inspection report to ``output_path``."""
    meta = meta or {}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    kfont = _korean_font()
    saved = {k: matplotlib.rcParams[k] for k in ("font.family", "axes.unicode_minus")}
    if kfont:
        matplotlib.rcParams["font.family"] = kfont
    matplotlib.rcParams["axes.unicode_minus"] = False
    try:
        with PdfPages(str(path)) as pdf:
            fig1 = _page_summary(result, robust_result, msa_result, qc_result, meta)
            pdf.savefig(fig1)
            plt.close(fig1)
            fig2 = _page_visuals(result, msa_result)
            pdf.savefig(fig2)
            plt.close(fig2)
    finally:
        matplotlib.rcParams.update(saved)
