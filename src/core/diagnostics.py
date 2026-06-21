"""Per-position outlier root-cause DIAGNOSTIC — a reference 감별 가이드, NOT a verdict.

Given the per-repeat profile stack for one position, this module surfaces the
signatures that distinguish *transient* causes (particle/contamination, sensor
glitch, vibration/noise) from *structural* ones (sample-chuck flatness, ball-screw
lubrication/straightness), and offers a heuristic guide WITH the evidence shown.

Read-only and self-contained: it never feeds the Spec judgment. Official pass/fail
remains the aggregate in the Spec panel; the judgment algorithm is unchanged.

Signatures (per selected position, across N repeats, Order-1 edge-only leveled):
    - excluded pixels (repeat Max-Min outliers): count, location(mm), spike width,
      and how many repeats carry the extreme (1 -> transient; many -> systematic).
    - drift: trend of per-repeat OPM across repeat index (lubrication/thermal).
    - Rep.1sigma vs OPM Max + coarse-resolution persistence (spike collapses, bow
      persists) -> transient vs structural.
    - dominant spatial period (FFT) vs an optional ball-screw lead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .data_loader import RecipeData, POSITION_LABELS
from .flatten import edge_only_flatten
from .analyzer import (_exclude_outlier_pixels, SPEC_REPEATABILITY,
                       POSITION_GROUPS, resample_profile)

# Confidence labels (Korean, user-facing)
LOW, MID, HIGH = "낮음", "중간", "높음"

_DISCLAIMER = "참고용 감별 가이드 — 공식 판정 아님(공식 합격/불합격은 Spec 패널 집계 기준)"


@dataclass
class Guide:
    """One candidate cause with its confidence and the evidence behind it."""
    cause: str
    confidence: str   # LOW / MID / HIGH
    evidence: str


@dataclass
class PositionDiagnosis:
    position: str
    group: str                       # Edge / Center / Side / -
    n_repeats: int
    n_excluded: int
    excluded_x_mm: list = field(default_factory=list)
    spike_width_px: int = 0
    spike_width_mm: float = 0.0
    n_repeats_involved: int = 0
    worst_repeat: int = 0            # 1-based
    rep_max_before: float = 0.0
    rep_max_after: float = 0.0
    rep_1sigma: float = 0.0
    spec_rep_1sigma: Optional[float] = None
    opm_max: float = 0.0
    drift_slope: float = 0.0         # nm per repeat
    drift_r2: float = 0.0
    drift_dir: str = "안정"          # 증가 / 감소 / 안정
    bow_persist_ratio: float = 1.0   # coarse OPM / fine OPM (1=persists -> structural)
    dominant_period_mm: float = 0.0
    dominant_power: float = 0.0
    lead_mm: Optional[float] = None
    lead_match: Optional[bool] = None
    guides: list = field(default_factory=list)
    note: str = ""
    disclaimer: str = _DISCLAIMER


# --------------------------------------------------------------------------- #
# Small signal helpers
# --------------------------------------------------------------------------- #

def _group_of(position: str) -> str:
    for grp, members in POSITION_GROUPS.items():
        if position in members:
            return grp
    return "-"


def _position_stack(recipe: RecipeData, position: str):
    """(repeats, pixels) Order-1 edge-only leveled stack + x_mm for one position.

    Returns (None, None) if fewer than 2 repeats hold this position.
    """
    profs, x_mm = [], None
    for r in recipe.repeats:
        if position in r.profiles:
            prof = r.profiles[position]
            profs.append(edge_only_flatten(prof.z_nm, order=1, edge_percent=1.0))
            if x_mm is None:
                x_mm = np.asarray(prof.x_mm, dtype=np.float64)
    if len(profs) < 2:
        return None, None
    return np.asarray(profs, dtype=np.float64), x_mm


def _max_run_length(mask: np.ndarray) -> int:
    """Longest contiguous run of True (e.g. widest excluded-pixel cluster)."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return int(best)


def _repeats_involved(stack: np.ndarray, ex_idx: np.ndarray):
    """At the excluded pixels, how many repeats carry a gross deviation, and which
    single repeat holds the largest deviation. 1 -> transient; many -> systematic."""
    if len(ex_idx) == 0:
        return 0, 0
    sub = stack[:, ex_idx]                       # (n_rep, n_ex)
    med = np.median(sub, axis=0)                 # (n_ex,)
    dev = np.abs(sub - med)                      # (n_rep, n_ex)
    mad = np.median(dev, axis=0)                 # (n_ex,)
    scale = 1.4826 * mad
    scale = np.where(scale <= 0, 1e-9, scale)
    z = dev / scale
    involved = z > 3.0                           # gross outlier per (repeat, pixel)
    n_inv = int(np.count_nonzero(involved.any(axis=1)))
    worst = int(np.argmax(dev) // dev.shape[1]) + 1
    return max(n_inv, 1), worst                  # >=1: the worst repeat always drives it


def _trend(y: np.ndarray):
    """Linear slope (per index) and R^2 of a per-repeat series."""
    n = len(y)
    if n < 3:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    coef = np.polyfit(x, y, 1)
    slope = float(coef[0])
    yhat = np.polyval(coef, x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return slope, max(0.0, r2)


def _drift_dir(slope: float, r2: float, y: np.ndarray) -> str:
    if len(y) < 3 or y.mean() <= 1e-9:
        return "안정"
    rel = slope * (len(y) - 1) / y.mean()        # fractional end-to-start change
    if r2 >= 0.6 and abs(rel) >= 0.15:
        return "증가" if slope > 0 else "감소"
    return "안정"


def _bow_persist(recipe: RecipeData, position: str, factor: int = 20) -> float:
    """coarse-resolution OPM / fine OPM, median over repeats.

    A localized spike averages out under block-decimation (ratio << 1 = transient);
    a global bow survives (ratio ~ 1 = structural)."""
    ratios = []
    for r in recipe.repeats:
        if position not in r.profiles:
            continue
        z = r.profiles[position].z_nm
        z_fine = edge_only_flatten(z, order=1, edge_percent=1.0)
        fine = float(z_fine.max() - z_fine.min())
        z_coarse = edge_only_flatten(resample_profile(z, factor), order=1,
                                     edge_percent=1.0)
        coarse = float(z_coarse.max() - z_coarse.min())
        if fine > 1e-9:
            ratios.append(min(1.0, coarse / fine))
    return float(np.median(ratios)) if ratios else 1.0


def dominant_spatial_period(series: np.ndarray, dx_mm: float):
    """Dominant spatial period (mm) and its power from a 1-D spatial series.

    Detrends, Hann-windows, takes the rFFT, and returns the period (1/freq) of the
    strongest non-DC component. Used to flag ball-screw-lead periodicity."""
    y = np.asarray(series, dtype=np.float64)
    n = len(y)
    if n < 16 or dx_mm <= 0:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    y = y - np.polyval(np.polyfit(x, y, 1), x)   # remove mean + linear trend
    spec = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, d=dx_mm)          # cycles per mm
    if len(spec) <= 2:
        return 0.0, 0.0
    # Ignore DC and ultra-low frequencies (period longer than half the scan).
    min_freq = 2.0 / (n * dx_mm)
    band = freqs >= min_freq
    band[0] = False
    if not band.any():
        return 0.0, 0.0
    masked = np.where(band, spec, 0.0)
    k = int(np.argmax(masked))
    f = freqs[k]
    period = 1.0 / f if f > 0 else 0.0
    return float(period), float(spec[k])


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

def diagnose_position(recipe: RecipeData, position: str,
                      mode: str = "percentile", value: float = 1.0,
                      lead_mm: Optional[float] = None) -> PositionDiagnosis:
    """Full per-position diagnosis (signatures + heuristic 감별 가이드)."""
    group = _group_of(position)
    stack, x_mm = _position_stack(recipe, position)
    if stack is None:
        d = PositionDiagnosis(position=position, group=group, n_repeats=0,
                              n_excluded=0, note="반복 2개 미만 — 진단 불가")
        d.guides = [Guide("데이터 부족", LOW, "이 포지션의 반복이 2개 미만입니다.")]
        return d

    n_rep, n_px = stack.shape
    dx_mm = float(abs(x_mm[1] - x_mm[0])) if (x_mm is not None and len(x_mm) > 1) else 0.0

    pixel_range = stack.max(axis=0) - stack.min(axis=0)
    valid = _exclude_outlier_pixels(stack, mode, value)
    excluded = ~valid
    n_excluded = int(excluded.sum())
    rep_max_before = float(pixel_range.max())
    rep_max_after = float(pixel_range[valid].max()) if valid.any() else 0.0

    spike_width_px = _max_run_length(excluded)
    spike_width_mm = spike_width_px * dx_mm
    ex_idx = np.where(excluded)[0]
    excluded_x_mm = ([float(x_mm[i]) for i in ex_idx[:50]]
                     if (x_mm is not None and len(ex_idx)) else [])
    n_involved, worst_repeat = _repeats_involved(stack, ex_idx)

    if n_rep >= 2 and valid.any():
        pixel_stds = stack.std(axis=0, ddof=1)
        rep_1sigma = float(np.sqrt(np.mean(pixel_stds[valid] ** 2)))
    else:
        rep_1sigma = 0.0
    spec_rep = SPEC_REPEATABILITY.get(int(round(getattr(recipe, "range_mm", 0) or 0)))

    opm_arr = np.array([float(stack[i, valid].max() - stack[i, valid].min())
                        if valid.any() else 0.0 for i in range(n_rep)])
    opm_max = float(opm_arr.max())
    drift_slope, drift_r2 = _trend(opm_arr)
    drift_dir = _drift_dir(drift_slope, drift_r2, opm_arr)

    bow_persist = _bow_persist(recipe, position)
    period_mm, period_pow = dominant_spatial_period(pixel_range, dx_mm)
    lead_match = None
    if lead_mm and lead_mm > 0 and period_mm > 0:
        lead_match = abs(period_mm - lead_mm) / lead_mm < 0.15

    d = PositionDiagnosis(
        position=position, group=group, n_repeats=n_rep, n_excluded=n_excluded,
        excluded_x_mm=excluded_x_mm, spike_width_px=spike_width_px,
        spike_width_mm=spike_width_mm, n_repeats_involved=n_involved,
        worst_repeat=worst_repeat, rep_max_before=rep_max_before,
        rep_max_after=rep_max_after, rep_1sigma=rep_1sigma, spec_rep_1sigma=spec_rep,
        opm_max=opm_max, drift_slope=drift_slope, drift_r2=drift_r2,
        drift_dir=drift_dir, bow_persist_ratio=bow_persist,
        dominant_period_mm=period_mm, dominant_power=period_pow, lead_mm=lead_mm,
        lead_match=lead_match)
    d.guides = _build_guides(d, n_px)
    return d


def _build_guides(d: PositionDiagnosis, n_px: int) -> list:
    """Conservative signature -> cause mapping. Each guide cites its evidence."""
    g: list = []
    narrow = max(3, int(0.003 * n_px))           # "narrow spike" px threshold
    rep_small = (d.spec_rep_1sigma is not None
                 and d.rep_1sigma <= 0.5 * d.spec_rep_1sigma)

    # Structural: bow persists at coarse resolution and repeatability is good.
    if d.bow_persist_ratio >= 0.7 and (rep_small or d.n_excluded == 0):
        conf = HIGH if (rep_small and d.bow_persist_ratio >= 0.85) else MID
        ev = (f"코어스 해상도에서 형상 잔존(비 {d.bow_persist_ratio:.2f}), "
              f"Rep.1σ {d.rep_1sigma:.2f}nm"
              + (f"≤½·Spec({d.spec_rep_1sigma:.1f})" if rep_small else "")
              + f", OPM Max {d.opm_max:.1f}nm(bow)")
        g.append(Guide("구조적: Sample chuck 평탄도 / 볼스크류 (이상치 아님)", conf, ev))

    # Time-dependent drift across repeats.
    if d.drift_dir in ("증가", "감소"):
        g.append(Guide("구조적/시간의존: 볼스크류 윤활 열화·열·정착", MID,
                       f"반복 진행에 따라 OPM {d.drift_dir} 추세"
                       f"(R²={d.drift_r2:.2f}, 기울기 {d.drift_slope:+.2f} nm/회)"))

    # Ball-screw lead periodicity (only when a lead was provided).
    if d.lead_match:
        g.append(Guide("구조적: 볼스크류 리드 주기성 의심", MID,
                       f"지배 공간 주기 {d.dominant_period_mm:.3f}mm "
                       f"≈ 입력 리드 {d.lead_mm:.3f}mm"))

    # Transient: a narrow spike present in essentially one repeat.
    if d.n_excluded > 0 and d.spike_width_px <= narrow and d.n_repeats_involved <= 1:
        g.append(Guide("일시적: 파티클/계측 글리치", HIGH,
                       f"좁은 스파이크(폭 {d.spike_width_px}px≈{d.spike_width_mm:.3f}mm)가 "
                       f"{d.n_repeats_involved}개 반복(R{d.worst_repeat})에서만, "
                       f"코어스에서 약화(비 {d.bow_persist_ratio:.2f}); "
                       f"Rep.Max {d.rep_max_before:.1f}→제외후 {d.rep_max_after:.1f}nm"))

    # Vibration / broad noise across many pixels or repeats.
    if d.n_excluded > 0 and (d.spike_width_px > narrow
                             or d.n_repeats_involved >= max(2, d.n_repeats // 2)):
        g.append(Guide("일시적: 진동/노이즈(성능)", MID,
                       f"넓거나 다중 반복 편차(폭 {d.spike_width_px}px, "
                       f"관여 반복 {d.n_repeats_involved}/{d.n_repeats})"))

    # Edge-position bow -> chuck flatness / edge effect.
    if d.group == "Edge" and d.bow_persist_ratio >= 0.7:
        g.append(Guide("구조적: Sample chuck 평탄도 / 엣지 효과", LOW,
                       f"엣지 포지션({d.position}) + 형상 잔존(비 {d.bow_persist_ratio:.2f})"))

    if not g:
        g.append(Guide("시료 표면 또는 불명 — 추가 확인 권장", LOW,
                       f"뚜렷한 이상치/구조 신호 없음(제외 {d.n_excluded}px, "
                       f"Rep.1σ {d.rep_1sigma:.2f}nm)"))

    order = {HIGH: 0, MID: 1, LOW: 2}
    g.sort(key=lambda x: order.get(x.confidence, 3))
    return g


def worst_position(recipe: RecipeData) -> Optional[str]:
    """Position with the largest repeat Max-Min range — matches the illustration
    figure's auto-pick, so a 'Auto(최대 편차)' readout describes the same position shown."""
    best, best_pos = -1.0, None
    for pos in POSITION_LABELS:
        stack, _ = _position_stack(recipe, pos)
        if stack is None:
            continue
        rng = float((stack.max(axis=0) - stack.min(axis=0)).max())
        if rng > best:
            best, best_pos = rng, pos
    return best_pos


def compute_outlier_wafer_metric(recipe: RecipeData, mode: str = "percentile",
                                 value: float = 1.0) -> dict:
    """Per-position Rep.Max reduction from outlier exclusion (전체 - 제외후, nm).

    Highlights where transient outliers dominate. Excluded-pixel COUNT is flat
    under percentile mode (every position drops the same fraction), so the Rep.Max
    delta is the meaningful per-position severity for the 3x3 outlier wafer map."""
    out = {}
    for pos in POSITION_LABELS:
        stack, _ = _position_stack(recipe, pos)
        if stack is None:
            continue
        pr = stack.max(axis=0) - stack.min(axis=0)
        valid = _exclude_outlier_pixels(stack, mode, value)
        before = float(pr.max())
        after = float(pr[valid].max()) if valid.any() else 0.0
        out[pos] = max(0.0, before - after)
    return out


__all__ = ["PositionDiagnosis", "Guide", "diagnose_position", "worst_position",
           "compute_outlier_wafer_metric", "dominant_spatial_period",
           "LOW", "MID", "HIGH"]
