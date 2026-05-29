"""OPM Repeatability analyzer.

Calculates OPM, Repeatability statistics, Best-5 Window selection,
and Spec pass/fail judgment.

Key Metrics:
    - OPM:  Max - Min of a single profile (nm)
    - Rep. Max:  Maximum OPM across repeats for a position
    - Rep. 1σ:   Standard deviation of OPM across repeats for a position
    - OPM Max:   Maximum of averaged profile's range across repeats
    - OPM 1σ:    Standard deviation of averaged profile range across repeats
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .data_loader import RecipeData, POSITION_LABELS
from .flatten import polynomial_flatten, edge_only_flatten

# Position group definitions for Edge/Center/Side analysis
POSITION_GROUPS = {
    "Center": ["5_CM"],
    "Side":   ["2_CT", "4_LM", "6_RM", "8_CB"],
    "Edge":   ["1_LT", "3_RT", "7_LB", "9_RB"],
}


# Spec limits (nm) - OPM Repeatability
SPEC_REPEATABILITY = {
    25: 12.9,
    10: 5.6,
    5: 3.3,
    1: 1.6,
}

# Spec limits (nm) - Max OPM
SPEC_MAX_OPM_DW = {   # Double Walled AE
    25: 250.0,
    10: 100.0,
    5: 50.0,
    1: 18.0,
}

SPEC_MAX_OPM_ISO = {   # Isolated AE
    25: 200.0,
    10: 80.0,
    5: 40.0,
    1: 13.0,
}


@dataclass
class PositionResult:
    """Analysis result for a single position across repeats."""
    position: str           # e.g., "1_LT"
    opm_values: np.ndarray  # OPM (nm) per repeat
    rep_max: float          # Max of OPM across repeats
    rep_1sigma: float       # Stdev of OPM across repeats
    opm_max: float          # Max of averaged profile range
    opm_1sigma: float       # Stdev of averaged profile range
    repeat_count: int       # Number of valid repeats


@dataclass
class WindowResult:
    """Result for a single Best-5 window evaluation."""
    start_index: int                    # 0-based start index
    end_index: int                      # 0-based end index (exclusive)
    repeat_range: str                   # e.g., "3-7"
    positions: dict[str, PositionResult]
    mean_rep_1sigma: float              # Mean of Rep. 1σ across positions
    max_rep_max: float                  # Max of Rep. Max across positions
    max_opm_max: float                  # Max of OPM Max across positions


@dataclass
class AnalysisResult:
    """Complete analysis result for a recipe."""
    range_mm: int
    range_label: str
    total_repeats: int
    equipment_type: str  # "iso" or "dw"

    # Per-position results using ALL repeats
    all_positions: dict[str, PositionResult]

    # Best-5 Window result
    best_window: Optional[WindowResult] = None

    # All window evaluations (for trend display)
    all_windows: list[WindowResult] = field(default_factory=list)

    # Summary statistics
    mean_rep_max: float = 0.0
    stdev_rep_max: float = 0.0
    max_rep_max: float = 0.0
    mean_opm_max: float = 0.0

    # Spec judgment — OPM Repeatability (Rep. 1σ)
    spec_limit: Optional[float] = None
    spec_pass: Optional[bool] = None
    spec_value: Optional[float] = None  # actual measured value used for judgment

    # Spec judgment — Max OPM
    spec_opm_limit: Optional[float] = None
    spec_opm_pass: Optional[bool] = None
    spec_opm_value: Optional[float] = None  # actual measured value used for judgment

    @property
    def overall_pass(self) -> Optional[bool]:
        """Both specs must pass for overall PASS."""
        if self.spec_pass is None and self.spec_opm_pass is None:
            return None
        rep_ok = self.spec_pass if self.spec_pass is not None else True
        opm_ok = self.spec_opm_pass if self.spec_opm_pass is not None else True
        return rep_ok and opm_ok

    @property
    def rms_rep_max(self) -> float:
        """RMS of Rep. Max values across positions."""
        vals = [p.rep_max for p in self.all_positions.values()]
        return float(np.sqrt(np.mean(np.array(vals) ** 2)))


def _exclude_outlier_pixels(stack: np.ndarray,
                            mode: str = "none",
                            value: float = 0.0) -> np.ndarray:
    """Return boolean mask (pixels,) where True = valid pixel.

    Outlier pixels are those with large repeat-to-repeat Max-Min range.
    - mode="percentile": exclude top value% of pixels by range
    - mode="pixels": exclude top int(value) pixels by range
    """
    pixel_range = stack.max(axis=0) - stack.min(axis=0)  # (pixels,)
    mask = np.ones(len(pixel_range), dtype=bool)

    if mode == "percentile" and value > 0:
        threshold = np.percentile(pixel_range, 100 - value)
        mask = pixel_range <= threshold
    elif mode == "pixels" and value > 0:
        n_remove = min(int(value), len(pixel_range) - 1)
        if n_remove > 0:
            sorted_indices = np.argsort(pixel_range)
            mask[sorted_indices[-n_remove:]] = False

    return mask


def _compute_position_result(position: str,
                              profiles_flat: list[np.ndarray],
                              outlier_mode: str = "none",
                              outlier_value: float = 0.0) -> PositionResult:
    """Compute 4 metrics for one position across repeats.

    All metrics use the same Order-1 edge-only flattened profiles.

    Metrics (per reference tool algorithm):
        - Rep. Max:  Max of pixel-wise range across repeats (valid pixels only)
        - Rep. 1σ:   RMS of per-pixel repeat stdev (valid pixels only)
        - OPM Max:   Max of per-repeat OPM on valid pixels
        - OPM 1σ:    RMS from zero of all heights at valid pixels across all repeats

    Args:
        profiles_flat: List of Order-1 edge-only flattened Z arrays (one per repeat).
        outlier_mode: "none", "percentile", or "pixels".
        outlier_value: Threshold for outlier exclusion.
    """
    stack = np.array(profiles_flat, dtype=np.float64)  # (repeats, pixels)

    # Outlier pixel exclusion based on repeat-to-repeat range
    valid = _exclude_outlier_pixels(stack, outlier_mode, outlier_value)
    has_valid = valid.any()

    # Rep. Max: max pixel-wise range on valid pixels
    pixel_range = stack.max(axis=0) - stack.min(axis=0)  # (pixels,)
    rep_max = float(pixel_range[valid].max()) if has_valid else 0.0

    # Rep. 1σ: RMS of per-pixel repeat stdev on valid pixels
    if stack.shape[0] >= 2 and has_valid:
        pixel_stds = stack.std(axis=0, ddof=0)  # std across repeats per pixel
        rep_1sigma = float(np.sqrt(np.mean(pixel_stds[valid] ** 2)))
    else:
        rep_1sigma = 0.0

    # OPM per repeat: max-min on valid pixels only
    opm_per_repeat = []
    for i in range(stack.shape[0]):
        if has_valid:
            vals = stack[i, valid]
            opm_per_repeat.append(float(vals.max() - vals.min()))
        else:
            opm_per_repeat.append(0.0)
    opm_arr = np.array(opm_per_repeat, dtype=np.float64)

    # OPM Max
    opm_max = float(opm_arr.max())

    # OPM 1σ: RMS from zero of ALL heights at valid pixels across ALL repeats
    if has_valid:
        all_heights = stack[:, valid].ravel()
        opm_1sigma = float(np.sqrt(np.mean(all_heights ** 2)))
    else:
        opm_1sigma = 0.0

    return PositionResult(
        position=position,
        opm_values=opm_arr,
        rep_max=rep_max,
        rep_1sigma=rep_1sigma,
        opm_max=opm_max,
        opm_1sigma=opm_1sigma,
        repeat_count=len(profiles_flat),
    )


def _evaluate_window(recipe: RecipeData, start: int, count: int,
                     outlier_mode: str = "none",
                     outlier_value: float = 0.0) -> Optional[WindowResult]:
    """Evaluate a single sliding window of consecutive repeats."""
    end = start + count
    if end > len(recipe.repeats):
        return None

    window_repeats = recipe.repeats[start:end]
    positions = {}

    for pos in POSITION_LABELS:
        profiles_flat = []
        for repeat in window_repeats:
            if pos in repeat.profiles:
                z_raw = repeat.profiles[pos].z_nm
                # Order-1 edge-only flatten for all metrics
                profiles_flat.append(edge_only_flatten(z_raw, order=1, edge_percent=1.0))

        if profiles_flat:
            positions[pos] = _compute_position_result(
                pos, profiles_flat, outlier_mode, outlier_value)

    if not positions:
        return None

    mean_rep_1sigma = float(np.mean([p.rep_1sigma for p in positions.values()]))
    max_rep_max = float(max(p.rep_max for p in positions.values()))
    max_opm_max = float(max(p.opm_max for p in positions.values()))

    return WindowResult(
        start_index=start,
        end_index=end,
        repeat_range=f"{start+1}-{end}",
        positions=positions,
        mean_rep_1sigma=mean_rep_1sigma,
        max_rep_max=max_rep_max,
        max_opm_max=max_opm_max,
    )


def analyze_recipe(recipe: RecipeData, window_size: int = 5,
                   equipment_type: str = "iso",
                   outlier_mode: str = "none",
                   outlier_value: float = 0.0) -> AnalysisResult:
    """Perform full OPM Repeatability analysis on a recipe.

    Args:
        recipe: RecipeData with loaded profiles.
        window_size: Number of consecutive repeats for Best-5 window.
        equipment_type: Equipment type - "iso" (Isolated AE / 분리형) or "dw" (Double Walled AE / 일체형).
        outlier_mode: "none", "percentile", or "pixels" for outlier pixel exclusion.
        outlier_value: Threshold for outlier exclusion.

    Returns:
        AnalysisResult with per-position stats, Best-5 window, and spec judgment.
    """
    n_repeats = recipe.repeat_count

    # --- Compute ALL-repeat statistics per position ---
    all_positions = {}
    for pos in POSITION_LABELS:
        profiles_flat = []
        for repeat in recipe.repeats:
            if pos in repeat.profiles:
                z_raw = repeat.profiles[pos].z_nm
                # Order-1 edge-only flatten for all metrics
                profiles_flat.append(edge_only_flatten(z_raw, order=1, edge_percent=1.0))

        if profiles_flat:
            all_positions[pos] = _compute_position_result(
                pos, profiles_flat, outlier_mode, outlier_value)

    # --- Evaluate ALL sliding windows ---
    all_windows = []
    if n_repeats >= window_size:
        for start in range(n_repeats - window_size + 1):
            w = _evaluate_window(recipe, start, window_size,
                                 outlier_mode, outlier_value)
            if w:
                all_windows.append(w)

    # --- Select Best-5 Window (minimum mean Rep. 1σ) ---
    best_window = None
    if all_windows:
        best_window = min(all_windows, key=lambda w: w.mean_rep_1sigma)

    # --- Summary statistics (from all repeats) ---
    rep_maxes = [p.rep_max for p in all_positions.values()]
    opm_maxes = [p.opm_max for p in all_positions.values()]

    mean_rep_max = float(np.mean(rep_maxes)) if rep_maxes else 0.0
    stdev_rep_max = float(np.std(rep_maxes, ddof=0)) if rep_maxes else 0.0
    max_rep_max_val = float(max(rep_maxes)) if rep_maxes else 0.0
    mean_opm_max = float(np.mean(opm_maxes)) if opm_maxes else 0.0

    # --- Spec judgment ---
    range_mm = recipe.range_mm
    source = best_window.positions if best_window else all_positions

    # 1) OPM Repeatability spec (Rep. 1σ)
    spec_limit = SPEC_REPEATABILITY.get(range_mm)
    spec_pass = None
    spec_value = None
    if spec_limit is not None and source:
        if equipment_type == "dw":
            # 일체형: Center(5_CM) Rep. 1σ 값
            center = source.get("5_CM")
            spec_value = center.rep_1sigma if center else None
        else:
            # 분리형: Total RMS of Rep. 1σ across all positions
            sigmas = [p.rep_1sigma for p in source.values()]
            spec_value = float(np.sqrt(np.mean(np.array(sigmas) ** 2))) if sigmas else None
        if spec_value is not None:
            spec_pass = spec_value <= spec_limit

    # 2) Max OPM spec
    opm_spec_table = SPEC_MAX_OPM_DW if equipment_type == "dw" else SPEC_MAX_OPM_ISO
    spec_opm_limit = opm_spec_table.get(range_mm)
    spec_opm_pass = None
    spec_opm_value = None
    if spec_opm_limit is not None and source:
        if equipment_type == "dw":
            # 일체형: Center(5_CM) OPM Max 값
            center = source.get("5_CM")
            spec_opm_value = center.opm_max if center else None
        else:
            # 분리형: Total Max of OPM Max across all positions
            spec_opm_value = float(max(p.opm_max for p in source.values()))
        if spec_opm_value is not None:
            spec_opm_pass = spec_opm_value <= spec_opm_limit

    return AnalysisResult(
        range_mm=range_mm,
        range_label=recipe.range_label,
        total_repeats=n_repeats,
        equipment_type=equipment_type,
        all_positions=all_positions,
        best_window=best_window,
        all_windows=all_windows,
        mean_rep_max=mean_rep_max,
        stdev_rep_max=stdev_rep_max,
        max_rep_max=max_rep_max_val,
        mean_opm_max=mean_opm_max,
        spec_limit=spec_limit,
        spec_pass=spec_pass,
        spec_value=spec_value,
        spec_opm_limit=spec_opm_limit,
        spec_opm_pass=spec_opm_pass,
        spec_opm_value=spec_opm_value,
    )


def get_summary_table(result: AnalysisResult,
                      use_best_window: bool = True) -> list[dict]:
    """Generate summary table rows (AFP-compatible format).

    Returns list of dicts with: Range, Position, Rep. Max, Rep. 1σ, OPM Max, OPM 1σ.
    """
    source = result.best_window.positions if (use_best_window and result.best_window) else result.all_positions

    rows = []
    for pos in POSITION_LABELS:
        if pos in source:
            p = source[pos]
            rows.append({
                "Range": result.range_label,
                "Position": pos,
                "Rep. Max (nm)": round(p.rep_max, 3),
                "Rep. 1σ (nm)": round(p.rep_1sigma, 3),
                "OPM Max (nm)": round(p.opm_max, 3),
                "OPM 1σ (nm)": round(p.opm_1sigma, 3),
            })

    # Position group rows (Edge / Side / Center)
    if rows:
        for group_name in ["Center", "Side", "Edge"]:
            group_positions = POSITION_GROUPS[group_name]
            group_rows = [r for r in rows if r["Position"] in group_positions]
            if group_rows:
                g_rep_max = [r["Rep. Max (nm)"] for r in group_rows]
                g_rep_sig = [r["Rep. 1σ (nm)"] for r in group_rows]
                g_opm_max = [r["OPM Max (nm)"] for r in group_rows]
                g_opm_sig = [r["OPM 1σ (nm)"] for r in group_rows]
                rows.append({
                    "Range": "Group", "Position": group_name,
                    "Rep. Max (nm)": round(float(np.mean(g_rep_max)), 3),
                    "Rep. 1σ (nm)": round(float(np.mean(g_rep_sig)), 3),
                    "OPM Max (nm)": round(float(max(g_opm_max)), 3),
                    "OPM 1σ (nm)": round(float(np.mean(g_opm_sig)), 3),
                })

    # Total row
    if rows:
        pos_rows = [r for r in rows if r["Range"] != "Group"]
        rep_maxes = [r["Rep. Max (nm)"] for r in pos_rows]
        rep_sigmas = [r["Rep. 1σ (nm)"] for r in pos_rows]
        opm_maxes = [r["OPM Max (nm)"] for r in pos_rows]
        opm_sigmas = [r["OPM 1σ (nm)"] for r in pos_rows]

        rows.append({"Range": "Total", "Position": "Mean",
                      "Rep. Max (nm)": round(float(np.mean(rep_maxes)), 3),
                      "Rep. 1σ (nm)": round(float(np.mean(rep_sigmas)), 3),
                      "OPM Max (nm)": round(float(np.mean(opm_maxes)), 3),
                      "OPM 1σ (nm)": round(float(np.mean(opm_sigmas)), 3)})
        rows.append({"Range": "Total", "Position": "Stdev",
                      "Rep. Max (nm)": round(float(np.std(rep_maxes, ddof=0)), 3),
                      "Rep. 1σ (nm)": round(float(np.std(rep_sigmas, ddof=0)), 3),
                      "OPM Max (nm)": round(float(np.std(opm_maxes, ddof=0)), 3),
                      "OPM 1σ (nm)": round(float(np.std(opm_sigmas, ddof=0)), 3)})
        rows.append({"Range": "Total", "Position": "Max",
                      "Rep. Max (nm)": round(float(max(rep_maxes)), 3),
                      "Rep. 1σ (nm)": "-",
                      "OPM Max (nm)": round(float(max(opm_maxes)), 3),
                      "OPM 1σ (nm)": "-"})
        rms_rep = float(np.sqrt(np.mean(np.array(rep_maxes) ** 2)))
        rms_opm = float(np.sqrt(np.mean(np.array(opm_maxes) ** 2)))
        rows.append({"Range": "Total", "Position": "RMS",
                      "Rep. Max (nm)": round(rms_rep, 3),
                      "Rep. 1σ (nm)": round(float(np.mean(rep_sigmas)), 3),
                      "OPM Max (nm)": round(rms_opm, 3),
                      "OPM 1σ (nm)": round(float(np.mean(opm_sigmas)), 3)})

    return rows


# ---------------------------------------------------------------------------
# Resolution normalization for cross-range comparison
# ---------------------------------------------------------------------------

def resample_profile(z_data: np.ndarray, factor: int) -> np.ndarray:
    """Downsample profile by block averaging.

    Args:
        z_data: 1D profile array.
        factor: Downsample factor (e.g., 25 means average every 25 pixels into 1).

    Returns:
        Downsampled 1D array.
    """
    if factor <= 1:
        return z_data
    n_new = len(z_data) // factor
    return z_data[:n_new * factor].reshape(n_new, factor).mean(axis=1)


def compute_normalized_opm(recipe: RecipeData,
                           target_res_nm: float) -> dict[str, dict]:
    """Compute OPM per position after normalizing to target resolution.

    Args:
        recipe: RecipeData with loaded profiles.
        target_res_nm: Target pixel resolution in nm/pixel.

    Returns:
        dict[position, {"original_opm": float, "normalized_opm": float,
                         "original_res": float, "factor": int}]
    """
    results = {}
    for pos in POSITION_LABELS:
        opm_orig_list = []
        opm_norm_list = []
        original_res = None

        for repeat in recipe.repeats:
            if pos not in repeat.profiles:
                continue
            prof = repeat.profiles[pos]
            px_count = len(prof.raw_data)
            res_nm = prof.scan_size_um * 1000 / px_count if px_count > 0 else 1.0
            if original_res is None:
                original_res = res_nm

            # Original OPM (Order-1 leveled)
            z_lev = polynomial_flatten(prof.z_nm, order=1)
            opm_orig_list.append(float(z_lev.max() - z_lev.min()))

            # Normalized OPM (resample then level)
            factor = max(1, int(round(target_res_nm / res_nm)))
            if factor > 1:
                z_resampled = resample_profile(prof.z_nm, factor)
                z_lev_norm = polynomial_flatten(z_resampled, order=1)
            else:
                z_lev_norm = z_lev
            opm_norm_list.append(float(z_lev_norm.max() - z_lev_norm.min()))

        if opm_orig_list:
            factor = max(1, int(round(target_res_nm / (original_res or 1))))
            results[pos] = {
                "original_opm": float(max(opm_orig_list)),
                "normalized_opm": float(max(opm_norm_list)),
                "original_res": original_res or 0,
                "factor": factor,
            }

    return results
