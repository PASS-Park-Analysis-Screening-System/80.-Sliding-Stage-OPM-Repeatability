"""Visualization manager using Matplotlib.

Generates all chart types needed for OPM Repeatability analysis:
- Profile Overlay Charts (9-position, N-repeat overlays)
- Flatten Preview (original + regression + flattened + histogram)
- Saturation Trend Charts
- Wafer Map Heatmap
- Best-5 Window comparison
"""
from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for embedding in Qt
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 0.1
matplotlib.rcParams["agg.path.chunksize"] = 10000
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.font_manager import FontProperties, findfont
import numpy as np

from ..core.data_loader import RecipeData, POSITION_LABELS, POSITION_GRID
from ..core.analyzer import (AnalysisResult, POSITION_GROUPS, resample_profile,
                             _exclude_outlier_pixels)
from ..core.flatten import FlattenResult, polynomial_flatten, edge_only_flatten

# --- Color Scheme ---
COLORS = {
    "bg": "#1e1e2e",
    "fg": "#cdd6f4",
    "grid": "#45475a",
    "accent": "#89b4fa",
    "accent2": "#74c7ec",
    "accent3": "#94e2d5",
    "green": "#a6e3a1",
    "red": "#f38ba8",
    "yellow": "#f9e2af",
    "overlay": [
        "#89b4fa", "#74c7ec", "#94e2d5", "#a6e3a1", "#f9e2af",
        "#fab387", "#f38ba8", "#cba6f7", "#f5c2e7", "#89dceb",
        "#b4befe", "#f5e0dc", "#eba0ac", "#a6d189", "#e78284",
        "#ef9f76", "#81c8be", "#ca9ee6", "#e5c890", "#babbf1",
    ],
}


def _apply_dark_theme(ax, fig=None):
    """Apply dark theme to axes and figure."""
    if fig:
        fig.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["fg"], labelsize=8)
    ax.xaxis.label.set_color(COLORS["fg"])
    ax.yaxis.label.set_color(COLORS["fg"])
    ax.title.set_color(COLORS["fg"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], alpha=0.3, linewidth=0.5)


def _korean_fontproperties() -> FontProperties:
    """A CJK-capable FontProperties (Malgun Gothic on Windows) so Korean labels
    don't render as tofu; falls back to default if none is available."""
    for name in ("Malgun Gothic", "Gulim", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"):
        try:
            if findfont(name, fallback_to_default=False):
                return FontProperties(family=name)
        except Exception:
            continue
    return FontProperties()


_KFONT = _korean_fontproperties()


def _shade_columns(ax, mask: np.ndarray, color: str, alpha: float = 0.18) -> None:
    """Shade contiguous runs of True columns (e.g. excluded pixels) on ax."""
    if not mask.any():
        return
    idx = np.where(mask)[0]
    start = prev = idx[0]
    for k in idx[1:]:
        if k == prev + 1:
            prev = k
            continue
        ax.axvspan(start - 0.5, prev + 0.5, color=color, alpha=alpha, lw=0)
        start = prev = k
    ax.axvspan(start - 0.5, prev + 0.5, color=color, alpha=alpha, lw=0)


def create_outlier_illustration_figure(recipe, mode: str = "percentile",
                                       value: float = 1.0, position: str | None = None,
                                       figsize: tuple = (8.6, 6.0),
                                       show_mm: bool = False) -> Figure:
    """Educational 2-panel figure showing which pixels outlier-exclusion drops.

    Top: every repeat profile for a position overlaid, with the excluded (high
    repeat-deviation) pixel columns shaded red. Bottom: the per-pixel repeat range
    (Max-Min) with the exclusion threshold; pixels above it are the dropped ones.
    The most illustrative position (largest pixel range) is auto-picked when not
    given. Falls back to a synthetic spike example if no data is available.
    """
    fig = Figure(figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # --- Build a (repeats, pixels) stack from real data, else synthetic ---
    stack = None
    title_pos = position
    if recipe is not None and getattr(recipe, "repeats", None):
        best = None
        for pos in POSITION_LABELS:
            profs = [edge_only_flatten(r.profiles[pos].z_nm, order=1, edge_percent=1.0)
                     for r in recipe.repeats if pos in r.profiles]
            if len(profs) < 2:
                continue
            s = np.asarray(profs, dtype=np.float64)
            if position is not None and pos == position:
                stack, title_pos = s, pos
                break
            rng = float((s.max(axis=0) - s.min(axis=0)).max())
            if best is None or rng > best[0]:
                best = (rng, pos, s)
        if stack is None and best is not None:
            stack, title_pos = best[2], best[1]

    synthetic = stack is None
    if synthetic:
        n = 400
        x = np.linspace(0.0, 1.0, n)
        base = 20.0 * (x - 0.5) ** 2
        rng_state = np.random.RandomState(0)
        rows = np.array([base + rng_state.normal(0, 0.3, n) for _ in range(5)])
        rows[2, 190:210] += 8.0   # a localized spike in one repeat
        stack, title_pos = rows, "예시 데이터"
        mode, value = "percentile", 5.0

    n_rep, n_px = stack.shape
    # Physical stage-travel axis (mm) for the chosen position, so the user can read
    # WHERE an outlier sits on the ball-screw travel, not just a pixel index.
    x_mm_sel = None
    if not synthetic and recipe is not None:
        for r in recipe.repeats:
            if title_pos in r.profiles:
                x_mm_sel = np.asarray(r.profiles[title_pos].x_mm, dtype=float)
                break
    px = np.arange(n_px)
    pixel_range = stack.max(axis=0) - stack.min(axis=0)
    valid = _exclude_outlier_pixels(stack, mode, value)
    excluded = ~valid

    # --- Top: overlaid repeats + excluded columns shaded red ---
    _apply_dark_theme(ax1, fig)
    for i in range(n_rep):
        ax1.plot(px, stack[i], color=COLORS["overlay"][i % len(COLORS["overlay"])],
                 lw=0.8, alpha=0.85)
    _shade_columns(ax1, excluded, COLORS["red"])
    ax1.set_title(f"반복 프로파일 오버레이 — {title_pos}   (빨강 = 제외 픽셀 {int(excluded.sum())}개)",
                  fontsize=10, fontweight="bold", fontproperties=_KFONT)
    ax1.set_ylabel("Height (nm)", fontproperties=_KFONT)

    # --- Bottom: per-pixel repeat range + threshold + excluded markers ---
    _apply_dark_theme(ax2, None)
    ax2.plot(px, pixel_range, color=COLORS["accent"], lw=0.9,
             label="픽셀별 반복 편차 (Max-Min)")
    if mode == "percentile" and value > 0:
        thr = float(np.percentile(pixel_range, 100 - value))
        ax2.axhline(thr, color=COLORS["yellow"], ls="--", lw=1.0,
                    label=f"임계값 (상위 {value:g}%)")
    if excluded.any():
        ax2.scatter(px[excluded], pixel_range[excluded], color=COLORS["red"],
                    s=12, zorder=5, label="제외 픽셀")
    ax2.set_xlabel("Pixel", fontproperties=_KFONT)
    ax2.set_ylabel("Range (nm)", fontproperties=_KFONT)
    # Optional physical mm axis (stage travel) on top of the range panel.
    if show_mm and x_mm_sel is not None and len(x_mm_sel) == n_px:
        def _p2mm(p):
            return np.interp(p, px, x_mm_sel)

        def _mm2p(m):
            return np.interp(m, x_mm_sel, px)

        secax = ax2.secondary_xaxis("top", functions=(_p2mm, _mm2p))
        secax.set_xlabel("Stage 위치 (mm)", fontproperties=_KFONT, fontsize=8)
        secax.tick_params(colors=COLORS["fg"], labelsize=7)
    leg = ax2.legend(prop=_KFONT, fontsize=7, facecolor=COLORS["bg"],
                     edgecolor=COLORS["grid"], labelcolor=COLORS["fg"])
    if leg:
        leg.get_frame().set_alpha(0.6)

    # Before/after Rep.Max caption (+ physical location of the worst-deviation pixel)
    after = float(pixel_range[valid].max()) if valid.any() else 0.0
    loc = ""
    if x_mm_sel is not None and len(x_mm_sel) == n_px:
        loc = f"   ·   최대 편차 위치 {float(x_mm_sel[int(np.argmax(pixel_range))]):.2f} mm"
    fig.text(0.5, 0.012,
             f"Rep.Max (픽셀 편차 최대): 전체 {pixel_range.max():.2f} → 제외 후 {after:.2f} nm{loc}",
             ha="center", color=COLORS["fg"], fontsize=8.5, fontproperties=_KFONT)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def create_profile_overlay_figure(recipe: RecipeData,
                                   figsize: tuple = (16, 12),
                                   scan_info: dict | None = None,
                                   y_scale_mode: str = "auto",
                                   sim_factor: int = 1) -> Figure:
    """Create 3×3 grid of profile overlay charts (one per position).

    Each subplot shows all repeat profiles overlaid for that position.
    Profiles are Order-1 leveled (line subtracted) so endpoints converge to 0.
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize)

    # Build suptitle with scan info if available
    title = f"Profile Overlay — {recipe.range_label}"
    if scan_info:
        info_parts = [
            f"{scan_info.get('range_label', '')}",
            f"{scan_info.get('pixels', '')}px",
            f"{scan_info.get('resolution_nm', 0):.0f}nm/px",
            f"{scan_info.get('speed', 0):.2g}mm/s",
            f"SP={scan_info.get('set_point', 0):.1f}",
        ]
        title = f"Profile Overlay — {' | '.join(info_parts)}"

    if sim_factor > 1:
        orig_res = scan_info.get("resolution_nm", 0) if scan_info else 0
        sim_res = orig_res * sim_factor
        title += f"  [Simulated: {sim_res:.0f} nm/px, ×{sim_factor}]"

    fig.suptitle(title, color=COLORS["fg"], fontsize=14, fontweight="bold")

    # Display decimation: downsample to ~1500 points for rendering speed
    _MAX_DISPLAY_PTS = 1500

    for pos in POSITION_LABELS:
        row, col = POSITION_GRID[pos]
        ax = axes[row][col]
        _apply_dark_theme(ax, fig if (row == 0 and col == 0) else None)

        profiles_found = False
        for i, repeat in enumerate(recipe.repeats):
            if pos in repeat.profiles:
                prof = repeat.profiles[pos]
                # Apply resampling if simulating lower resolution
                # Display leveling MUST match the metric leveling (analyzer uses
                # edge-only Order-1, fit on the outer 1% each end) so the overlay
                # shapes are consistent with the reported numbers and endpoints
                # converge to ~0 (matching the reference Tool's view).
                if sim_factor > 1:
                    z_rs = resample_profile(prof.z_nm, sim_factor)
                    x_rs = resample_profile(prof.x_mm, sim_factor)
                    z_leveled = edge_only_flatten(z_rs, order=1, edge_percent=1.0)
                    x_plot = x_rs
                else:
                    z_leveled = edge_only_flatten(prof.z_nm, order=1, edge_percent=1.0)
                    x_plot = prof.x_mm
                # Decimate for display (min-max preserving)
                if len(x_plot) > _MAX_DISPLAY_PTS:
                    step = len(x_plot) // _MAX_DISPLAY_PTS
                    x_plot = x_plot[::step]
                    z_leveled = z_leveled[::step]
                color = COLORS["overlay"][i % len(COLORS["overlay"])]
                ax.plot(x_plot, z_leveled, color=color, alpha=0.6,
                        linewidth=0.5, label=f"R{repeat.repeat_no}",
                        rasterized=True)
                profiles_found = True

        ax.set_title(f"{pos} (Repeats: {recipe.repeat_count})",
                     color=COLORS["fg"], fontsize=9)
        ax.set_xlabel("mm", fontsize=7)
        ax.set_ylabel("nm", fontsize=7)

        if not profiles_found:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes,
                    ha="center", va="center", color=COLORS["fg"], fontsize=10)

    if y_scale_mode != "auto":
        _sync_y_axes(axes, y_scale_mode)

    fig.set_facecolor(COLORS["bg"])
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _sync_y_axes(axes, mode: str):
    """Synchronize Y-axis limits: 'unified' (all same) or 'group' (per group)."""
    _PADDING = 1.1

    if mode == "unified":
        max_abs = max(
            (np.max(np.abs(line.get_ydata()))
             for row in axes for ax in row for line in ax.get_lines()
             if len(line.get_ydata()) > 0),
            default=0.0)
        if max_abs > 0:
            lim = max_abs * _PADDING
            for row in axes:
                for ax in row:
                    ax.set_ylim(-lim, lim)

    elif mode == "group":
        for group_positions in POSITION_GROUPS.values():
            group_axes = [axes[POSITION_GRID[p][0]][POSITION_GRID[p][1]]
                          for p in group_positions if p in POSITION_GRID]
            max_abs = max(
                (np.max(np.abs(line.get_ydata()))
                 for ax in group_axes for line in ax.get_lines()
                 if len(line.get_ydata()) > 0),
                default=0.0)
            if max_abs > 0:
                lim = max_abs * _PADDING
                for ax in group_axes:
                    ax.set_ylim(-lim, lim)


def create_flatten_preview_figure(flatten_result: FlattenResult,
                                   x_mm: np.ndarray,
                                   figsize: tuple = (10, 8)) -> Figure:
    """Create XEI-style flatten preview with original, flattened, and histogram.

    Layout:
        Top:    Original profile
        Bottom: Flattened + regression curve + histogram
    """
    fig = plt.figure(figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    fig.suptitle(f"Flatten — Order {flatten_result.order}",
                 color=COLORS["fg"], fontsize=12, fontweight="bold")

    # Top: Original
    ax1 = fig.add_axes([0.08, 0.55, 0.86, 0.38])
    _apply_dark_theme(ax1)
    ax1.plot(x_mm, flatten_result.original, color=COLORS["accent"], linewidth=0.5)
    ax1.set_ylabel("nm", fontsize=9, color=COLORS["fg"])
    ax1.set_title("Original", fontsize=10, color=COLORS["fg"])

    # Bottom left: Flattened + regression
    ax2 = fig.add_axes([0.08, 0.08, 0.58, 0.38])
    _apply_dark_theme(ax2)
    ax2.plot(x_mm, flatten_result.flattened, color=COLORS["accent"],
             linewidth=0.5, alpha=0.7, label="Flattened")
    ax2.plot(x_mm, flatten_result.regression - flatten_result.original.mean(),
             color=COLORS["red"], linewidth=1.2, alpha=0.8, label="Regression")
    ax2.set_xlabel("mm", fontsize=9, color=COLORS["fg"])
    ax2.set_ylabel("nm", fontsize=9, color=COLORS["fg"])
    ax2.set_title("Parameters", fontsize=10, color=COLORS["fg"])
    ax2.legend(fontsize=7, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["fg"])

    # Bottom right: Histogram
    ax3 = fig.add_axes([0.72, 0.08, 0.22, 0.38])
    _apply_dark_theme(ax3)
    ax3.hist(flatten_result.flattened, bins=50, orientation="horizontal",
             color=COLORS["yellow"], alpha=0.7, edgecolor="none")
    ax3.set_xlabel("Count", fontsize=8, color=COLORS["fg"])

    return fig


def create_saturation_trend_figure(result: AnalysisResult,
                                    figsize: tuple = (10, 6)) -> Figure:
    """Show Rep. 1σ Mean trend as repeat count increases."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    _apply_dark_theme(ax, fig)

    if result.all_windows:
        x = [w.start_index + 1 for w in result.all_windows]
        y = [w.mean_rep_1sigma for w in result.all_windows]
        ax.plot(x, y, 'o-', color=COLORS["accent"], linewidth=2, markersize=8)

        # Highlight best window
        if result.best_window:
            bw = result.best_window
            ax.axvline(x=bw.start_index + 1, color=COLORS["green"],
                       linestyle="--", alpha=0.7, label=f"Best: R{bw.repeat_range}")
            ax.scatter([bw.start_index + 1], [bw.mean_rep_1sigma],
                       color=COLORS["green"], s=150, zorder=5, marker="*")

        # Spec line
        if result.spec_limit:
            ax.axhline(y=result.spec_limit, color=COLORS["red"],
                       linestyle=":", linewidth=2, label=f"Spec: {result.spec_limit} nm")

        ax.set_xlabel("Window Start (Repeat #)", fontsize=10, color=COLORS["fg"])
        ax.set_ylabel("Mean Rep. 1σ (nm)", fontsize=10, color=COLORS["fg"])
        ax.set_title(f"Saturation Trend — {result.range_label}",
                     fontsize=12, color=COLORS["fg"], fontweight="bold")
        ax.legend(fontsize=9, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
                  labelcolor=COLORS["fg"])

    fig.tight_layout()
    return fig


def create_wafer_map_figure(result: AnalysisResult,
                             metric: str = "opm_max",
                             figsize: tuple = (8, 7)) -> Figure:
    """Create 3×3 wafer map heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    _apply_dark_theme(ax, fig)

    data = np.full((3, 3), np.nan)
    source = result.best_window.positions if result.best_window else result.all_positions

    for pos, pr in source.items():
        if pos in POSITION_GRID:
            r, c = POSITION_GRID[pos]
            val = getattr(pr, metric, pr.opm_max)
            data[r][c] = val

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="equal",
                   interpolation="nearest")

    # Labels
    for pos in POSITION_LABELS:
        if pos in POSITION_GRID:
            r, c = POSITION_GRID[pos]
            val = data[r][c]
            label = f"{pos}\n{val:.2f}" if not np.isnan(val) else pos
            ax.text(c, r, label, ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Left", "Center", "Right"], color=COLORS["fg"])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Top", "Middle", "Bottom"], color=COLORS["fg"])

    metric_label = metric.replace("_", " ").title()
    ax.set_title(f"Wafer Map — {result.range_label} ({metric_label})",
                 fontsize=12, color=COLORS["fg"], fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=COLORS["fg"])
    cbar.set_label("nm", color=COLORS["fg"])

    fig.tight_layout(pad=2.0)
    return fig


def create_outlier_wafer_map_figure(recipe, mode: str = "percentile",
                                    value: float = 1.0,
                                    figsize: tuple = (8, 7)) -> Figure:
    """3x3 wafer map of per-position outlier severity (Rep.Max 전체 - 제외 후, nm).

    Shows WHERE across the 9-point grid the transient outliers concentrate — a
    spatial pattern that helps separate a localized sample issue from a systematic
    (chuck/stage) one. Mirrors create_wafer_map_figure."""
    from ..core.diagnostics import compute_outlier_wafer_metric

    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    _apply_dark_theme(ax, fig)

    metric = compute_outlier_wafer_metric(recipe, mode, value)
    data = np.full((3, 3), np.nan)
    for pos, val in metric.items():
        if pos in POSITION_GRID:
            r, c = POSITION_GRID[pos]
            data[r][c] = val

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="equal", interpolation="nearest")
    for pos in POSITION_LABELS:
        if pos in POSITION_GRID:
            r, c = POSITION_GRID[pos]
            val = data[r][c]
            label = f"{pos}\n{val:.1f}" if not np.isnan(val) else pos
            ax.text(c, r, label, ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")

    ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Left", "Center", "Right"], color=COLORS["fg"])
    ax.set_yticklabels(["Top", "Middle", "Bottom"], color=COLORS["fg"])
    ax.set_title("이상치 공간 패턴 — Rep.Max 감소량 (전체 - 제외 후)\n"
                 "※ 절대 Rep.Max는 Visualization > Wafer Map 참고",
                 fontsize=10, color=COLORS["fg"], fontweight="bold",
                 fontproperties=_KFONT)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=COLORS["fg"])
    cbar.set_label("nm", color=COLORS["fg"])
    fig.tight_layout(pad=2.0)
    return fig


def create_periodicity_figure(recipe, position: str, lead_mm: float | None = None,
                              figsize: tuple = (8.6, 4.2)) -> Figure:
    """Spatial-frequency spectrum of the per-pixel repeat range for one position.

    A strong peak at the ball-screw lead period is a structural (ball-screw)
    signature. The dominant period is marked; an optional user lead is overlaid."""
    from ..core.diagnostics import dominant_spatial_period

    fig = Figure(figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    ax = fig.add_subplot(1, 1, 1)
    _apply_dark_theme(ax, fig)

    profs, x_mm = [], None
    if recipe is not None and getattr(recipe, "repeats", None):
        for r in recipe.repeats:
            if position in r.profiles:
                profs.append(edge_only_flatten(r.profiles[position].z_nm, order=1,
                                               edge_percent=1.0))
                if x_mm is None:
                    x_mm = np.asarray(r.profiles[position].x_mm, dtype=float)

    if len(profs) < 2 or x_mm is None or len(x_mm) < 16:
        ax.text(0.5, 0.5, "주기성 분석 불가 (반복/데이터 부족)", transform=ax.transAxes,
                ha="center", va="center", color=COLORS["fg"], fontproperties=_KFONT)
        fig.tight_layout()
        return fig

    stack = np.asarray(profs, dtype=np.float64)
    pr = stack.max(axis=0) - stack.min(axis=0)
    dx = float(abs(x_mm[1] - x_mm[0]))
    n = len(pr)
    xx = np.arange(n, dtype=float)
    y = pr - pr.mean()
    y = y - np.polyval(np.polyfit(xx, y, 1), xx)
    spec = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, d=dx)
    band = freqs >= 2.0 / (n * dx)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.nan)

    ax.plot(periods[band], spec[band], color=COLORS["accent"], lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("공간 주기 (mm)", fontproperties=_KFONT)
    ax.set_ylabel("Power", fontproperties=_KFONT)

    per, _pw = dominant_spatial_period(pr, dx)
    if per > 0:
        ax.axvline(per, color=COLORS["yellow"], ls="--", lw=1.0,
                   label=f"지배 주기 {per:.3f} mm")
    if lead_mm and lead_mm > 0:
        ax.axvline(lead_mm, color=COLORS["red"], ls=":", lw=1.3,
                   label=f"입력 리드 {lead_mm:.3f} mm")
    leg = ax.legend(prop=_KFONT, fontsize=8, facecolor=COLORS["bg"],
                    edgecolor=COLORS["grid"], labelcolor=COLORS["fg"])
    if leg:
        leg.get_frame().set_alpha(0.6)
    ax.set_title(f"공간 주기 스펙트럼 — {position}  (반복 편차 Max-Min)",
                 fontproperties=_KFONT, fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def create_best5_comparison_figure(result: AnalysisResult,
                                    figsize: tuple = (12, 6)) -> Figure:
    """Compare Best-5 Window vs All Repeats statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    _apply_dark_theme(ax1, fig)
    _apply_dark_theme(ax2)

    positions = list(result.all_positions.keys())
    x = np.arange(len(positions))
    width = 0.35

    # Left: Rep. Max comparison
    all_rep_max = [result.all_positions[p].rep_max for p in positions]
    ax1.bar(x - width/2, all_rep_max, width, color=COLORS["accent"], alpha=0.7,
            label="All Repeats")

    if result.best_window:
        bw_rep_max = [result.best_window.positions.get(p, result.all_positions[p]).rep_max
                      for p in positions]
        ax1.bar(x + width/2, bw_rep_max, width, color=COLORS["green"], alpha=0.7,
                label=f"Best-5 (R{result.best_window.repeat_range})")

    ax1.set_xlabel("Position", fontsize=10, color=COLORS["fg"])
    ax1.set_ylabel("Rep. Max (nm)", fontsize=10, color=COLORS["fg"])
    ax1.set_title("Rep. Max Comparison", fontsize=11, color=COLORS["fg"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions, fontsize=7)
    ax1.legend(fontsize=8, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["fg"])

    # Right: Rep. 1σ comparison
    all_rep_sigma = [result.all_positions[p].rep_1sigma for p in positions]
    ax2.bar(x - width/2, all_rep_sigma, width, color=COLORS["accent"], alpha=0.7,
            label="All Repeats")

    if result.best_window:
        bw_rep_sigma = [result.best_window.positions.get(p, result.all_positions[p]).rep_1sigma
                        for p in positions]
        ax2.bar(x + width/2, bw_rep_sigma, width, color=COLORS["green"], alpha=0.7,
                label=f"Best-5 (R{result.best_window.repeat_range})")

    if result.spec_limit:
        ax2.axhline(y=result.spec_limit, color=COLORS["red"],
                     linestyle=":", linewidth=2, label=f"Spec: {result.spec_limit} nm")

    ax2.set_xlabel("Position", fontsize=10, color=COLORS["fg"])
    ax2.set_ylabel("Rep. 1σ (nm)", fontsize=10, color=COLORS["fg"])
    ax2.set_title("Rep. 1σ Comparison", fontsize=11, color=COLORS["fg"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions, fontsize=7)
    ax2.legend(fontsize=8, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["fg"])

    fig.suptitle(f"Best-5 Window Analysis — {result.range_label}",
                 fontsize=13, color=COLORS["fg"], fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def create_resolution_comparison_figure(
        norm_data: dict[str, dict[str, dict]],
        figsize: tuple = (14, 7),
        spec_limits: dict[int, float] | None = None) -> Figure:
    """Create Original vs Normalized OPM comparison across ranges.

    Args:
        norm_data: dict[range_label, dict[position, {original_opm, normalized_opm, ...}]]
        figsize: Figure size.
        spec_limits: Optional {range_mm: limit_nm} for Spec lines on normalized chart.

    Returns:
        matplotlib Figure.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.set_facecolor(COLORS["bg"])
    _apply_dark_theme(ax1, fig)
    _apply_dark_theme(ax2)
    for ax in (ax1, ax2):
        ax.grid(axis='y', color=COLORS["grid"], alpha=0.3)

    range_labels = list(norm_data.keys())
    range_colors = [COLORS["accent"], COLORS["accent2"], COLORS["accent3"], COLORS["yellow"]]

    # --- Left: Position-wise bars grouped by range ---
    positions = ["1_LT", "2_CT", "3_RT", "4_LM", "5_CM", "6_RM", "7_LB", "8_CB", "9_RB"]
    n_ranges = len(range_labels)
    x = np.arange(len(positions))
    bar_width = 0.8 / max(n_ranges, 1)

    for i, rlabel in enumerate(range_labels):
        orig_vals = [norm_data[rlabel].get(p, {}).get("original_opm", 0) for p in positions]
        norm_vals = [norm_data[rlabel].get(p, {}).get("normalized_opm", 0) for p in positions]
        offset = (i - n_ranges / 2 + 0.5) * bar_width
        ax1.bar(x + offset, orig_vals, bar_width * 0.9,
                color=range_colors[i % len(range_colors)], alpha=0.7, label=rlabel)

    ax1.set_xlabel("Position", fontsize=10, color=COLORS["fg"])
    ax1.set_ylabel("OPM Max (nm)", fontsize=10, color=COLORS["fg"])
    ax1.set_title("Original OPM Max per Position", fontsize=11, color=COLORS["fg"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(positions, fontsize=7)
    ax1.legend(fontsize=8, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["fg"])

    # --- Right: Normalized bars (same resolution) ---
    for i, rlabel in enumerate(range_labels):
        norm_vals = [norm_data[rlabel].get(p, {}).get("normalized_opm", 0) for p in positions]
        offset = (i - n_ranges / 2 + 0.5) * bar_width
        ax2.bar(x + offset, norm_vals, bar_width * 0.9,
                color=range_colors[i % len(range_colors)], alpha=0.7, label=rlabel)

    ax2.set_xlabel("Position", fontsize=10, color=COLORS["fg"])
    ax2.set_ylabel("OPM Max (nm) — Normalized", fontsize=10, color=COLORS["fg"])
    target_res = max(
        norm_data[rl].get("5_CM", {}).get("original_res", 0)
        for rl in range_labels
    ) if range_labels else 0
    ax2.set_title(f"Normalized OPM Max (target: {target_res:.0f} nm/px)",
                  fontsize=11, color=COLORS["fg"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(positions, fontsize=7)
    ax2.legend(fontsize=8, facecolor=COLORS["bg"], edgecolor=COLORS["grid"],
               labelcolor=COLORS["fg"])

    # --- Spec lines on normalized chart ---
    if spec_limits:
        for i, rlabel in enumerate(range_labels):
            range_mm = int(rlabel.replace("mm", ""))
            if range_mm in spec_limits:
                spec_val = spec_limits[range_mm]
                ax2.axhline(y=spec_val, color=range_colors[i % len(range_colors)],
                            linestyle="--", alpha=0.5, linewidth=1.0)
                ax2.text(len(positions) - 0.5, spec_val,
                         f" Spec {rlabel}: {spec_val:.0f}nm",
                         color=range_colors[i % len(range_colors)],
                         fontsize=7, va="bottom", ha="right")

    # --- Reduction % annotations on normalized chart ---
    for i, rlabel in enumerate(range_labels):
        orig_vals = [norm_data[rlabel].get(p, {}).get("original_opm", 0) for p in positions]
        norm_vals = [norm_data[rlabel].get(p, {}).get("normalized_opm", 0) for p in positions]
        orig_mean = np.mean([v for v in orig_vals if v > 0]) if any(v > 0 for v in orig_vals) else 0
        norm_mean = np.mean([v for v in norm_vals if v > 0]) if any(v > 0 for v in norm_vals) else 0
        if orig_mean > 0:
            reduction = (orig_mean - norm_mean) / orig_mean * 100
            offset = (i - n_ranges / 2 + 0.5) * bar_width
            max_norm = max(norm_vals) if norm_vals else 0
            ax2.text(x[-1] + offset, max_norm * 1.02, f"{reduction:+.0f}%",
                     color=range_colors[i % len(range_colors)],
                     fontsize=7, fontweight="bold", ha="center", va="bottom")

    fig.suptitle("Cross-Range Resolution Comparison",
                 fontsize=14, color=COLORS["fg"], fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# --------------------------------------------------------------------------- #
# Spatial (real stage-XY) views — reference aids, NOT verdicts
# --------------------------------------------------------------------------- #

def _position_xy_mm(recipe) -> dict:
    """Map position label -> (x_mm, y_mm) from the Info-CSV stage coordinates.

    Falls back to a schematic 3×3 layout (mimicking the real ±85mm / 80mm
    spacing) when coordinates are missing or all zero (datasets without an
    Info CSV use the filename fallback with x_um=y_um=0)."""
    xy = {}
    if recipe is not None and getattr(recipe, "repeats", None):
        for p in recipe.repeats[0].valid_points:
            xy[p.position] = (p.x_um / 1000.0, p.y_um / 1000.0)
    coords = list(xy.values())
    degenerate = (not coords) or all(abs(x) < 1e-6 and abs(y) < 1e-6
                                     for x, y in coords)
    if degenerate:
        xy = {pos: (-90.0 + c * 80.0, 85.0 - r * 85.0)
              for pos, (r, c) in POSITION_GRID.items()}
    return xy


def _position_metric(result, metric: str) -> dict:
    """Per-position scalar (best window if available, else all repeats)."""
    src = None
    if result is not None:
        src = result.best_window.positions if result.best_window else result.all_positions
    out = {}
    if src:
        for pos, pr in src.items():
            out[pos] = float(getattr(pr, metric, pr.opm_max))
    return out


_METRIC_LABELS = {"opm_max": "OPM Max", "rep_max": "Rep. Max",
                  "rep_1sigma": "Rep. 1σ", "opm_1sigma": "OPM 1σ"}


def _metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def create_spatial_wafer_map_figure(result, recipe, metric: str = "opm_max",
                                    figsize: tuple = (8, 7)) -> Figure:
    """9 measurement points at their REAL stage XY (mm), colored/sized by metric.

    Projects the per-position metric onto the physical wafer layout so edge-vs-
    center patterns (chuck flatness / edge effects) read at a glance. Reference
    aid — official judgment stays the Spec-panel aggregate."""
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=figsize)
    _apply_dark_theme(ax, fig)

    xy = _position_xy_mm(recipe)
    vals = _position_metric(result, metric)
    xs, ys, cs, labels = [], [], [], []
    for pos in POSITION_LABELS:
        if pos in xy:
            x, y = xy[pos]
            xs.append(x); ys.append(y)
            cs.append(vals.get(pos, np.nan)); labels.append(pos)

    # 300mm wafer outline for edge-proximity context.
    ax.add_patch(Circle((0, 0), 150.0, fill=False, ls="--", lw=1.0,
                        ec=COLORS["grid"], zorder=1))

    finite = [c for c in cs if np.isfinite(c)]
    vmax = max(finite) if finite else 1.0
    sizes = [300 + 700 * (c / vmax) if (np.isfinite(c) and vmax > 0) else 300
             for c in cs]
    sc = ax.scatter(xs, ys, c=cs, cmap="RdYlGn_r", s=sizes, edgecolors="white",
                    linewidths=1.0, zorder=3)
    import matplotlib.patheffects as pe
    _halo = [pe.withStroke(linewidth=2.5, foreground="white")]  # readable on any color
    for x, y, c, lab in zip(xs, ys, cs, labels):
        txt = f"{lab}\n{c:.1f}" if np.isfinite(c) else lab
        ax.annotate(txt, (x, y), ha="center", va="center", color="#11111b",
                    fontsize=8, fontweight="bold", fontproperties=_KFONT, zorder=4,
                    path_effects=_halo)

    ax.set_xlim(-170, 170); ax.set_ylim(-170, 170)
    ax.set_aspect("equal")
    ax.set_xlabel("Stage X (mm)", fontproperties=_KFONT)
    ax.set_ylabel("Stage Y (mm)", fontproperties=_KFONT)
    metric_label = _metric_label(metric)
    rl = result.range_label if result is not None else ""
    ax.set_title(f"Spatial Map — {rl} ({metric_label}, nm)\n실제 스테이지 좌표 · 참고용",
                 fontproperties=_KFONT, fontsize=11, color=COLORS["fg"],
                 fontweight="bold")
    if np.isfinite(cs).any() if len(cs) else False:
        cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.ax.tick_params(colors=COLORS["fg"])
        cbar.set_label("nm", color=COLORS["fg"])
    fig.tight_layout()
    return fig
