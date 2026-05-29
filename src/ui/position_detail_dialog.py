"""Position Detail Dialog — interactive deep-dive into a single position.

Double-click any subplot in the 3×3 Profile Charts grid to open this dialog.
Uses pyqtgraph for scroll-zoom, drag-pan, and real-time crosshair.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QFont, QColor, QPen
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QHeaderView, QSizePolicy,
)

from ..core.data_loader import RecipeData
from ..core.analyzer import AnalysisResult


# ── pyqtgraph global dark theme ──────────────────────────────
pg.setConfigOptions(antialias=False, background="#1e1e2e", foreground="#cdd6f4",
                    useOpenGL=True)

# Colors
_COLORS = {
    "bg": "#1e1e2e",
    "fg": "#cdd6f4",
    "grid": "#45475a",
    "accent": "#89b4fa",
    "mean": "#f5c2e7",
    "sigma": "#cba6f7",
    "overlay": [
        "#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8",
        "#fab387", "#cba6f7", "#f5c2e7", "#89dceb",
        "#b4befe", "#f5e0dc", "#eba0ac", "#a6d189",
        "#e78284", "#ef9f76", "#81c8be", "#ca9ee6",
    ],
}

DARK_STYLE = """
QDialog { background-color: #1e1e2e; }
QLabel { color: #cdd6f4; }
QCheckBox { color: #cdd6f4; font-size: 12px; spacing: 4px; }
QCheckBox::indicator { width: 14px; height: 14px; }
QCheckBox::indicator:checked {
    background-color: #89b4fa; border: 2px solid #b4befe; border-radius: 3px; }
QCheckBox::indicator:unchecked {
    background-color: #313244; border: 2px solid #585b70; border-radius: 3px; }
QGroupBox {
    border: 1px solid #45475a; border-radius: 6px;
    margin-top: 8px; padding-top: 16px;
    font-weight: bold; color: #89b4fa; font-size: 12px;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QTableWidget {
    background-color: #181825; color: #cdd6f4;
    gridline-color: #313244; border: 1px solid #45475a; font-size: 12px;
}
QTableWidget::item { padding: 4px; }
QHeaderView::section {
    background-color: #313244; color: #89b4fa;
    padding: 6px; border: 1px solid #45475a; font-weight: bold; font-size: 12px;
}
"""


class PositionDetailDialog(QDialog):
    """Interactive detail view for a single wafer position (pyqtgraph)."""

    def __init__(self, position: str, recipe: RecipeData,
                 result: Optional[AnalysisResult] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.position = position
        self.recipe = recipe
        self.result = result

        # Collect profiles for this position
        self.profiles: list[tuple[int, np.ndarray, np.ndarray, float]] = []
        for repeat in recipe.repeats:
            if position in repeat.profiles:
                prof = repeat.profiles[position]
                self.profiles.append((
                    repeat.repeat_no,
                    prof.x_mm.astype(np.float64),
                    prof.z_nm.astype(np.float64),
                    float(prof.opm_nm),
                ))

        self.setWindowTitle(f"Position Detail — {position}")
        self.setMinimumSize(1000, 650)
        self.resize(1150, 720)
        self.setStyleSheet(DARK_STYLE)

        self._curve_items: list[pg.PlotDataItem] = []
        self._mean_item: Optional[pg.PlotDataItem] = None
        self._sigma_fill: Optional[pg.FillBetweenItem] = None
        # Segment measurement state (caliper + band)
        self._measure_mode = "off"          # "off" | "caliper" | "band"
        self._caliper_pts: list[float] = []     # clicked x values (max 2)
        self._caliper_dots: list[tuple] = []    # (x, z) of placed dots
        self._measure_items: list = []          # pg items to remove on clear
        self._band_item = None                  # pg.LinearRegionItem
        self._active_window = None              # (x_a, x_b) of current measurement
        self._setup_ui()
        self._build_curves()
        self._update_visibility()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ─── Left: pyqtgraph PlotWidget ───
        chart_container = QVBoxLayout()
        chart_container.setSpacing(2)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_widget.setLabel("bottom", "mm", **{"font-size": "11px"})
        self.plot_widget.setLabel("left", "nm", **{"font-size": "11px"})
        self.plot_widget.setTitle(
            f"{self.position} — {self.recipe.range_label}",
            size="14px", color=_COLORS["fg"])
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.getPlotItem().getViewBox().setMouseMode(
            pg.ViewBox.RectMode)  # default: drag to zoom box

        # Legend
        self.legend = self.plot_widget.addLegend(
            offset=(10, 10), labelTextSize="9px",
            brush=pg.mkBrush("#313244CC"), pen=pg.mkPen("#45475a"))

        # Crosshair
        self.vline = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen("#585b70", width=1, style=Qt.DashLine))
        self.hline = pg.InfiniteLine(angle=0, movable=False,
                                     pen=pg.mkPen("#585b70", width=1, style=Qt.DashLine))
        self.plot_widget.addItem(self.vline, ignoreBounds=True)
        self.plot_widget.addItem(self.hline, ignoreBounds=True)
        self.crosshair_label = QLabel("X: — mm  Y: — nm")
        self.crosshair_label.setStyleSheet(
            "font-size: 11px; color: #a6adc8; padding: 2px 6px;"
            "background-color: #181825; border-radius: 3px;")
        self.crosshair_label.setFixedHeight(22)
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_clicked)

        chart_container.addWidget(self.plot_widget)
        chart_container.addWidget(self.crosshair_label)
        main_layout.addLayout(chart_container, 7)

        # ─── Right: Controls + Stats ───
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        # Title
        title = QLabel(f"{self.position} — {self.recipe.range_label}")
        title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #89b4fa; padding: 4px;")
        right_panel.addWidget(title)

        subtitle = QLabel(f"{len(self.profiles)} Repeats")
        subtitle.setStyleSheet("font-size: 12px; color: #a6adc8; padding: 0 4px;")
        right_panel.addWidget(subtitle)

        # Interaction hint
        hint = QLabel("Scroll: Zoom  |  Drag: Pan  |  Right-click: Menu")
        hint.setStyleSheet(
            "font-size: 10px; color: #585b70; padding: 2px 4px; font-style: italic;")
        right_panel.addWidget(hint)

        # ─── Repeat Checkboxes ───
        repeat_group = QGroupBox("Repeat Toggle")
        repeat_layout = QGridLayout(repeat_group)
        repeat_layout.setSpacing(4)

        self.repeat_checkboxes: list[QCheckBox] = []
        for i, (rep_no, _, _, opm) in enumerate(self.profiles):
            cb = QCheckBox(f"R{rep_no}")
            cb.setChecked(True)
            color = _COLORS["overlay"][i % len(_COLORS["overlay"])]
            cb.setStyleSheet(
                f"QCheckBox {{ color: {color}; font-weight: bold; }}"
                f"QCheckBox::indicator:checked {{ background-color: {color};"
                f"border: 2px solid {color}; border-radius: 3px; }}")
            cb.toggled.connect(self._update_visibility)
            repeat_layout.addWidget(cb, i // 3, i % 3)
            self.repeat_checkboxes.append(cb)

        right_panel.addWidget(repeat_group)

        # ─── Overlay Options ───
        overlay_group = QGroupBox("Overlay Options")
        overlay_layout = QVBoxLayout(overlay_group)

        self.mean_cb = QCheckBox("Mean Profile")
        self.mean_cb.setChecked(True)
        self.mean_cb.setStyleSheet(
            f"QCheckBox {{ color: {_COLORS['mean']}; font-weight: bold; }}")
        self.mean_cb.toggled.connect(self._update_visibility)
        overlay_layout.addWidget(self.mean_cb)

        self.sigma_cb = QCheckBox("±1σ Band")
        self.sigma_cb.setChecked(False)
        self.sigma_cb.setStyleSheet(
            f"QCheckBox {{ color: {_COLORS['sigma']}; font-weight: bold; }}")
        self.sigma_cb.toggled.connect(self._update_visibility)
        overlay_layout.addWidget(self.sigma_cb)

        right_panel.addWidget(overlay_group)

        # ─── Measure Panel ───
        self._setup_measure_ui(right_panel)

        # ─── Stats Panel ───
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)

        # Per-repeat OPM table
        self.opm_table = QTableWidget()
        self.opm_table.setColumnCount(2)
        self.opm_table.setHorizontalHeaderLabels(["Repeat", "OPM (nm)"])
        self.opm_table.setRowCount(len(self.profiles))
        for i, (rep_no, _, _, opm) in enumerate(self.profiles):
            r_item = QTableWidgetItem(f"R{rep_no}")
            r_item.setTextAlignment(Qt.AlignCenter)
            self.opm_table.setItem(i, 0, r_item)
            v_item = QTableWidgetItem(f"{opm:.3f}")
            v_item.setTextAlignment(Qt.AlignCenter)
            self.opm_table.setItem(i, 1, v_item)
        self.opm_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.opm_table.verticalHeader().setVisible(False)
        self.opm_table.setMaximumHeight(min(30 * len(self.profiles) + 30, 200))
        stats_layout.addWidget(self.opm_table)

        # Summary
        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.summary_label.setStyleSheet(
            "font-size: 12px; color: #cdd6f4; padding: 6px;"
            "background-color: #181825; border-radius: 4px;")
        self.summary_label.setWordWrap(True)
        self._update_summary()
        stats_layout.addWidget(self.summary_label)

        right_panel.addWidget(stats_group, 1)
        main_layout.addLayout(right_panel, 3)

    # ─── Crosshair ────────────────────────────────────────────

    def _on_mouse_moved(self, pos):
        """Update crosshair and coordinate label."""
        vb = self.plot_widget.getPlotItem().vb
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            self.vline.setPos(x)
            self.hline.setPos(y)
            self.crosshair_label.setText(f"X: {x:.3f} mm   Y: {y:.3f} nm")

    # ─── Stats ────────────────────────────────────────────────

    def _update_summary(self):
        if not self.profiles:
            self.summary_label.setText("No data")
            return

        opm_values = np.array([p[3] for p in self.profiles])
        z_arrays = [p[2] for p in self.profiles]

        if len(z_arrays) >= 2:
            stack = np.array(z_arrays, dtype=np.float64)
            pixel_range = stack.max(axis=0) - stack.min(axis=0)
            rep_max = float(pixel_range.max())
            rep_1sigma = float(pixel_range.std(ddof=0))
        else:
            rep_max = rep_1sigma = 0.0

        mean_z = np.mean(z_arrays, axis=0) if z_arrays else np.array([])
        mean_val = float(mean_z.mean()) if len(mean_z) > 0 else 0.0
        std_val = float(np.std([z.mean() for z in z_arrays])) if z_arrays else 0.0

        self.summary_label.setText(
            f"<b>Rep. Max:</b> {rep_max:.3f} nm<br>"
            f"<b>Rep. 1\u03c3:</b> {rep_1sigma:.3f} nm<br>"
            f"<b>OPM Max:</b> {float(opm_values.max()):.3f} nm<br>"
            f"<b>OPM Mean:</b> {float(opm_values.mean()):.3f} nm<br>"
            f"<b>OPM Stdev:</b> {float(opm_values.std(ddof=0)):.3f} nm<br>"
            f"<br>"
            f"<b>Z Mean:</b> {mean_val:.2f} nm<br>"
            f"<b>Z Spread:</b> \u00b1{std_val:.2f} nm"
        )

    # ─── Chart Drawing ────────────────────────────────────────

    @staticmethod
    def _decimate(arr: np.ndarray, max_pts: int = 2000) -> np.ndarray:
        """Downsample array for display performance."""
        if len(arr) <= max_pts:
            return arr
        step = len(arr) // max_pts
        return arr[::step]

    def _build_curves(self):
        """Create all curve items once (called on init only)."""
        self._curve_items = []
        for i, (rep_no, x_mm, z_nm, opm) in enumerate(self.profiles):
            color = _COLORS["overlay"][i % len(_COLORS["overlay"])]
            pen = pg.mkPen(color, width=1.2)
            x_dec = self._decimate(x_mm)
            z_dec = self._decimate(z_nm)
            item = pg.PlotDataItem(x_dec, z_dec, pen=pen,
                                   name=f"R{rep_no} (OPM {opm:.1f})")
            # pyqtgraph 0.14.0: add to the plot BEFORE enabling clip/downsampling.
            # Configuring them first makes addItem's view-range cascade query an
            # un-associated ViewBox -> AttributeError: autoRangeEnabled.
            self.plot_widget.addItem(item)
            item.setDownsampling(auto=True, method="peak")
            item.setClipToView(True)
            self._curve_items.append(item)

        # Pre-build mean curve (hidden initially if needed)
        if self.profiles:
            all_z = [p[2] for p in self.profiles]
            x0 = self.profiles[0][1]
            mean_z = np.mean(all_z, axis=0)
            x_dec = self._decimate(x0)
            mean_dec = self._decimate(mean_z)
            pen = pg.mkPen(_COLORS["mean"], width=2.5)
            self._mean_item = pg.PlotDataItem(x_dec, mean_dec, pen=pen, name="Mean")
            self.plot_widget.addItem(self._mean_item)  # add before clip/downsampling (pg 0.14.0)
            self._mean_item.setDownsampling(auto=True, method="peak")
            self._mean_item.setClipToView(True)

            # Pre-build ±1σ band
            if len(all_z) >= 2:
                std_z = np.std(all_z, axis=0, ddof=0)
                upper_dec = self._decimate(mean_z + std_z)
                lower_dec = self._decimate(mean_z - std_z)
                self._sigma_fill = pg.FillBetweenItem(
                    pg.PlotDataItem(x_dec, upper_dec),
                    pg.PlotDataItem(x_dec, lower_dec),
                    brush=pg.mkBrush(QColor(_COLORS["sigma"]).lighter(120).name() + "30"))
                self.plot_widget.addItem(self._sigma_fill)

    def _update_visibility(self):
        """Toggle curve visibility without rebuilding (fast)."""
        visible_indices = []
        for i, item in enumerate(self._curve_items):
            visible = (i < len(self.repeat_checkboxes)
                       and self.repeat_checkboxes[i].isChecked())
            item.setVisible(visible)
            if visible:
                visible_indices.append(i)

        # Update mean + sigma with visible-only data
        show_mean = self.mean_cb.isChecked() and visible_indices
        if self._mean_item is not None:
            if show_mean:
                visible_z = [self.profiles[i][2] for i in visible_indices]
                x0 = self.profiles[0][1]
                mean_z = np.mean(visible_z, axis=0)
                x_dec = self._decimate(x0)
                self._mean_item.setData(x_dec, self._decimate(mean_z))
                self._mean_item.setVisible(True)
            else:
                self._mean_item.setVisible(False)

        show_sigma = (self.sigma_cb.isChecked() and show_mean
                      and len(visible_indices) >= 2)
        if self._sigma_fill is not None:
            if show_sigma:
                visible_z = [self.profiles[i][2] for i in visible_indices]
                x0 = self.profiles[0][1]
                mean_z = np.mean(visible_z, axis=0)
                std_z = np.std(visible_z, axis=0, ddof=0)
                x_dec = self._decimate(x0)
                self._sigma_fill.setCurves(
                    pg.PlotDataItem(x_dec, self._decimate(mean_z + std_z)),
                    pg.PlotDataItem(x_dec, self._decimate(mean_z - std_z)))
                self._sigma_fill.setVisible(True)
            else:
                self._sigma_fill.setVisible(False)

        # Refresh active measurement when the visible repeat set changes
        if getattr(self, "_active_window", None) is not None:
            self._recompute(*self._active_window)

    # ─── Segment Measurement (Caliper + Band) ─────────────────

    def _setup_measure_ui(self, parent_layout):
        """Build the Measure group: mode toggles + Clear + readout table."""
        measure_group = QGroupBox("Measure (Segment OPM)")
        mlay = QVBoxLayout(measure_group)
        mlay.setSpacing(6)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.caliper_btn = QPushButton("\U0001f4cf Caliper")
        self.caliper_btn.setCheckable(True)
        self.caliper_btn.clicked.connect(
            lambda: self._set_measure_mode(
                "caliper" if self.caliper_btn.isChecked() else "off"))
        self.band_btn = QPushButton("▭ Band")
        self.band_btn.setCheckable(True)
        self.band_btn.clicked.connect(
            lambda: self._set_measure_mode(
                "band" if self.band_btn.isChecked() else "off"))
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self._set_measure_mode("off"))
        for b in (self.caliper_btn, self.band_btn, clear_btn):
            b.setStyleSheet(
                "QPushButton { background-color: #313244; color: #cdd6f4;"
                " border: 1px solid #45475a; border-radius: 4px; padding: 4px 8px; }"
                "QPushButton:checked { background-color: #89b4fa; color: #11111b;"
                " font-weight: bold; }")
            btn_row.addWidget(b)
        mlay.addLayout(btn_row)

        self.measure_header = QLabel("두 점을 클릭하거나 밴드를 드래그하세요.")
        self.measure_header.setWordWrap(True)
        self.measure_header.setStyleSheet(
            "font-size: 11px; color: #a6adc8; padding: 2px 4px;")
        mlay.addWidget(self.measure_header)

        self.measure_table = QTableWidget()
        self.measure_table.setColumnCount(2)
        self.measure_table.setHorizontalHeaderLabels(["Repeat", "구간 OPM (nm)"])
        self.measure_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.measure_table.verticalHeader().setVisible(False)
        self.measure_table.setMaximumHeight(min(28 * len(self.profiles) + 28, 180))
        mlay.addWidget(self.measure_table)

        self.measure_across = QLabel("")
        self.measure_across.setStyleSheet(
            "font-size: 12px; color: #f9e2af; font-weight: bold; padding: 2px 4px;")
        mlay.addWidget(self.measure_across)

        parent_layout.addWidget(measure_group)

    def _set_measure_mode(self, mode):
        """Switch measurement mode: 'off' | 'caliper' | 'band'."""
        self._clear_measurement()
        self._measure_mode = mode
        self.caliper_btn.setChecked(mode == "caliper")
        self.band_btn.setChecked(mode == "band")

        vb = self.plot_widget.getPlotItem().getViewBox()
        if mode == "caliper":
            vb.setMouseMode(pg.ViewBox.PanMode)
            self.measure_header.setText(
                "곡선 위 두 점을 클릭하세요 (3번째 클릭 = 리셋).")
        elif mode == "band":
            vb.setMouseMode(pg.ViewBox.RectMode)
            self._add_band()
            self.measure_header.setText("밴드 경계를 드래그해 구간을 조절하세요.")
        else:
            vb.setMouseMode(pg.ViewBox.RectMode)
            self.measure_header.setText("두 점을 클릭하거나 밴드를 드래그하세요.")

    def _add_band(self):
        """Add a draggable LinearRegionItem at the central third of the X view."""
        vb = self.plot_widget.getPlotItem().getViewBox()
        (x0, x1), _ = vb.viewRange()
        span = x1 - x0
        a, b = x0 + span / 3.0, x1 - span / 3.0
        self._band_item = pg.LinearRegionItem(
            values=(a, b), brush=pg.mkBrush("#89b4fa22"),
            pen=pg.mkPen("#89b4fa", width=1))
        self._band_item.setZValue(-5)
        self.plot_widget.addItem(self._band_item)
        self._measure_items.append(self._band_item)
        self._band_item.sigRegionChanged.connect(self._on_band_changed)
        self._recompute(a, b)

    def _on_band_changed(self):
        if self._band_item is None:
            return
        a, b = self._band_item.getRegion()
        self._recompute(a, b)

    def _on_plot_clicked(self, ev):
        """Place caliper points on left-click (caliper mode only)."""
        if self._measure_mode != "caliper":
            return
        if ev.button() != Qt.LeftButton:
            return
        if not self.plot_widget.sceneBoundingRect().contains(ev.scenePos()):
            return
        vb = self.plot_widget.getPlotItem().getViewBox()
        pt = vb.mapSceneToView(ev.scenePos())
        snap = self._snap_x(pt.x(), pt.y())
        if snap is None:
            return
        _, sx, sz = snap
        self._place_caliper_point(sx, sz)

    def _place_caliper_point(self, sx, sz):
        """Add a caliper dot/marker; on the 2nd point draw the segment + readout."""
        if len(self._caliper_pts) >= 2:   # 3rd point resets
            self._clear_measurement()

        self._caliper_pts.append(sx)
        self._caliper_dots.append((sx, sz))
        dot = pg.ScatterPlotItem([sx], [sz], size=10,
                                 brush=pg.mkBrush("#f38ba8"), pen=pg.mkPen("#11111b"))
        vline = pg.InfiniteLine(pos=sx, angle=90, movable=False,
                                pen=pg.mkPen("#f38ba8", width=1, style=Qt.DashLine))
        self.plot_widget.addItem(dot)
        self.plot_widget.addItem(vline, ignoreBounds=True)
        self._measure_items += [dot, vline]

        if len(self._caliper_pts) == 2:
            (ax, az), (bx, bz) = self._caliper_dots
            # plain 2-point PlotDataItem; NO clip/downsampling (pg 0.14.0 crash guard)
            line = pg.PlotDataItem([ax, bx], [az, bz],
                                   pen=pg.mkPen("#f38ba8", width=1.5, style=Qt.DashLine))
            self.plot_widget.addItem(line)
            region = pg.LinearRegionItem(values=sorted((ax, bx)), movable=False,
                                         brush=pg.mkBrush("#f38ba822"), pen=pg.mkPen(None))
            region.setZValue(-10)
            self.plot_widget.addItem(region)
            self._measure_items += [line, region]
            self._recompute(ax, bx)

    def _snap_x(self, x_view, y_view):
        """Snap to the nearest sample of the visible curve closest to the click.

        Returns (index, x, z) or None when no repeat is visible.
        """
        best = None  # (vertical_dist, idx, x, z)
        for i, (_rep_no, x_mm, z_nm, _opm) in enumerate(self.profiles):
            if not self.repeat_checkboxes[i].isChecked():
                continue
            if len(x_mm) == 0:
                continue
            idx = int(np.argmin(np.abs(x_mm - x_view)))
            xx, zz = float(x_mm[idx]), float(z_nm[idx])
            d = abs(zz - y_view)
            if best is None or d < best[0]:
                best = (d, idx, xx, zz)
        if best is None:
            return None
        return best[1], best[2], best[3]

    def _segment_stats(self, x_a, x_b):
        """Per-repeat segment OPM (max-min within [x_a, x_b]) over visible repeats."""
        lo, hi = sorted((float(x_a), float(x_b)))
        rows = []
        gmax, gmin = -np.inf, np.inf
        for i, (rep_no, x_mm, z_nm, _opm) in enumerate(self.profiles):
            if not self.repeat_checkboxes[i].isChecked():
                continue
            m = (x_mm >= lo) & (x_mm <= hi)
            if not m.any():
                continue
            zw = z_nm[m]
            rows.append((rep_no, float(zw.max() - zw.min())))
            gmax = max(gmax, float(zw.max()))
            gmin = min(gmin, float(zw.min()))
        across = (gmax - gmin) if rows else 0.0
        return rows, across, (hi - lo)

    def _recompute(self, x_a, x_b):
        """Compute segment stats for [x_a, x_b] and refresh the readout."""
        rows, across, width = self._segment_stats(x_a, x_b)
        lo, hi = sorted((float(x_a), float(x_b)))

        if self._measure_mode == "caliper" and len(self._caliper_dots) == 2:
            (ax, az), (bx, bz) = self._caliper_dots
            self.measure_header.setText(
                f"A: x={ax:.3f} z={az:.2f}  |  B: x={bx:.3f} z={bz:.2f}  |  "
                f"ΔX={abs(bx - ax):.3f} mm  ΔZ={abs(bz - az):.2f} nm")
        else:
            self.measure_header.setText(
                f"구간: [{lo:.3f}, {hi:.3f}] mm  |  폭 ΔX={width:.3f} mm")

        self.measure_table.setRowCount(len(rows))
        for r, (rep_no, seg) in enumerate(rows):
            a_item = QTableWidgetItem(f"R{rep_no}")
            a_item.setTextAlignment(Qt.AlignCenter)
            b_item = QTableWidgetItem(f"{seg:.3f}")
            b_item.setTextAlignment(Qt.AlignCenter)
            self.measure_table.setItem(r, 0, a_item)
            self.measure_table.setItem(r, 1, b_item)

        self.measure_across.setText(
            f"Across (envelope) max−min: {across:.3f} nm" if rows
            else "구간 내 데이터 없음")
        self._active_window = (float(x_a), float(x_b))

    def _clear_measurement(self):
        """Remove all measurement items and reset the readout."""
        for it in self._measure_items:
            try:
                self.plot_widget.removeItem(it)
            except Exception:
                pass
        self._measure_items = []
        self._band_item = None
        self._caliper_pts = []
        self._caliper_dots = []
        self._active_window = None
        if hasattr(self, "measure_table"):
            self.measure_table.setRowCount(0)
        if hasattr(self, "measure_across"):
            self.measure_across.setText("")
