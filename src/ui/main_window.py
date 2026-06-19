"""PySide6 GUI for Sliding Stage OPM Repeatability Analyzer.

UX Flow:
    1. Click "Open Folder" → select root data folder or single recipe folder
    2. Auto-detect Range → auto-analyze → display results
    3. Switch between recipes via Range selector
    4. Change Signal Source → auto-reload
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QShortcut, QKeySequence, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QTabWidget, QTreeWidget, QTreeWidgetItem,
    QGroupBox, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QProgressBar,
    QFrame, QGridLayout, QScrollArea, QSlider,
)
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from ..core.data_loader import (
    load_recipe, load_dataset, DataSet, RecipeData, POSITION_LABELS,
    POSITION_GRID, _detect_range_mm, find_recipe_directories,
)
from ..core.analyzer import (analyze_recipe, AnalysisResult, get_summary_table,
                             get_dual_summary_table,
                             ROBUST_OUTLIER_MODE, ROBUST_OUTLIER_VALUE)
from ..core.qc_checker import run_qc_checks, QCResult
from ..core.comparator import compare_results, get_compare_table, CompareResult
from ..core.analyzer import compute_normalized_opm
from ..core.flatten import FlattenProcessor
from ..core.time_analysis import extract_recipe_timing, RecipeTiming, format_timing_summary
from ..core.ball_screw_analyzer import (
    analyze_ball_screw, BallScrewAnalysisResult, get_dishing_matrix,
    SPEC_DISHING, POSITION_LABELS as BS_POSITION_LABELS,
)
from ..visualization.plot_manager import (
    create_profile_overlay_figure,
    create_flatten_preview_figure,
    create_saturation_trend_figure,
    create_wafer_map_figure,
    create_best5_comparison_figure,
)
from ..visualization.report_generator import (
    export_summary_csv, export_avg_line_csv, export_all_lines_csv, export_checklist,
    export_ball_screw_csv,
)

# --- Style ---
DARK_STYLE = """
QMainWindow, QWidget { background-color: #1e1e2e; color: #cdd6f4; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 8px;
            padding-top: 14px; font-weight: bold; color: #89b4fa; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
              border-radius: 4px; padding: 6px 16px; font-size: 12px;
              min-width: 60px; min-height: 24px; }
QPushButton:hover { background-color: #45475a; border: 1px solid #89b4fa; }
QPushButton:pressed { background-color: #585b70; }
QPushButton#export_btn { background-color: #1e66f5; color: white; font-weight: bold; }
QPushButton#export_btn:hover { background-color: #2e7fff; }
QPushButton#load_btn { background-color: #40a02b; color: white; font-weight: bold; }
QPushButton#load_btn:hover { background-color: #50c03b; }
QComboBox { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a;
            border-radius: 4px; padding: 4px 8px; min-height: 22px; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background-color: #313244; color: #cdd6f4;
                               selection-background-color: #45475a; }
QSpinBox, QDoubleSpinBox { background-color: #313244; color: #cdd6f4;
                            border: 1px solid #45475a; border-radius: 4px;
                            padding: 3px; min-height: 22px; min-width: 60px; }
QTabWidget::pane { border: 1px solid #45475a; background-color: #1e1e2e; }
QTabBar::tab { background-color: #313244; color: #a6adc8; padding: 8px 16px;
               border: 1px solid #45475a; border-bottom: none; border-radius: 4px 4px 0 0;
               font-size: 12px; }
QTabBar::tab:selected { background-color: #1e1e2e; color: #89b4fa;
                         border-bottom: 2px solid #89b4fa; }
QTabBar::tab:hover { background-color: #45475a; }
QTableWidget { background-color: #181825; color: #cdd6f4; gridline-color: #313244;
               border: 1px solid #45475a; }
QTableWidget::item { padding: 4px; }
QTableWidget::item:selected { background-color: #45475a; }
QHeaderView::section { background-color: #313244; color: #89b4fa; padding: 6px;
                        border: 1px solid #45475a; font-weight: bold; }
QTreeWidget { background-color: #181825; color: #cdd6f4; border: 1px solid #45475a; }
QTreeWidget::item:hover { background-color: #313244; }
QTreeWidget::item:selected { background-color: #45475a; }
QProgressBar { background-color: #313244; border: 1px solid #45475a; border-radius: 4px;
               text-align: center; color: #cdd6f4; }
QProgressBar::chunk { background-color: #89b4fa; border-radius: 3px; }
QStatusBar { background-color: #181825; color: #a6adc8; border-top: 1px solid #313244; }
QScrollBar:vertical { background: #181825; width: 10px; }
QScrollBar::handle:vertical { background: #45475a; border-radius: 5px; min-height: 20px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""


class LoadWorker(QThread):
    """Background worker for loading data."""
    finished_single = Signal(object)
    finished_multi = Signal(object)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, path: str, signal_source: str = "Height",
                 multi: bool = False):
        super().__init__()
        self.path = path
        self.signal_source = signal_source
        self.multi = multi

    def run(self):
        try:
            if self.multi:
                self.progress.emit("Loading all recipes...")
                dataset = load_dataset(self.path, signal_source=self.signal_source)
                self.progress.emit(f"Loaded {len(dataset.recipes)} recipes.")
                self.finished_multi.emit(dataset)
            else:
                self.progress.emit("Loading recipe...")
                recipe = load_recipe(self.path, signal_source=self.signal_source)
                self.progress.emit(f"Loaded {recipe.repeat_count} repeats.")
                self.finished_single.emit(recipe)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


class CopyWorker(QThread):
    """Copy server/network folder to local temp via robocopy (DLP bypass)."""
    finished = Signal(str)   # local path after copy
    error = Signal(str)

    def __init__(self, source: str):
        super().__init__()
        self.source = source

    def run(self):
        try:
            tmp_dir = tempfile.mkdtemp(prefix="opm_")
            dest = os.path.join(tmp_dir, Path(self.source).name)
            result = subprocess.run(
                ["robocopy", self.source, dest, "/E", "/NP", "/NFL", "/NDL"],
                capture_output=True, text=True, timeout=300,
            )
            # robocopy exit codes: 0-7 = success, 8+ = error
            if result.returncode >= 8:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError(
                    f"robocopy failed (exit {result.returncode}): {result.stderr.strip()}"
                )
            self.finished.emit(dest)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


def _detect_folder_type(path: Path, max_depth: int = 3) -> str:
    """Detect folder type: 'root', 'recipe', or 'unknown'.

    Searches up to ``max_depth`` levels below ``path`` for a recipe
    directory (matching ``\\d+mm`` anywhere in the name). This tolerates
    an intermediate folder layer (e.g., server layouts like
    ``.../03. Sliding Stage OPM Repeatability !!/Profile_25mm_Dynamic/...``).
    """
    if _detect_range_mm(path.name) is not None:
        return "recipe"
    if find_recipe_directories(path, max_depth=max_depth):
        return "root"
    return "unknown"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sliding Stage OPM Repeatability Analyzer")
        self.setMinimumSize(1024, 640)

        # State
        self.dataset: Optional[DataSet] = None
        self.current_recipe: Optional[RecipeData] = None
        self.current_result: Optional[AnalysisResult] = None
        self.current_result_robust: Optional[AnalysisResult] = None  # outlier-excluded companion
        self.current_timing: Optional[RecipeTiming] = None
        self.current_bs_result: Optional[BallScrewAnalysisResult] = None
        self.current_qc_result: Optional[QCResult] = None
        self.current_compare_result: Optional[CompareResult] = None
        self.reference_dataset = None  # DataSet for comparison
        self.reference_result: Optional[AnalysisResult] = None
        self.flatten_proc = FlattenProcessor()
        self._worker: Optional[LoadWorker] = None
        self._loaded_path: Optional[str] = None
        self._block_range_signal = False

        # Debounced re-layout of chart canvases on window resize
        self._resize_debounce = QTimer()
        self._resize_debounce.setSingleShot(True)
        self._resize_debounce.setInterval(150)
        self._resize_debounce.timeout.connect(self._relayout_visible_canvas)

        self._setup_ui()
        self.setStyleSheet(DARK_STYLE)
        self._fit_to_screen()

    def _fit_to_screen(self):
        """Open the window at a ratio of the available screen, never exceeding it."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1440, 900)
            return
        avail = screen.availableGeometry()
        min_w, min_h = self.minimumWidth(), self.minimumHeight()
        # Screen too small even for our minimum → start maximized.
        if avail.width() < min_w or avail.height() < min_h:
            self.setWindowState(Qt.WindowMaximized)
            return
        w = max(min(int(avail.width() * 0.85), 1600), min_w)
        h = max(min(int(avail.height() * 0.85), 1000), min_h)
        self.resize(w, h)
        self.move(avail.x() + (avail.width() - w) // 2,
                  avail.y() + (avail.height() - h) // 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Debounce: reflow chart labels to the new size after resizing settles.
        if hasattr(self, "_resize_debounce"):
            self._resize_debounce.start()

    def _relayout_visible_canvas(self):
        """Re-run tight_layout + redraw on visible chart canvases so labels reflow."""
        for name in ("profile_canvas", "trend_canvas", "wafer_canvas",
                     "best5_canvas", "res_compare_canvas", "flatten_canvas",
                     "bs_bar_canvas", "bs_heatmap_canvas"):
            canvas = getattr(self, name, None)
            if canvas is None or not canvas.isVisible():
                continue
            try:
                canvas.figure.tight_layout()
            except Exception:
                pass
            canvas.draw_idle()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 0)

        # Top: Data Loading
        load_group = self._create_load_panel()
        layout.addWidget(load_group)

        # Main: Splitter (Settings | Tabs)
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Left: Settings (scrollable so content never clips on short screens)
        self.settings_widget = self._create_settings_panel()
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidget(self.settings_widget)
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.settings_scroll.setFrameShape(QFrame.NoFrame)
        self.main_splitter.addWidget(self.settings_scroll)

        # Right: Category Tabs (2-level nested QTabWidget)
        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(600)

        # -- Category tab styling (outer tabs: larger, with icons) --
        self.tabs.setStyleSheet("""
            QTabWidget > QTabBar::tab {
                font-size: 13px; font-weight: bold; padding: 10px 20px;
            }
        """)

        # Create all tab widgets first
        self.profile_canvas = FigureCanvas(Figure(figsize=(12, 9)))
        self.summary_tab = self._create_summary_table()  # sets self.summary_table (inner)
        self.flatten_widget = self._create_flatten_tab()
        self.trend_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.wafer_canvas = FigureCanvas(Figure(figsize=(8, 7)))
        self.best5_canvas = FigureCanvas(Figure(figsize=(12, 6)))
        self.time_widget = self._create_time_tab()
        self.bs_widget = self._create_ball_screw_tab()
        self.res_compare_canvas = FigureCanvas(Figure(figsize=(14, 7)))
        self.qc_widget = self._create_qc_tab()
        self.compare_widget = self._create_compare_tab()
        self.remark_widget = self._create_remark_tab()

        # Profile Charts wrapper with Y-axis scale toolbar
        self.profile_tab_widget = QWidget()
        _pl = QVBoxLayout(self.profile_tab_widget)
        _pl.setContentsMargins(0, 0, 0, 0)
        _pl.setSpacing(2)
        _toolbar = QHBoxLayout()
        _toolbar.addWidget(QLabel("Y-Axis:"))
        self.y_scale_combo = QComboBox()
        self.y_scale_combo.addItems(["Auto", "Unified", "Group"])
        self.y_scale_combo.setFixedWidth(110)
        self.y_scale_combo.currentTextChanged.connect(self._update_profile_chart)
        _toolbar.addWidget(self.y_scale_combo)

        # Separator
        _sep = QFrame()
        _sep.setFrameShape(QFrame.VLine)
        _sep.setFixedHeight(20)
        _toolbar.addWidget(_sep)

        # Resolution simulation slider
        _toolbar.addWidget(QLabel("Resolution:"))
        self.res_slider = QSlider(Qt.Horizontal)
        self.res_slider.setMinimum(1)
        self.res_slider.setMaximum(1)  # Updated when recipe loads
        self.res_slider.setValue(1)
        self.res_slider.setFixedWidth(200)
        self.res_slider.setToolTip("Simulate lower resolution by block-averaging pixels")
        # Debounced connection: update label immediately, but defer chart redraw
        self._res_debounce = QTimer()
        self._res_debounce.setSingleShot(True)
        self._res_debounce.setInterval(150)
        self._res_debounce.timeout.connect(self._update_profile_chart)
        self.res_slider.valueChanged.connect(self._on_res_slider_changed)
        _toolbar.addWidget(self.res_slider)
        self.res_slider_label = QLabel("Original")
        self.res_slider_label.setFixedWidth(160)
        _toolbar.addWidget(self.res_slider_label)
        self.res_reset_btn = QPushButton("Reset")
        self.res_reset_btn.setFixedWidth(50)
        self.res_reset_btn.clicked.connect(lambda: self.res_slider.setValue(1))
        _toolbar.addWidget(self.res_reset_btn)

        _toolbar.addStretch()
        _pl.addLayout(_toolbar)
        _pl.addWidget(self.profile_canvas)

        # Inner tab style (compact)
        _inner_tab_style = """
            QTabBar::tab { font-size: 12px; padding: 6px 14px; }
        """

        # Analysis category
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setStyleSheet(_inner_tab_style)
        self.analysis_tabs.addTab(self.profile_tab_widget, "Profile Charts")
        self.analysis_tabs.addTab(self.summary_tab, "Summary Table")
        self.analysis_tabs.addTab(self.flatten_widget, "Flatten")
        self.analysis_tabs.addTab(self.bs_widget, "Ball Screw Pitch")

        # Visualization category
        self.viz_tabs = QTabWidget()
        self.viz_tabs.setStyleSheet(_inner_tab_style)
        self.viz_tabs.addTab(self.trend_canvas, "Saturation Trend")
        self.viz_tabs.addTab(self.wafer_canvas, "Wafer Map")
        self.viz_tabs.addTab(self.res_compare_canvas, "Resolution Compare")

        # Quality category
        self.quality_tabs = QTabWidget()
        self.quality_tabs.setStyleSheet(_inner_tab_style)
        self.quality_tabs.addTab(self.qc_widget, "QC Check")
        self.quality_tabs.addTab(self.compare_widget, "Compare")

        # Tools category
        self.tools_tabs = QTabWidget()
        self.tools_tabs.setStyleSheet(_inner_tab_style)
        self.tools_tabs.addTab(self.time_widget, "Time Analysis")
        self.tools_tabs.addTab(self.best5_canvas, "Best-5 Window")
        self.tools_tabs.addTab(self.remark_widget, "Remark")

        # Register categories in outer tab
        self.tabs.addTab(self.analysis_tabs, "Analysis")
        self.tabs.addTab(self.viz_tabs, "Visualization")
        self.tabs.addTab(self.quality_tabs, "Quality")
        self.tabs.addTab(self.tools_tabs, "Tools")

        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([260, 1000])

        layout.addWidget(self.main_splitter)

        # F11: Toggle side panel
        self._side_panel_visible = True
        shortcut = QShortcut(QKeySequence(Qt.Key_F11), self)
        shortcut.activated.connect(self._toggle_side_panel)

        # Status Bar
        self.statusBar().showMessage("Ready. Select a data folder to begin.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)

    def _toggle_side_panel(self):
        """Toggle left settings panel visibility (F11)."""
        self._side_panel_visible = not self._side_panel_visible
        self.settings_scroll.setVisible(self._side_panel_visible)
        if self._side_panel_visible:
            self.main_splitter.setSizes([260, 1000])
            self.statusBar().showMessage("Side panel shown (F11)", 2000)
        else:
            self.statusBar().showMessage("Side panel hidden (F11)", 2000)

    def _create_load_panel(self) -> QGroupBox:
        group = QGroupBox("Data Loading")
        layout = QHBoxLayout(group)

        self.path_label = QLabel("No data loaded")
        self.path_label.setStyleSheet("color: #a6adc8; font-size: 12px;")
        layout.addWidget(self.path_label, 1)

        self.load_btn = QPushButton("Open Folder")
        self.load_btn.setObjectName("load_btn")
        self.load_btn.setFixedHeight(32)
        self.load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.load_btn)

        return group

    def _create_settings_panel(self) -> QWidget:
        widget = QWidget()
        widget.setMinimumWidth(240)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Signal Source
        source_group = QGroupBox("Signal Source")
        source_layout = QVBoxLayout(source_group)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Height", "Z Drive"])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.source_combo)
        layout.addWidget(source_group)

        # Range Selector
        range_group = QGroupBox("Recipe Range")
        range_layout = QVBoxLayout(range_group)
        self.range_combo = QComboBox()
        self.range_combo.setStyleSheet(
            "QComboBox { font-size: 13px; font-weight: bold; padding: 6px; }")
        self.range_combo.currentTextChanged.connect(self._on_range_changed)
        range_layout.addWidget(self.range_combo)
        self.range_info_label = QLabel("")
        self.range_info_label.setStyleSheet("font-size: 10px; color: #a6adc8;")
        range_layout.addWidget(self.range_info_label)
        layout.addWidget(range_group)

        # Best-5 Window — use QFrame instead of QGroupBox to avoid clipping
        best5_frame = QFrame()
        best5_frame.setStyleSheet(
            "QFrame#best5Frame { border: 1px solid #45475a; border-radius: 6px; }")
        best5_frame.setObjectName("best5Frame")
        best5_inner = QVBoxLayout(best5_frame)
        best5_inner.setContentsMargins(10, 6, 10, 8)
        best5_inner.setSpacing(6)
        best5_title = QLabel("Best-5 Window")
        best5_title.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 12px;")
        best5_inner.addWidget(best5_title)
        best5_row = QHBoxLayout()
        best5_row.setSpacing(8)
        ws_label = QLabel("Window Size:")
        ws_label.setStyleSheet("font-size: 12px;")
        best5_row.addWidget(ws_label)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(2, 20)
        self.window_spin.setValue(5)
        self.window_spin.setFixedSize(80, 32)
        self.window_spin.setStyleSheet(
            "QSpinBox { padding: 4px 6px; font-size: 14px; }"
            "QSpinBox::up-button { width: 20px; }"
            "QSpinBox::down-button { width: 20px; }")
        self.window_spin.valueChanged.connect(self._on_reanalyze)
        best5_row.addWidget(self.window_spin)
        best5_row.addStretch()
        best5_inner.addLayout(best5_row)
        layout.addWidget(best5_frame)

        # Outlier Exclusion
        outlier_frame = QFrame()
        outlier_frame.setStyleSheet(
            "QFrame#outlierFrame { border: 1px solid #45475a; border-radius: 6px; }")
        outlier_frame.setObjectName("outlierFrame")
        outlier_inner = QVBoxLayout(outlier_frame)
        outlier_inner.setContentsMargins(10, 6, 10, 8)
        outlier_inner.setSpacing(4)

        outlier_title = QLabel("Robust 제외 (Raw와 병기)")
        outlier_title.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 12px;")
        outlier_title.setToolTip(
            "Raw(전체 데이터)는 항상 표시되고, 여기서 고른 제외 방식으로 계산한 "
            "Robust 값이 Summary/Spec에 함께 표시됩니다. None이면 Robust 열 없음.")
        outlier_inner.addWidget(outlier_title)

        outlier_row1 = QHBoxLayout()
        outlier_row1.addWidget(QLabel("Mode:"))
        self.outlier_mode_combo = QComboBox()
        self.outlier_mode_combo.addItems(["None", "Percentile", "Pixels"])
        self.outlier_mode_combo.setFixedSize(120, 28)
        self.outlier_mode_combo.setStyleSheet(
            "QComboBox { background: #1e1e2e; border: 1px solid #45475a; "
            "border-radius: 4px; padding: 2px 6px; color: #cdd6f4; }")
        self.outlier_mode_combo.currentTextChanged.connect(self._on_outlier_mode_changed)
        outlier_row1.addWidget(self.outlier_mode_combo)
        outlier_row1.addStretch()
        outlier_inner.addLayout(outlier_row1)

        outlier_row2 = QHBoxLayout()
        self.outlier_value_label = QLabel("Value:")
        outlier_row2.addWidget(self.outlier_value_label)
        self.outlier_value_spin = QDoubleSpinBox()
        self.outlier_value_spin.setRange(0.0, 100.0)
        self.outlier_value_spin.setValue(5.0)
        self.outlier_value_spin.setDecimals(1)
        self.outlier_value_spin.setSuffix(" %")
        self.outlier_value_spin.setFixedSize(100, 28)
        self.outlier_value_spin.setStyleSheet(
            "QDoubleSpinBox { background: #1e1e2e; border: 1px solid #45475a; "
            "border-radius: 4px; padding: 2px 6px; color: #cdd6f4; }")
        self.outlier_value_spin.setEnabled(False)
        self.outlier_value_spin.valueChanged.connect(self._on_reanalyze)
        outlier_row2.addWidget(self.outlier_value_spin)
        outlier_row2.addStretch()
        outlier_inner.addLayout(outlier_row2)

        # Default to Percentile 1% to match the reference Tool, which always
        # excludes outlier pixels (analyzer.DEFAULT_OUTLIER_MODE/VALUE).
        self.outlier_mode_combo.setCurrentText("Percentile")

        layout.addWidget(outlier_frame)

        # Spec Judgment — redesigned with equipment type + dual spec
        spec_frame = QFrame()
        spec_frame.setStyleSheet(
            "QFrame#specFrame { border: 1px solid #45475a; border-radius: 6px; }")
        spec_frame.setObjectName("specFrame")
        spec_inner = QVBoxLayout(spec_frame)
        spec_inner.setContentsMargins(10, 6, 10, 8)
        spec_inner.setSpacing(4)

        spec_title_row = QHBoxLayout()
        spec_title = QLabel("Spec Judgment")
        spec_title.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 12px;")
        spec_title_row.addWidget(spec_title)
        spec_title_row.addStretch()

        # Help button — OS standard info icon
        from PySide6.QtWidgets import QStyle, QToolTip
        from PySide6.QtCore import QSize
        _spec_tooltip = (
            "<b>OPM Repeatability</b>: Based on Rep. 1\u03c3<br>"
            "<b>Max OPM</b>: Based on maximum OPM value<br><br>"
            "Both must PASS to qualify.<br>"
            "Click for full spec reference table.")
        self.spec_help_btn = QPushButton()
        self.spec_help_btn.setIcon(self.style().standardIcon(
            QStyle.StandardPixmap.SP_MessageBoxInformation))
        self.spec_help_btn.setIconSize(QSize(16, 16))
        self.spec_help_btn.setFixedSize(22, 22)
        self.spec_help_btn.setCursor(Qt.WhatsThisCursor)
        self.spec_help_btn.setStyleSheet(
            "QPushButton { background: transparent; border: none; }"
            "QPushButton:hover { background: #181825; border-radius: 11px; }")
        self.spec_help_btn.setToolTip(_spec_tooltip)
        self.spec_help_btn.clicked.connect(
            lambda checked=False, b=self.spec_help_btn, t=_spec_tooltip:
                QToolTip.showText(b.mapToGlobal(
                    b.rect().bottomLeft()), t, b, b.rect(), 10000))
        self.spec_help_btn.clicked.connect(self._show_spec_info_popup)
        spec_title_row.addWidget(self.spec_help_btn)
        spec_inner.addLayout(spec_title_row)

        # Equipment type radio buttons
        from PySide6.QtWidgets import QRadioButton, QButtonGroup
        radio_style = (
            "QRadioButton { font-size: 12px; font-weight: bold; color: #cdd6f4;"
            "spacing: 6px; padding: 2px 4px; }"
            "QRadioButton::indicator { width: 14px; height: 14px; }"
            "QRadioButton::indicator:checked { "
            "background-color: #89b4fa; border: 2px solid #b4befe; border-radius: 8px; }"
            "QRadioButton::indicator:unchecked { "
            "background-color: #313244; border: 2px solid #585b70; border-radius: 8px; }")
        equip_row = QHBoxLayout()
        equip_row.setSpacing(6)
        self.equip_group = QButtonGroup(self)
        self.radio_iso = QRadioButton("Isolated AE")
        self.radio_dw = QRadioButton("Double Walled AE")
        self.radio_iso.setStyleSheet(radio_style)
        self.radio_dw.setStyleSheet(radio_style)
        self.radio_iso.setChecked(True)  # Default: Isolated AE
        self.equip_group.addButton(self.radio_iso)
        self.equip_group.addButton(self.radio_dw)
        self.radio_iso.toggled.connect(self._on_equipment_changed)
        equip_row.addWidget(self.radio_iso)
        equip_row.addWidget(self.radio_dw)
        equip_row.addStretch()
        spec_inner.addLayout(equip_row)

        # Spec value lines (vertical, left-aligned)
        self.spec_lines_label = QLabel("\u2014")
        self.spec_lines_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.spec_lines_label.setStyleSheet(
            "font-size: 12px; color: #cdd6f4; padding: 4px 2px;")
        self.spec_lines_label.setWordWrap(True)
        spec_inner.addWidget(self.spec_lines_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #45475a;")
        spec_inner.addWidget(sep)

        # Overall verdict
        self.spec_verdict_label = QLabel("\u2014")
        self.spec_verdict_label.setAlignment(Qt.AlignCenter)
        self.spec_verdict_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 4px;")
        spec_inner.addWidget(self.spec_verdict_label)

        layout.addWidget(spec_frame)

        # Scan Parameters
        scan_frame = QFrame()
        scan_frame.setObjectName("scanFrame")
        scan_frame.setStyleSheet(
            "QFrame#scanFrame { border: 1px solid #45475a; border-radius: 6px; }")
        scan_inner = QVBoxLayout(scan_frame)
        scan_inner.setContentsMargins(10, 8, 10, 8)
        scan_inner.setSpacing(2)

        scan_title = QLabel("Scan Parameters")
        scan_title.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #89b4fa; border: none;")
        scan_inner.addWidget(scan_title)

        self.scan_info_label = QLabel("—")
        self.scan_info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.scan_info_label.setStyleSheet(
            "font-size: 11px; color: #a6adc8; line-height: 1.4; border: none;")
        self.scan_info_label.setWordWrap(True)
        scan_inner.addWidget(self.scan_info_label)

        layout.addWidget(scan_frame)

        # Data Info — stretch to fill remaining space
        info_group = QGroupBox("Data Info")
        info_layout = QVBoxLayout(info_group)
        self.info_tree = QTreeWidget()
        self.info_tree.setHeaderLabels(["Property", "Value"])
        self.info_tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        info_layout.addWidget(self.info_tree)
        layout.addWidget(info_group, 1)  # stretch factor = 1 → fills remaining space

        return widget

    def _create_summary_table(self) -> QWidget:
        table = QTableWidget()
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "Range", "Position", "Rep. Max (nm)", "Rep. 1σ (nm)",
            "OPM Max (nm)", "OPM 1σ (nm)"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setAlternatingRowColors(True)
        table.setWordWrap(True)
        # Bigger font for readability
        table.setStyleSheet("""
            QTableWidget { font-size: 13px; }
            QTableWidget::item { padding: 6px; }
            QHeaderView::section { font-size: 13px; padding: 8px; }
        """)
        # Taller rows so each metric cell can show two lines (raw / robust).
        table.verticalHeader().setDefaultSectionSize(46)
        self.summary_table = table

        # Wrap with a legend so the raw/robust dual display is self-explanatory.
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        legend = QLabel(
            "각 지표 칸 — 상단: <b>Raw</b>(전체 데이터, 참 측정값) · 하단: "
            "<b>Robust</b>(outlier 제외). 두 값이 다른 칸은 노랑으로 강조됩니다.")
        legend.setWordWrap(True)
        legend.setStyleSheet("color:#a6adc8; font-size:11px; padding:3px 6px;")
        v.addWidget(legend)
        v.addWidget(table)
        return container

    def _create_flatten_tab(self) -> QWidget:
        """Flatten tab — single row controls for maximum chart space."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Controls — single compact row
        ctrl_row = QHBoxLayout()
        ctrl_row.setContentsMargins(4, 4, 4, 4)
        ctrl_row.setSpacing(6)

        ctrl_row.addWidget(QLabel("Pos:"))
        self.flat_pos_combo = QComboBox()
        self.flat_pos_combo.addItems(POSITION_LABELS)
        self.flat_pos_combo.setFixedWidth(70)
        ctrl_row.addWidget(self.flat_pos_combo)

        ctrl_row.addWidget(QLabel("Rep:"))
        self.flat_rep_combo = QComboBox()
        self.flat_rep_combo.setFixedWidth(90)
        ctrl_row.addWidget(self.flat_rep_combo)

        ctrl_row.addWidget(QLabel("Ord:"))
        self.flat_order_combo = QComboBox()
        self.flat_order_combo.addItems([str(i) for i in range(13)])
        self.flat_order_combo.setCurrentIndex(1)
        self.flat_order_combo.setFixedWidth(50)
        ctrl_row.addWidget(self.flat_order_combo)

        ctrl_row.addWidget(QLabel("Edge%:"))
        self.flat_edge_spin = QDoubleSpinBox()
        self.flat_edge_spin.setRange(0, 10)
        self.flat_edge_spin.setValue(1.0)
        self.flat_edge_spin.setSingleStep(0.5)
        self.flat_edge_spin.setFixedSize(75, 30)
        self.flat_edge_spin.setStyleSheet(
            "QDoubleSpinBox { padding: 3px 4px; font-size: 12px; }"
            "QDoubleSpinBox::up-button { width: 18px; }"
            "QDoubleSpinBox::down-button { width: 18px; }")
        ctrl_row.addWidget(self.flat_edge_spin)

        self.flat_execute_btn = QPushButton("Execute")
        self.flat_execute_btn.setStyleSheet(
            "background-color: #40a02b; color: white; font-weight: bold;"
            "padding: 4px 12px;")
        self.flat_execute_btn.clicked.connect(self._on_flatten_execute)
        ctrl_row.addWidget(self.flat_execute_btn)

        self.flat_undo_btn = QPushButton("Undo")
        self.flat_undo_btn.setStyleSheet("padding: 4px 12px;")
        self.flat_undo_btn.clicked.connect(self._on_flatten_undo)
        self.flat_undo_btn.setEnabled(False)
        ctrl_row.addWidget(self.flat_undo_btn)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Stats (minimal height)
        self.flat_stats_label = QLabel("")
        self.flat_stats_label.setStyleSheet("font-size: 11px; color: #a6adc8; padding: 2px 4px;")
        self.flat_stats_label.setFixedHeight(20)
        layout.addWidget(self.flat_stats_label)

        # Canvas
        self.flatten_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.flatten_canvas)

        return widget

    def _create_time_tab(self) -> QWidget:
        """Time Analysis tab — shows per-repeat timing, gaps, estimation."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # Summary section
        summary_frame = QFrame()
        summary_frame.setStyleSheet(
            "QFrame { background-color: #181825; border-radius: 6px; padding: 12px; }")
        summary_layout = QGridLayout(summary_frame)
        summary_layout.setHorizontalSpacing(24)
        summary_layout.setVerticalSpacing(6)

        self.time_total_label = QLabel("—")
        self.time_total_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa;")
        summary_layout.addWidget(QLabel("Total Duration:"), 0, 0)
        summary_layout.addWidget(self.time_total_label, 0, 1)

        self.time_avg_repeat_label = QLabel("—")
        self.time_avg_repeat_label.setStyleSheet("font-size: 14px; color: #cdd6f4;")
        summary_layout.addWidget(QLabel("Avg per Repeat:"), 1, 0)
        summary_layout.addWidget(self.time_avg_repeat_label, 1, 1)

        self.time_avg_point_label = QLabel("—")
        self.time_avg_point_label.setStyleSheet("font-size: 14px; color: #cdd6f4;")
        summary_layout.addWidget(QLabel("Avg per Point:"), 2, 0)
        summary_layout.addWidget(self.time_avg_point_label, 2, 1)

        self.time_continuous_label = QLabel("—")
        self.time_continuous_label.setStyleSheet("font-size: 14px;")
        summary_layout.addWidget(QLabel("Continuity:"), 3, 0)
        summary_layout.addWidget(self.time_continuous_label, 3, 1)

        # Estimation section
        summary_layout.addWidget(QLabel(""), 4, 0)  # spacer
        est_title = QLabel("Estimate for N Repeats:")
        est_title.setStyleSheet("font-weight: bold; color: #89b4fa;")
        summary_layout.addWidget(est_title, 5, 0, 1, 2)

        est_row = QHBoxLayout()
        self.time_est_spin = QSpinBox()
        self.time_est_spin.setRange(1, 100)
        self.time_est_spin.setValue(10)
        self.time_est_spin.setFixedWidth(80)
        self.time_est_spin.valueChanged.connect(self._update_time_estimate)
        est_row.addWidget(QLabel("Repeat Count:"))
        est_row.addWidget(self.time_est_spin)
        self.time_est_result = QLabel("—")
        self.time_est_result.setStyleSheet("font-size: 16px; font-weight: bold; color: #f9e2af;")
        est_row.addWidget(QLabel("→"))
        est_row.addWidget(self.time_est_result)
        est_row.addStretch()

        est_widget = QWidget()
        est_widget.setLayout(est_row)
        summary_layout.addWidget(est_widget, 6, 0, 1, 2)

        layout.addWidget(summary_frame)

        # Per-repeat table
        self.time_table = QTableWidget()
        self.time_table.setColumnCount(7)
        self.time_table.setHorizontalHeaderLabels([
            "Repeat", "Folder", "Start", "End", "Duration", "Per Point", "Gap"
        ])
        self.time_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.time_table.setStyleSheet("""
            QTableWidget { font-size: 13px; }
            QTableWidget::item { padding: 6px; }
            QHeaderView::section { font-size: 13px; padding: 8px; }
        """)
        self.time_table.verticalHeader().setDefaultSectionSize(30)
        layout.addWidget(self.time_table)

        return widget

    def _create_ball_screw_tab(self) -> QWidget:
        """Ball Screw Pitch analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── Controls row ─────────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        ctrl_row.addWidget(QLabel("Material:"))
        self.bs_material_combo = QComboBox()
        self.bs_material_combo.addItems(["AL (≤6.0 nm)", "SUS (≤4.5 nm)"])
        self.bs_material_combo.setFixedWidth(130)
        ctrl_row.addWidget(self.bs_material_combo)

        from PySide6.QtWidgets import QCheckBox
        self.bs_show_stab_check = QCheckBox("안정화 포인트 표시")
        self.bs_show_stab_check.setChecked(False)
        self.bs_show_stab_check.stateChanged.connect(self._on_bs_filter_changed)
        ctrl_row.addWidget(self.bs_show_stab_check)

        self.bs_analyze_btn = QPushButton("Analyze")
        self.bs_analyze_btn.setStyleSheet(
            "background-color: #40a02b; color: white; font-weight: bold; padding: 4px 14px;")
        self.bs_analyze_btn.clicked.connect(self._on_bs_analyze)
        ctrl_row.addWidget(self.bs_analyze_btn)

        # Verdict badge
        self.bs_verdict_label = QLabel("—")
        self.bs_verdict_label.setAlignment(Qt.AlignCenter)
        self.bs_verdict_label.setFixedWidth(80)
        self.bs_verdict_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; border: 2px solid #45475a;"
            "border-radius: 6px; padding: 4px; color: #a6adc8;")
        ctrl_row.addWidget(self.bs_verdict_label)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # ── Plot area: bar chart (left) + heatmap (right) ────────────────────
        plot_row = QHBoxLayout()
        self.bs_bar_canvas = FigureCanvas(Figure(figsize=(8, 4)))
        self.bs_heatmap_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        plot_row.addWidget(self.bs_bar_canvas, 6)
        plot_row.addWidget(self.bs_heatmap_canvas, 4)
        layout.addLayout(plot_row)

        # ── Summary table ────────────────────────────────────────────────────
        self.bs_table = QTableWidget()
        self.bs_table.setMinimumHeight(180)
        self.bs_table.setMaximumHeight(220)
        self.bs_table.setStyleSheet("""
            QTableWidget { font-size: 12px; }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section { font-size: 12px; padding: 6px;
                background-color: #313244; color: #89b4fa;
                border: 1px solid #45475a; font-weight: bold; }
        """)
        self.bs_table.verticalHeader().setDefaultSectionSize(26)
        layout.addWidget(self.bs_table)

        return widget

    # ─── QC Check Tab ────────────────────────────────────────

    _QC_COLORS = {"PASS": "#a6e3a1", "WARN": "#f9e2af", "FAIL": "#f38ba8"}
    _QC_CHECK_NAMES = [
        ("QC-1", "File Matching (Recipe vs Raw)"),
        ("QC-2", "Data Equivalence"),
        ("QC-3", "Scan Parameter Consistency"),
        ("QC-4", "Position Completeness"),
        ("QC-5", "Outlier Detection"),
        ("QC-6", "Pixel Count Consistency"),
    ]

    def _create_qc_tab(self) -> QWidget:
        """QC Check tab — data integrity verification."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── Controls row ──
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.qc_run_btn = QPushButton("Run QC")
        self.qc_run_btn.setStyleSheet(
            "background-color: #40a02b; color: white; font-weight: bold; padding: 4px 14px;")
        self.qc_run_btn.clicked.connect(self._on_qc_run)
        ctrl_row.addWidget(self.qc_run_btn)

        self.qc_verdict_label = QLabel("-")
        self.qc_verdict_label.setAlignment(Qt.AlignCenter)
        self.qc_verdict_label.setFixedWidth(80)
        self.qc_verdict_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; border: 2px solid #45475a;"
            "border-radius: 6px; padding: 4px; color: #a6adc8;")
        ctrl_row.addWidget(self.qc_verdict_label)

        self.qc_timestamp_label = QLabel("")
        self.qc_timestamp_label.setStyleSheet("font-size: 11px; color: #6c7086;")
        ctrl_row.addWidget(self.qc_timestamp_label)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # ── Summary panel (6 check items) ──
        summary_frame = QFrame()
        summary_frame.setStyleSheet(
            "QFrame#qcSummary { background-color: #181825; border: 1px solid #45475a;"
            "border-radius: 6px; }")
        summary_frame.setObjectName("qcSummary")
        summary_grid = QGridLayout(summary_frame)
        summary_grid.setContentsMargins(12, 8, 12, 8)
        summary_grid.setHorizontalSpacing(12)
        summary_grid.setVerticalSpacing(4)

        self.qc_summary_labels: list[tuple[QLabel, QLabel, QLabel]] = []
        for i, (check_id, check_name) in enumerate(self._QC_CHECK_NAMES):
            status_lbl = QLabel("-")
            status_lbl.setFixedWidth(36)
            status_lbl.setAlignment(Qt.AlignCenter)
            status_lbl.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #6c7086;")

            name_lbl = QLabel(f"{check_id}: {check_name}")
            name_lbl.setStyleSheet("font-size: 12px; color: #cdd6f4;")

            summary_lbl = QLabel("")
            summary_lbl.setStyleSheet("font-size: 11px; color: #a6adc8;")

            summary_grid.addWidget(status_lbl, i, 0)
            summary_grid.addWidget(name_lbl, i, 1)
            summary_grid.addWidget(summary_lbl, i, 2)
            self.qc_summary_labels.append((status_lbl, name_lbl, summary_lbl))

        layout.addWidget(summary_frame)

        # ── Detail section ──
        detail_row = QHBoxLayout()
        detail_row.setSpacing(8)
        detail_row.addWidget(QLabel("Detail:"))
        self.qc_detail_combo = QComboBox()
        for check_id, check_name in self._QC_CHECK_NAMES:
            self.qc_detail_combo.addItem(f"{check_id}: {check_name}")
        self.qc_detail_combo.setFixedWidth(280)
        self.qc_detail_combo.currentIndexChanged.connect(self._on_qc_detail_changed)
        detail_row.addWidget(self.qc_detail_combo)
        detail_row.addStretch()
        layout.addLayout(detail_row)

        self.qc_detail_table = QTableWidget()
        self.qc_detail_table.setStyleSheet("""
            QTableWidget { font-size: 12px; }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section { font-size: 12px; padding: 6px;
                background-color: #313244; color: #89b4fa;
                border: 1px solid #45475a; font-weight: bold; }
        """)
        self.qc_detail_table.verticalHeader().setDefaultSectionSize(26)
        self.qc_detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.qc_detail_table)

        return widget

    def _on_qc_run(self):
        """Run QC checks on current recipe."""
        if not self.current_recipe:
            QMessageBox.warning(self, "No Data", "Load data first.")
            return
        try:
            self.statusBar().showMessage("Running QC checks...")
            QApplication.processEvents()
            signal_source = self.source_combo.currentText()
            self.current_qc_result = run_qc_checks(self.current_recipe, signal_source)
            self._update_qc_tab()
            self.statusBar().showMessage(
                f"QC Check complete: {self.current_qc_result.overall_status}")
        except Exception as e:
            QMessageBox.critical(self, "QC Check Error", f"{type(e).__name__}: {e}")

    def _update_qc_tab(self):
        """Update QC tab from current_qc_result."""
        qc = self.current_qc_result
        if qc is None:
            self._clear_qc_tab()
            return

        # Verdict badge
        color = self._QC_COLORS.get(qc.overall_status, "#a6adc8")
        self.qc_verdict_label.setText(qc.overall_status)
        self.qc_verdict_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; border: 2px solid {color};"
            f"border-radius: 6px; padding: 4px; color: {color};")
        self.qc_timestamp_label.setText(qc.timestamp)

        # Summary rows
        for i, check in enumerate(qc.checks):
            if i >= len(self.qc_summary_labels):
                break
            status_lbl, name_lbl, summary_lbl = self.qc_summary_labels[i]
            c = self._QC_COLORS.get(check.status, "#a6adc8")
            status_lbl.setText(check.status)
            status_lbl.setStyleSheet(
                f"font-size: 12px; font-weight: bold; color: {c};"
                f"border: 1px solid {c}; border-radius: 3px;")
            summary_lbl.setText(check.summary)

        # Detail table
        self._on_qc_detail_changed()

    def _clear_qc_tab(self):
        """Reset QC tab to initial state."""
        self.qc_verdict_label.setText("-")
        self.qc_verdict_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; border: 2px solid #45475a;"
            "border-radius: 6px; padding: 4px; color: #a6adc8;")
        self.qc_timestamp_label.setText("")

        for status_lbl, name_lbl, summary_lbl in self.qc_summary_labels:
            status_lbl.setText("-")
            status_lbl.setStyleSheet(
                "font-size: 12px; font-weight: bold; color: #6c7086;")
            summary_lbl.setText("")

        self.qc_detail_table.clear()
        self.qc_detail_table.setRowCount(0)
        self.qc_detail_table.setColumnCount(0)

    def _on_qc_detail_changed(self):
        """Populate detail table for the selected QC check."""
        idx = self.qc_detail_combo.currentIndex()
        qc = self.current_qc_result
        if qc is None or idx < 0 or idx >= len(qc.checks):
            self.qc_detail_table.clear()
            self.qc_detail_table.setRowCount(0)
            self.qc_detail_table.setColumnCount(0)
            return

        check = qc.checks[idx]
        details = check.details
        if not details:
            self.qc_detail_table.clear()
            self.qc_detail_table.setRowCount(0)
            self.qc_detail_table.setColumnCount(1)
            self.qc_detail_table.setHorizontalHeaderLabels(["Info"])
            self.qc_detail_table.setRowCount(1)
            self.qc_detail_table.setItem(0, 0, QTableWidgetItem("No detail data"))
            return

        # Build table from detail dicts
        columns = list(details[0].keys())
        self.qc_detail_table.clear()
        self.qc_detail_table.setColumnCount(len(columns))
        self.qc_detail_table.setHorizontalHeaderLabels(columns)
        self.qc_detail_table.setRowCount(len(details))

        for row_idx, row_data in enumerate(details):
            for col_idx, col_key in enumerate(columns):
                val = row_data.get(col_key, "")
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)

                # Color-code status column and outlier rows
                status_val = row_data.get("status", "")
                if col_key == "status":
                    c = self._QC_COLORS.get(status_val, "#cdd6f4")
                    item.setForeground(QColor(c))
                elif status_val in ("FAIL", "WARN"):
                    if col_key == "is_outlier" and row_data.get("is_outlier"):
                        item.setForeground(QColor(self._QC_COLORS["WARN"]))
                    elif status_val == "FAIL":
                        item.setForeground(QColor(self._QC_COLORS["FAIL"]))

                self.qc_detail_table.setItem(row_idx, col_idx, item)

    # ─── Resolution Compare ──────────────────────────────────

    def _update_resolution_compare(self):
        """Update cross-range resolution comparison chart."""
        if not self.dataset or len(self.dataset.recipes) < 2:
            return

        try:
            from ..visualization.plot_manager import create_resolution_comparison_figure

            # Find max resolution (lowest res = largest nm/px)
            max_res = 0
            for label, recipe in self.dataset.recipes.items():
                for repeat in recipe.repeats:
                    for pos, prof in repeat.profiles.items():
                        px = len(prof.raw_data)
                        res = prof.scan_size_um * 1000 / px if px > 0 else 0
                        if res > max_res:
                            max_res = res
                    break  # Only need first repeat

            if max_res == 0:
                return

            # Compute normalized OPM for each range
            norm_data = {}
            for label in sorted(self.dataset.recipes.keys(),
                                key=lambda x: int(x.replace('mm', '')),
                                reverse=True):
                recipe = self.dataset.recipes[label]
                norm_data[label] = compute_normalized_opm(recipe, max_res)

            # Get spec limits based on equipment type
            from ..core.analyzer import SPEC_MAX_OPM_ISO, SPEC_MAX_OPM_DW
            equipment_type = "iso" if self.radio_iso.isChecked() else "dw"
            spec_limits = SPEC_MAX_OPM_ISO if equipment_type == "iso" else SPEC_MAX_OPM_DW

            fig = create_resolution_comparison_figure(norm_data, figsize=(14, 7),
                                                      spec_limits=spec_limits)
            self._update_canvas(self.res_compare_canvas, fig)
        except Exception as e:
            self.statusBar().showMessage(f"Resolution Compare error: {e}")

    # ─── Compare Tab ─────────────────────────────────────────

    def _create_compare_tab(self) -> QWidget:
        """Compare tab — cross-process comparison."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Controls
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        self.compare_load_btn = QPushButton("Load Reference")
        self.compare_load_btn.setStyleSheet(
            "background-color: #1e66f5; color: white; font-weight: bold; padding: 4px 14px;")
        self.compare_load_btn.clicked.connect(self._on_compare_load)
        ctrl_row.addWidget(self.compare_load_btn)

        self.compare_info_label = QLabel("No reference loaded")
        self.compare_info_label.setStyleSheet("font-size: 12px; color: #a6adc8;")
        ctrl_row.addWidget(self.compare_info_label)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Compare table
        self.compare_table = QTableWidget()
        self.compare_table.setStyleSheet("""
            QTableWidget { font-size: 12px; }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section { font-size: 12px; padding: 6px;
                background-color: #313244; color: #89b4fa;
                border: 1px solid #45475a; font-weight: bold; }
        """)
        self.compare_table.verticalHeader().setDefaultSectionSize(26)
        self.compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.compare_table)

        return widget

    def _on_compare_load(self):
        """Load reference data for comparison."""
        if not self.current_result:
            QMessageBox.warning(self, "No Data", "Load primary data first.")
            return

        from PySide6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(
            self, "Select Reference Data Folder")
        if not folder:
            return

        try:
            self.statusBar().showMessage("Loading reference data...")
            QApplication.processEvents()

            from ..core.data_loader import load_recipe, load_dataset
            from ..core.analyzer import analyze_recipe
            from pathlib import Path

            ref_path = Path(folder)

            # Try to load matching range
            signal = self.source_combo.currentText()
            ref_recipe = load_recipe(ref_path, signal_source=signal)

            equipment_type = "dw" if self.equip_dw_radio.isChecked() else "iso"
            window_size = self.window_spin.value()
            outlier_mode = self.outlier_mode_combo.currentText().lower()
            outlier_value = self.outlier_value_spin.value() if outlier_mode != "none" else 0.0
            self.reference_result = analyze_recipe(
                ref_recipe, window_size=window_size,
                equipment_type=equipment_type,
                outlier_mode=outlier_mode,
                outlier_value=outlier_value)

            self.current_compare_result = compare_results(
                self.current_result, self.reference_result)
            self.current_compare_result.current_label = self.current_recipe.range_label
            self.current_compare_result.reference_label = f"Ref ({ref_path.name})"

            self._update_compare_tab()
            self.compare_info_label.setText(
                f"Reference: {ref_path.name} ({ref_recipe.range_label}, "
                f"{ref_recipe.repeat_count} repeats)")
            self.statusBar().showMessage("Comparison complete.")
        except Exception as e:
            QMessageBox.critical(self, "Compare Error", f"{type(e).__name__}: {e}")

    def _update_compare_tab(self):
        """Update compare table from current_compare_result."""
        cmp = self.current_compare_result
        if cmp is None:
            self._clear_compare_tab()
            return

        rows = get_compare_table(cmp)
        if not rows:
            return

        columns = list(rows[0].keys())
        self.compare_table.clear()
        self.compare_table.setColumnCount(len(columns))
        self.compare_table.setHorizontalHeaderLabels(columns)
        self.compare_table.setRowCount(len(rows))

        for i, row in enumerate(rows):
            for j, col in enumerate(columns):
                val = row.get(col, "")
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)

                # Color-code delta columns
                if "Δ" in col and isinstance(val, (int, float)):
                    if val > 0:
                        item.setForeground(QColor("#f38ba8"))  # red = worse
                    elif val < 0:
                        item.setForeground(QColor("#a6e3a1"))  # green = better

                # Group rows
                if row.get("Position", "").startswith("["):
                    item.setBackground(QColor("#1e1e2e"))
                    item.setForeground(QColor("#f9e2af"))
                    item.setFont(QFont("Segoe UI", 11, QFont.Bold))

                self.compare_table.setItem(i, j, item)

    def _clear_compare_tab(self):
        """Reset compare tab."""
        self.compare_table.clear()
        self.compare_table.setRowCount(0)
        self.compare_table.setColumnCount(0)
        self.compare_info_label.setText("No reference loaded")

    def _create_remark_tab(self) -> QWidget:
        """Remark tab — Export + Split-Pane Usage Guide."""
        from PySide6.QtWidgets import QTextBrowser, QListWidget, QSplitter

        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Export section
        export_frame = QFrame()
        export_frame.setStyleSheet(
            "QFrame#exportFrame { border: 1px solid #45475a; border-radius: 6px; }")
        export_frame.setObjectName("exportFrame")
        export_layout = QVBoxLayout(export_frame)
        export_layout.setContentsMargins(16, 12, 16, 12)
        export_layout.setSpacing(8)

        export_title = QLabel("Export Analysis Results")
        export_title.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #89b4fa;")
        export_layout.addWidget(export_title)

        export_desc = QLabel(
            "Export summary CSV, average line profiles, spec checklist,\n"
            "and all chart images (PNG) to a selected folder.")
        export_desc.setStyleSheet("font-size: 12px; color: #a6adc8;")
        export_desc.setWordWrap(True)
        export_layout.addWidget(export_desc)

        self.export_btn = QPushButton("Export Results")
        self.export_btn.setObjectName("export_btn")
        self.export_btn.setFixedHeight(40)
        self.export_btn.setStyleSheet(
            "QPushButton { background-color: #1e66f5; color: white;"
            "font-weight: bold; font-size: 14px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #2e7fff; }"
            "QPushButton:disabled { background-color: #45475a; color: #6c7086; }")
        self.export_btn.clicked.connect(self._on_export)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)

        layout.addWidget(export_frame)

        # Split-Pane Guide: Left menu + Right content
        self._guide_contents = self._get_guide_contents()

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(
            "QSplitter::handle { background-color: #45475a; width: 2px; }")

        # Left: topic list
        self.guide_list = QListWidget()
        self.guide_list.setStyleSheet("""
            QListWidget {
                background-color: #181825; color: #cdd6f4;
                border: 1px solid #45475a; border-radius: 6px;
                font-size: 13px; padding: 4px;
            }
            QListWidget::item {
                padding: 8px 12px; border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #313244; color: #89b4fa; font-weight: bold;
            }
            QListWidget::item:hover {
                background-color: #1e1e2e;
            }
        """)
        for title, _ in self._guide_contents:
            self.guide_list.addItem(title)
        self.guide_list.setFixedWidth(210)
        self.guide_list.currentRowChanged.connect(self._on_guide_topic_changed)

        # Right: content browser
        self.guide_browser = QTextBrowser()
        self.guide_browser.setOpenExternalLinks(False)
        self.guide_browser.setStyleSheet(
            "QTextBrowser { background-color: #181825; color: #cdd6f4;"
            "border: 1px solid #45475a; border-radius: 6px;"
            "padding: 16px; font-size: 12px; }")

        splitter.addWidget(self.guide_list)
        splitter.addWidget(self.guide_browser)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

        # Select first topic
        self.guide_list.setCurrentRow(0)

        return widget

    def _on_guide_topic_changed(self, index: int):
        """Display the selected guide topic content."""
        if 0 <= index < len(self._guide_contents):
            _, html = self._guide_contents[index]
            self.guide_browser.setHtml(html)

    @staticmethod
    def _guide_style() -> str:
        return """<style>
            h2 { color: #89b4fa; margin-top: 12px; margin-bottom: 8px; font-size: 16px; }
            h3 { color: #f9e2af; margin-top: 14px; margin-bottom: 4px; font-size: 13px; }
            p, li { color: #cdd6f4; font-size: 12px; line-height: 1.6; }
            ul { margin-left: 16px; margin-top: 4px; }
            .metric { color: #a6e3a1; font-weight: bold; }
            .note { color: #fab387; font-style: italic; }
            .key { color: #89b4fa; font-weight: bold; }
            .warn { color: #f9e2af; font-weight: bold; }
            .fail { color: #f38ba8; font-weight: bold; }
            .pass { color: #a6e3a1; font-weight: bold; }
            table { border-collapse: collapse; margin: 8px 0; width: 100%; }
            th { background-color: #313244; color: #89b4fa; padding: 6px 10px;
                 text-align: left; font-size: 12px; border: 1px solid #45475a; }
            td { padding: 5px 10px; font-size: 12px; border: 1px solid #45475a; }
        </style>"""

    @staticmethod
    def _get_guide_contents() -> list[tuple[str, str]]:
        s = MainWindow._guide_style()
        return [
            ("Overview", f"""{s}
            <h2>Sliding Stage OPM Repeatability Analyzer</h2>
            <p>Park Systems Sliding Stage의 OPM 재현성을 분석하는 데스크탑 도구입니다.</p>

            <h3>탭 카테고리</h3>
            <p>12개 기능이 4개 카테고리로 분류되어 있습니다:</p>

            <table>
            <tr><th>카테고리</th><th>탭</th><th>용도</th></tr>
            <tr><td><b>Analysis</b></td>
                <td>Profile Charts, Summary Table, Flatten, Ball Screw Pitch</td>
                <td>핵심 분석 기능</td></tr>
            <tr><td><b>Visualization</b></td>
                <td>Saturation Trend, Wafer Map, Resolution Compare</td>
                <td>차트/맵 시각화</td></tr>
            <tr><td><b>Quality</b></td>
                <td>QC Check, Compare</td>
                <td>품질 검증</td></tr>
            <tr><td><b>Tools</b></td>
                <td>Time Analysis, Best-5 Window, Remark</td>
                <td>보조 도구 + 가이드</td></tr>
            </table>

            <h3>사용 순서</h3>
            <ul>
            <li>1. 좌측 상단 <b>Open Folder</b>로 데이터 폴더 선택</li>
            <li>2. Signal Source (Height / Z Drive) 선택</li>
            <li>3. Range 선택 시 자동 분석 실행</li>
            <li>4. 각 탭에서 상세 결과 확인</li>
            <li>5. Remark 탭에서 <b>Export Results</b>로 결과 내보내기</li>
            </ul>

            <h3>데이터 소스</h3>
            <p>장비 측정 후 저장되는 <b>TIFF 파일</b>을 분석합니다.</p>
            <ul>
            <li>지원 형식: Park Systems XE 계열 Custom TIFF (Tag 50434/50435)</li>
            <li>Profile: 1D Height 또는 Z Drive 신호 (8192 pixels)</li>
            <li>측정 Grid: 9개 Position (3×3, 1_LT ~ 9_RB)</li>
            </ul>
            """),

            ("1. Scan Parameters", f"""{s}
            <h2>Scan Parameters (좌측 패널)</h2>
            <p>현재 로드된 Range의 측정 파라미터를 좌측 패널에 표시합니다.</p>

            <h3>표시 항목</h3>
            <table>
            <tr><th>항목</th><th>의미</th><th>예시</th></tr>
            <tr><td><b>Recipe</b></td><td>측정 Recipe 이름</td><td>Profile_25mm_Dynamic</td></tr>
            <tr><td><b>Size</b></td><td>스캔 영역 크기 (µm)</td><td>25000 µm</td></tr>
            <tr><td><b>Px</b></td><td>프로파일 데이터 포인트 수</td><td>8192</td></tr>
            <tr><td><b>Resolution</b></td><td>픽셀당 물리적 크기 (nm/px)</td><td>3052 nm/px</td></tr>
            <tr><td><b>Speed</b></td><td>스캔 속도 (mm/s)</td><td>0.1 mm/s</td></tr>
            <tr><td><b>SP</b></td><td>Set Point — Tip-Surface 상호작용 강도</td><td>30.0</td></tr>
            <tr><td><b>Z Gain</b></td><td>Z Servo Gain — 피드백 제어 이득</td><td>1.5</td></tr>
            </table>

            <h3>Profile Charts 연동</h3>
            <p>Profile Charts 탭의 상단 제목(suptitle)에도 동일한 파라미터가 한 줄로 요약 표시됩니다.</p>
            <p>예: <code>25mm | 8192px | 3052nm/px | 0.1mm/s | SP=30.0</code></p>

            <h3>활용</h3>
            <ul>
            <li>Range 변경 시 파라미터가 자동 갱신 → 측정 조건 즉시 확인</li>
            <li>Range 간 파라미터 차이 파악 (예: 1mm만 Set Point가 다른 경우)</li>
            <li>Resolution 차이가 OPM에 미치는 영향을 이해하는 기초 정보</li>
            </ul>
            """),

            ("2. Profile Charts", f"""{s}
            <h2>Profile Charts</h2>
            <p>9개 Position (3×3 Grid)의 프로파일 오버레이를 표시합니다.</p>

            <h3>읽는 방법</h3>
            <ul>
            <li>각 서브플롯은 해당 Position의 <b>모든 Repeat</b> 프로파일을 겹쳐 보여줍니다.</li>
            <li>겹침이 클수록 재현성이 좋고, 산포가 클수록 편차가 큽니다.</li>
            <li>특정 Repeat에서 이상 프로파일이 보이면 장비 이상 / 환경 변화를 의심합니다.</li>
            </ul>

            <h3>Position 배치</h3>
            <table>
            <tr><td>1_LT (좌상)</td><td>2_CT (중상)</td><td>3_RT (우상)</td></tr>
            <tr><td>4_LM (좌중)</td><td>5_CM (중앙)</td><td>6_RM (우중)</td></tr>
            <tr><td>7_LB (좌하)</td><td>8_CB (중하)</td><td>9_RB (우하)</td></tr>
            </table>

            <h3>Y-Axis 스케일 모드</h3>
            <p>차트 상단의 <b>Y-Axis</b> 콤보박스로 9개 서브플롯의 Y축 스케일을 전환할 수 있습니다.</p>

            <table>
            <tr><th>모드</th><th>동작</th><th>용도</th></tr>
            <tr><td><b>Auto</b></td>
                <td>각 서브플롯 독립 auto-scale</td>
                <td>개별 Position의 세부 형상 관찰</td></tr>
            <tr><td><b>Unified</b></td>
                <td>9개 전체 동일 Y축 범위 (±max)</td>
                <td>Position 간 OPM 크기 비교</td></tr>
            <tr><td><b>Group</b></td>
                <td>Center/Side/Edge 그룹별 동일 Y축</td>
                <td>그룹 내 비교 (Edge가 Center를 압축하지 않음)</td></tr>
            </table>

            <p><b>Unified</b>로 전환하면 2nm 변동과 20nm 변동의 차이가 시각적으로 즉시 드러납니다.
            <b>Group</b> 모드는 Edge의 큰 OPM이 Center 디테일을 압축하는 것을 방지하면서
            같은 그룹 내 Position을 비교할 때 유용합니다.</p>

            <h3>활용 팁</h3>
            <ul>
            <li>Edge 영역(1_LT, 3_RT, 7_LB, 9_RB)은 Center보다 OPM이 높은 것이 일반적입니다.</li>
            <li>프로파일 형상이 Repeat 간 크게 변하면 Stage 안정성 점검이 필요합니다.</li>
            <li><b>Unified</b> 모드에서 특정 Position만 유독 크면 해당 위치의 Stage 문제를 의심합니다.</li>
            </ul>
            """),

            ("3. Summary Table", f"""{s}
            <h2>Summary Table</h2>
            <p>Position별 통계 테이블입니다. Best-5 Window 기준 데이터를 사용합니다.</p>

            <h3>지표 계산 알고리즘</h3>
            <p><b>공통 전처리</b>: Order-1 LS Flatten (양끝 1% 픽셀로만 fitting, 전체에 적용) + Outlier pixel 제외</p>
            <table>
            <tr><th>지표</th><th>공식</th><th>의미</th></tr>
            <tr><td><span class='metric'>Rep. Max</span></td>
                <td>유효 pixel의 repeat간 Max-Min 중 최대값</td><td>재현성 최악 지점</td></tr>
            <tr><td><span class='metric'>Rep. 1σ</span></td>
                <td>유효 pixel별 repeat std의 RMS<br><code>sqrt(mean(pixel_stds²))</code></td><td>재현성 RMS 산포</td></tr>
            <tr><td><span class='metric'>OPM Max</span></td>
                <td>repeat별 유효 pixel Max-Min 중 최대값</td><td>프로파일 형상 크기</td></tr>
            <tr><td><span class='metric'>OPM 1σ</span></td>
                <td>전체 repeat×유효 pixel 높이의 RMS from zero<br><code>sqrt(mean(all_heights²))</code></td><td>Leveling 후 형상 RMS (Bow 크기)</td></tr>
            </table>
            <p class='note'>상세 명세: <code>docs/algorithm_spec.md</code> 참조</p>

            <h3>Total 행 해석</h3>
            <ul>
            <li><b>Mean</b>: 9개 Position 평균값</li>
            <li><b>Stdev</b>: Position 간 편차</li>
            <li><b>Max</b>: 최대값 (Spec 판정에 사용)</li>
            <li><b>RMS</b>: Root Mean Square (Spec 판정에 사용)</li>
            </ul>
            """),

            ("4. Flatten", f"""{s}
            <h2>Flatten</h2>
            <p>개별 프로파일에 대해 Polynomial Flattening을 적용합니다.</p>

            <h3>사용 방법</h3>
            <ul>
            <li><b>Position</b> / <b>Repeat</b> 선택 후 <b>Order</b> 설정</li>
            <li><b>Edge%</b>: 양쪽 가장자리 데이터 제외 비율 (기본 1%)</li>
            <li><b>Execute</b> 클릭 시 Original / Flattened / Histogram 시각화</li>
            <li>OPM 변화량과 RMS 변화량이 Status Bar에 표시됩니다.</li>
            </ul>

            <h3>Order 선택 가이드</h3>
            <table>
            <tr><th>Order</th><th>제거 성분</th><th>용도</th></tr>
            <tr><td>1 (Linear)</td><td>Tilt</td><td>OPM + Repeatability 분석 (기본)</td></tr>
            <tr><td>2 (Quadratic)</td><td>Tilt + Bow</td><td>수동 탐색용</td></tr>
            <tr><td>3+</td><td>고차 Waviness</td><td>특수 분석</td></tr>
            </table>

            <p class='note'>Undo/Redo로 이전 상태 복구 가능합니다.</p>
            """),

            ("5. Saturation Trend", f"""{s}
            <h2>Saturation Trend</h2>
            <p>Repeat 수 증가에 따른 Rep. 1σ Mean 추이를 보여줍니다.</p>

            <h3>읽는 방법</h3>
            <ul>
            <li>그래프가 <b>수렴</b>하면 현재 Repeat 수가 충분합니다.</li>
            <li>아직 하강 추세이면 Repeat를 더 늘려야 합니다.</li>
            <li>초기 값이 매우 높다가 급감하는 경우, 첫 Repeat에 이상이 있을 수 있습니다.</li>
            </ul>

            <h3>판단 기준</h3>
            <ul>
            <li><b>안정화 도달</b>: 마지막 3~4개 Window에서 값 변동 &lt; 10%</li>
            <li><b>추가 Repeat 필요</b>: 여전히 하강 중이거나 변동폭이 큰 경우</li>
            </ul>
            """),

            ("6. Wafer Map", f"""{s}
            <h2>Wafer Map</h2>
            <p>3×3 Grid로 각 Position의 OPM Max를 Heatmap으로 표시합니다.</p>

            <h3>색상 해석</h3>
            <ul>
            <li><b style='color:#f38ba8'>빨간색</b>: 높은 값 (편차 큼) → 해당 위치 점검 필요</li>
            <li><b style='color:#a6e3a1'>녹색</b>: 낮은 값 (편차 작음) → 양호</li>
            </ul>

            <h3>패턴 분석</h3>
            <ul>
            <li>특정 영역에 빨간색이 몰려있으면 Stage의 기계적 문제를 의심합니다.</li>
            <li>Edge vs Center 편차가 크면 Stage Flatness 점검이 필요합니다.</li>
            <li>비대칭 패턴은 Stage 정렬(Alignment) 문제를 시사합니다.</li>
            </ul>
            """),

            ("7. Best-5 Window", f"""{s}
            <h2>Best-5 Window</h2>
            <p>연속된 5개(기본) Repeat 구간 중 최적 구간을 찾습니다.</p>

            <h3>선정 기준</h3>
            <ul>
            <li><b>Rep. 1σ Mean이 최소</b>인 연속 구간</li>
            <li>좌측 패널의 <b>Window Size</b>를 변경하면 즉시 재분석됩니다.</li>
            </ul>

            <h3>그래프 해석</h3>
            <ul>
            <li>그래프: Best Window vs All Repeats의 Position별 비교</li>
            <li>Best Window의 Rep. Max / 1σ가 전체보다 작으면 초기 불안정 Repeat가 있었음을 의미합니다.</li>
            </ul>

            <h3>활용</h3>
            <p>Spec 판정은 Best-5 Window 기준으로 수행됩니다.
            이를 통해 장비 안정화 전 초기 측정의 영향을 배제합니다.</p>
            """),

            ("8. Time Analysis", f"""{s}
            <h2>Time Analysis</h2>
            <p>측정 소요 시간을 분석합니다.</p>

            <h3>표시 항목</h3>
            <ul>
            <li>Repeat별 Start / End / Duration 및 포인트당 소요 시간</li>
            <li>연속 측정 여부 확인 (Gap 2분 이상이면 중단이 있었음)</li>
            </ul>

            <h3>시간 추정</h3>
            <ul>
            <li>하단에서 N-repeat에 필요한 <b>예상 소요 시간</b>을 추정합니다.</li>
            <li class='note'>공수 반영 시 활용: 10-repeat는 약 2배, 20-repeat는 약 4배 소요</li>
            </ul>
            """),

            ("9. Ball Screw Pitch", f"""{s}
            <h2>Ball Screw Pitch</h2>
            <p>Sliding Stage Ball Screw의 Pitch 편차를 분석합니다.</p>

            <h3>사용 방법</h3>
            <ul>
            <li><b>Material</b> 선택 (AL / SUS) → Ball Screw 재질에 따른 기준 Pitch 변경</li>
            <li><b>Exclude Stabilization</b> 체크 → Point 1 (안정화 측정) 제외</li>
            <li><b>Analyze</b> 클릭 → 분석 실행</li>
            </ul>

            <h3>결과 해석</h3>
            <ul>
            <li><b>Bar Chart</b>: Position별 Pitch 편차 분포</li>
            <li><b>Heatmap</b>: Position × Repeat 매트릭스</li>
            <li><b>Verdict</b>: 기준 이내면 PASS, 초과 시 FAIL</li>
            </ul>
            """),

            ("10. Resolution Compare", f"""{s}
            <h2>Resolution Compare</h2>
            <p>서로 다른 Scan Range(25/10/5/1mm)의 OPM을 <b>공정하게</b> 비교하기 위한 기능입니다.</p>

            <h3>왜 필요한가?</h3>
            <p>모든 Range는 동일하게 <b>8192 pixels</b>로 측정되지만, 스캔 범위가 다르므로
            픽셀당 해상도(nm/px)가 크게 달라집니다:</p>
            <table>
            <tr><th>Range</th><th>Scan Size</th><th>Pixels</th><th>Resolution</th></tr>
            <tr><td>25mm</td><td>25,000 µm</td><td>8,192</td><td><b>3,052 nm/px</b></td></tr>
            <tr><td>10mm</td><td>10,000 µm</td><td>8,192</td><td><b>1,221 nm/px</b></td></tr>
            <tr><td>5mm</td><td>5,000 µm</td><td>8,192</td><td><b>610 nm/px</b></td></tr>
            <tr><td>1mm</td><td>1,000 µm</td><td>8,192</td><td><b>122 nm/px</b></td></tr>
            </table>
            <p>해상도가 높을수록(1mm) 미세한 요철이 더 잘 보이므로 OPM이 자연스럽게 높아집니다.
            따라서 <b>Range 간 OPM을 단순 비교하면 불공평</b>합니다.</p>

            <h3>정규화 원리</h3>
            <ul>
            <li>가장 낮은 해상도(25mm의 3,052 nm/px)를 기준으로 선택</li>
            <li>고해상도 프로파일을 <b>Block Averaging</b>으로 다운샘플링하여 동일 해상도로 맞춤</li>
            <li>예: 1mm(122nm/px) → factor 25 → 8192÷25 ≈ 328 pixels로 축소</li>
            <li>축소된 프로파일에서 OPM을 다시 계산 → <b>Normalized OPM</b></li>
            </ul>

            <h3>차트 읽는 방법</h3>
            <ul>
            <li><b>좌측 (Original OPM)</b>: 실제 측정 해상도 그대로의 OPM Max</li>
            <li><b>우측 (Normalized OPM)</b>: 동일 해상도(3,052 nm/px)로 정규화한 OPM Max</li>
            </ul>

            <h3>결과 해석</h3>
            <table>
            <tr><th>현상</th><th>의미</th></tr>
            <tr><td>정규화 후 1mm OPM이 <b>크게 감소</b></td>
                <td>원래 높았던 이유는 <b>고해상도 효과</b> → Stage 자체는 양호</td></tr>
            <tr><td>정규화 후에도 1mm OPM이 <b>여전히 높음</b></td>
                <td><b>실제 Stage 문제</b>일 가능성 → 추가 점검 필요</td></tr>
            <tr><td>모든 Range의 Normalized OPM이 <b>비슷</b></td>
                <td>Stage 성능이 전 Range에서 균일 → 이상적인 상태</td></tr>
            </table>

            <p class='note'>⚠ 데이터 루트 폴더(2개 이상 Range 포함)를 로드해야 차트가 표시됩니다.
            단일 Range만 로드하면 비교 대상이 없으므로 빈 상태가 유지됩니다.</p>
            """),

            ("11. QC Check", f"""{s}
            <h2>QC Check — Data Collection Quality Control</h2>
            <p>측정 데이터의 무결성을 6개 항목으로 자동 검증합니다.</p>

            <h3>사용 방법</h3>
            <ul>
            <li>1. 데이터 로드 후 <b>QC Check</b> 탭 선택</li>
            <li>2. <b>Run QC</b> 버튼 클릭</li>
            <li>3. Summary 패널에서 6개 항목의 PASS / WARN / FAIL 확인</li>
            <li>4. Detail ComboBox에서 항목 선택 → 상세 결과 테이블 확인</li>
            </ul>

            <h3>검사 항목</h3>
            <table>
            <tr><th>항목</th><th>검증 내용</th><th>검출 사례</th></tr>
            <tr><td><b>QC-1: File Matching</b></td>
                <td>Recipe TIFF와 Raw TIFF 파일 1:1 매칭</td>
                <td>저장 오류, 디스크 용량 부족</td></tr>
            <tr><td><b>QC-2: Data Equivalence</b></td>
                <td>Order-2 Flatten 후 분석 결과 동일성</td>
                <td>파일 손상, Recipe 설정 오류</td></tr>
            <tr><td><b>QC-3: Scan Parameters</b></td>
                <td>Z Sensitivity, Scan Size 등 일관성</td>
                <td>Recipe 변경 후 부분 재측정</td></tr>
            <tr><td><b>QC-4: Position Completeness</b></td>
                <td>9개 Position × N Repeat 완전성</td>
                <td>측정 중단, 팁 파손</td></tr>
            <tr><td><b>QC-5: Outlier Detection</b></td>
                <td>Median ± 3×MAD 기준 이상치 탐지</td>
                <td>Stage 진동, 팁 오염</td></tr>
            <tr><td><b>QC-6: Pixel Count</b></td>
                <td>프로파일 데이터 포인트 수 (8192)</td>
                <td>비정상 종료, 파일 손상</td></tr>
            </table>

            <h3>판정 기준</h3>
            <ul>
            <li><span class='pass'>PASS</span>: 모든 항목 정상 — 데이터 신뢰 가능</li>
            <li><span class='warn'>WARN</span>: 이상치 감지 등 주의 필요 — 데이터 사용 가능하나 확인 권장</li>
            <li><span class='fail'>FAIL</span>: 데이터 무결성 문제 — 원인 확인 후 재측정 고려</li>
            </ul>

            <h3>QC-5 Outlier Detection 상세</h3>
            <p>MAD(Median Absolute Deviation) 기반 이상치 탐지를 사용합니다.</p>
            <ul>
            <li><b>기준</b>: |OPM - Median| > 3 × 1.4826 × MAD</li>
            <li>1.4826은 MAD를 정규분포 σ와 일관되게 하는 보정 계수입니다.</li>
            <li>Repeat 3회 미만 시 탐지를 스킵합니다.</li>
            </ul>

            <h3>Overall 판정</h3>
            <ul>
            <li>6개 항목 중 하나라도 <span class='fail'>FAIL</span>이면 Overall = <span class='fail'>FAIL</span></li>
            <li>FAIL 없이 <span class='warn'>WARN</span>이 있으면 Overall = <span class='warn'>WARN</span></li>
            <li>모두 <span class='pass'>PASS</span>이면 Overall = <span class='pass'>PASS</span></li>
            </ul>
            """),

            ("12. Compare", f"""{s}
            <h2>Compare — Cross-Process Comparison</h2>
            <p>동일 모듈의 다른 공정(모듈 조립 vs 완제품) 테스트 결과를 비교합니다.</p>

            <h3>사용 방법</h3>
            <ul>
            <li>1. 현재 데이터를 먼저 로드 (Open Folder)</li>
            <li>2. Compare 탭에서 <b>Load Reference</b> 클릭</li>
            <li>3. 비교 대상 데이터 폴더 선택</li>
            <li>4. Position별 Δ(차이) 테이블 자동 표시</li>
            </ul>

            <h3>테이블 해석</h3>
            <table>
            <tr><th>열</th><th>의미</th></tr>
            <tr><td>Curr OPM Max</td><td>현재 데이터 OPM Max</td></tr>
            <tr><td>Ref OPM Max</td><td>Reference 데이터 OPM Max</td></tr>
            <tr><td>Δ OPM (nm)</td><td>차이값 (빨강=악화, 초록=개선)</td></tr>
            <tr><td>Δ OPM (%)</td><td>변화율</td></tr>
            </table>

            <h3>활용</h3>
            <ul>
            <li>모듈 조립 → 완제품 간 OPM 변화 추적</li>
            <li>Edge/Side/Center 그룹별 변화 패턴 분석</li>
            <li>특정 Position에서 큰 편차 → 조립 공정 문제 시사</li>
            </ul>
            """),

            ("Spec Judgment", f"""{s}
            <h2>Spec Judgment (좌측 패널)</h2>
            <p>장비 타입에 따라 다른 기준으로 PASS/FAIL을 판단합니다.</p>

            <h3>장비 타입별 판정 기준</h3>
            <table>
            <tr><th>타입</th><th>OPM Repeatability 기준</th><th>Max OPM 기준</th></tr>
            <tr><td><b>분리형 (Isolated AE)</b></td>
                <td>Total RMS of Rep. 1σ</td><td>Total Max of OPM Max</td></tr>
            <tr><td><b>일체형 (Double Walled AE)</b></td>
                <td>Center(5_CM) Rep. 1σ</td><td>Center(5_CM) OPM Max</td></tr>
            </table>

            <h3>Spec 한도 (OPM Repeatability)</h3>
            <table>
            <tr><th>Range</th><th>한도 (nm)</th></tr>
            <tr><td>25 mm</td><td>12.9</td></tr>
            <tr><td>10 mm</td><td>5.6</td></tr>
            <tr><td>5 mm</td><td>3.3</td></tr>
            <tr><td>1 mm</td><td>1.6</td></tr>
            </table>

            <h3>합격 조건</h3>
            <ul>
            <li>OPM Repeatability + Max OPM <b>두 항목 모두 PASS</b>해야 합격</li>
            <li><b>?</b> 버튼을 클릭하면 전체 Spec 테이블을 확인할 수 있습니다.</li>
            </ul>
            """),
        ]

    # ─── Data Loading ────────────────────────────────────────

    def _on_load_clicked(self):
        start_dir = self._loaded_path or ""
        if not start_dir:
            for candidate in [Path("data"), Path(".")]:
                if candidate.is_dir():
                    start_dir = str(candidate.resolve())
                    break

        folder = QFileDialog.getExistingDirectory(
            self, "Select Data Folder (root or recipe)", start_dir)
        if not folder:
            return
        self._start_load(folder)

    def _start_load(self, folder: str):
        self._cleanup_temp()
        path = Path(folder)

        # DLP bypass: if iterdir() raises PermissionError, copy via robocopy
        if self._needs_local_copy(path):
            self.load_btn.setEnabled(False)
            self.statusBar().showMessage("Copying from server (DLP bypass)...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)
            self._copy_worker = CopyWorker(folder)
            self._copy_worker.finished.connect(self._on_copy_finished)
            self._copy_worker.error.connect(self._on_load_error)
            self._copy_worker.start()
            return

        self._start_load_local(folder)

    def _needs_local_copy(self, path: Path) -> bool:
        """Check if path triggers PermissionError (DLP-blocked network path)."""
        try:
            next(path.iterdir())
            return False
        except PermissionError:
            return True
        except StopIteration:
            return False

    def _on_copy_finished(self, local_path: str):
        """After robocopy completes, load from local temp path."""
        self._temp_data_dir = local_path
        self.statusBar().showMessage("Server copy complete. Loading data...")
        self._start_load_local(local_path)

    def _cleanup_temp(self):
        """Remove temporary data directory from previous server copy."""
        if hasattr(self, '_temp_data_dir') and self._temp_data_dir:
            shutil.rmtree(self._temp_data_dir, ignore_errors=True)
            self._temp_data_dir = None

    def _start_load_local(self, folder: str):
        path = Path(folder)
        folder_type = _detect_folder_type(path)

        if folder_type == "unknown":
            parent_type = _detect_folder_type(path.parent)
            if parent_type == "recipe":
                QMessageBox.warning(
                    self, "Wrong folder level",
                    f"Selected a repeat folder:\n{path.name}\n\n"
                    f"Please select the recipe folder:\n{path.parent}\n\n"
                    f"Or select the root data folder to load all recipes."
                )
                self.load_btn.setEnabled(True)
                self.progress_bar.setVisible(False)
                return
            else:
                QMessageBox.warning(
                    self, "Unrecognized folder",
                    f"Could not detect recipe data in:\n{folder}\n\n"
                    f"Expected folder names containing a range pattern like\n"
                    f"'25mm', '10mm', '5mm', '1mm' (e.g., 'Profile_25mm_Dynamic').\n"
                    f"Recipe folders may be up to 3 levels below the selected folder."
                )
                self.load_btn.setEnabled(True)
                self.progress_bar.setVisible(False)
                return

        self._loaded_path = folder
        signal = self.source_combo.currentText()
        is_multi = (folder_type == "root")

        self.load_btn.setEnabled(False)
        self.statusBar().showMessage(f"Loading {'all recipes' if is_multi else 'recipe'}...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self._worker = LoadWorker(folder, signal, multi=is_multi)
        self._worker.finished_single.connect(self._on_single_loaded)
        self._worker.finished_multi.connect(self._on_multi_loaded)
        self._worker.error.connect(self._on_load_error)
        self._worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self._worker.start()

    def _on_single_loaded(self, recipe: RecipeData):
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        from ..core.data_loader import DataSet
        self.dataset = DataSet(
            root_directory=recipe.directory.parent,
            recipes={recipe.range_label: recipe}
        )
        self._populate_range_selector()
        self.path_label.setText(
            f"{recipe.directory} — {recipe.range_label} ({recipe.repeat_count} repeats)")

    def _on_multi_loaded(self, dataset: DataSet):
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if not dataset.recipes:
            QMessageBox.warning(self, "No Data", "No recipe data found.")
            return

        self.dataset = dataset
        self._populate_range_selector()

        total = sum(r.repeat_count for r in dataset.recipes.values())
        ranges = ", ".join(dataset.available_ranges)
        self.path_label.setText(f"{dataset.root_directory} — {ranges} ({total} total repeats)")

    def _on_load_error(self, msg: str):
        self.load_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Load Error", msg)

    def closeEvent(self, event):
        self._cleanup_temp()
        super().closeEvent(event)

    def _populate_range_selector(self):
        self._block_range_signal = True
        self.range_combo.clear()
        if self.dataset:
            for label in self.dataset.available_ranges:
                recipe = self.dataset.recipes[label]
                self.range_combo.addItem(f"{label} ({recipe.repeat_count} repeats)")
        self._block_range_signal = False

        if self.range_combo.count() > 0:
            self.range_combo.setCurrentIndex(0)
            self._switch_to_current_range()

    def _on_range_changed(self, text: str):
        if self._block_range_signal or not text:
            return
        self._switch_to_current_range()

    def _switch_to_current_range(self):
        if not self.dataset:
            return

        combo_text = self.range_combo.currentText()
        range_label = combo_text.split(" (")[0] if " (" in combo_text else combo_text

        if range_label not in self.dataset.recipes:
            return

        self.current_recipe = self.dataset.recipes[range_label]
        self.range_info_label.setText(
            f"{self.current_recipe.repeat_count} repeats, "
            f"{sum(len(r.profiles) for r in self.current_recipe.repeats)} profiles")

        self.flat_rep_combo.clear()
        self.flat_rep_combo.addItems(
            [f"Repeat {r.repeat_no}" for r in self.current_recipe.repeats])

        self._update_info_tree()
        self._run_analysis()

    def _on_source_changed(self, source: str):
        if self._loaded_path and self.dataset:
            reply = QMessageBox.question(
                self, "Reload Data?",
                f"Signal source changed to '{source}'.\nReload all data?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self._start_load(self._loaded_path)

    def _on_equipment_changed(self):
        """Re-analyze when equipment type radio is toggled."""
        if self.current_recipe:
            self._run_analysis()

    def _show_spec_info_popup(self):
        """Show spec table popup dialog."""
        from PySide6.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QLabel, QPushButton, QHeaderView
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor
        from ..core.analyzer import SPEC_REPEATABILITY, SPEC_MAX_OPM_DW, SPEC_MAX_OPM_ISO

        dlg = QDialog(self)
        dlg.setWindowTitle("Spec Reference Table")
        dlg.setMinimumSize(600, 400)
        dlg.setStyleSheet(
            "QDialog { background-color: #1e1e2e; }"
            "QLabel { color: #cdd6f4; }"
            "QTableWidget { background-color: #181825; color: #cdd6f4;"
            "gridline-color: #313244; border: 1px solid #45475a; font-size: 13px; }"
            "QTableWidget::item { padding: 6px; }"
            "QHeaderView::section { background-color: #313244; color: #89b4fa;"
            "padding: 8px; border: 1px solid #45475a; font-weight: bold; font-size: 13px; }")

        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("Sliding Stage Spec Limits")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #89b4fa;")
        layout.addWidget(title)

        equip_type = "iso" if self.radio_iso.isChecked() else "dw"
        equip_label = "Isolated AE" if equip_type == "iso" else "Double Walled AE"
        current_label = QLabel(f"Current: {equip_label}")
        current_label.setStyleSheet("font-size: 12px; color: #a6adc8; margin-bottom: 8px;")
        layout.addWidget(current_label)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels([
            "Range", "OPM\nRepeatability",
            "Max OPM\n(Double Walled)", "Max OPM\n(Isolated)", "Basis"])
        table.horizontalHeader().setMinimumHeight(44)

        ranges = [25, 10, 5, 1]
        table.setRowCount(len(ranges))
        for i, mm in enumerate(ranges):
            table.setItem(i, 0, QTableWidgetItem(f"{mm}mm"))
            table.setItem(i, 1, QTableWidgetItem(f"{SPEC_REPEATABILITY.get(mm, 'N/A')} nm"))
            table.setItem(i, 2, QTableWidgetItem(f"{SPEC_MAX_OPM_DW.get(mm, 'N/A')} nm"))
            table.setItem(i, 3, QTableWidgetItem(f"{SPEC_MAX_OPM_ISO.get(mm, 'N/A')} nm"))

            if equip_type == "dw":
                basis = "Center (5_CM)"
            else:
                basis = "Total RMS / Max"
            table.setItem(i, 4, QTableWidgetItem(basis))

            for j in range(5):
                item = table.item(i, j)
                if item:
                    item.setTextAlignment(Qt.AlignCenter)

        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        layout.addWidget(table)

        # Note
        note = QLabel(
            "\u2022 OPM Repeatability: Based on Rep. 1\u03c3 (DW=Center, ISO=Total RMS)\n"
            "\u2022 Max OPM: Based on max OPM value (DW=Center, ISO=Total Max)\n"
            "\u2022 Both items must PASS to qualify")
        note.setStyleSheet("font-size: 11px; color: #a6adc8; padding: 8px;")
        layout.addWidget(note)

        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            "QPushButton { background-color: #45475a; color: #cdd6f4;"
            "padding: 8px 24px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background-color: #585b70; }")
        close_btn.clicked.connect(dlg.close)
        layout.addWidget(close_btn, alignment=Qt.AlignRight)

        dlg.exec()

    def _on_outlier_mode_changed(self, text: str):
        is_active = text != "None"
        self.outlier_value_spin.setEnabled(is_active)
        if text == "Pixels":
            self.outlier_value_spin.setSuffix("")
            self.outlier_value_spin.setDecimals(0)
            self.outlier_value_spin.setRange(0, 9999)
            self.outlier_value_spin.setValue(10)
        elif text == "Percentile":
            self.outlier_value_spin.setSuffix(" %")
            self.outlier_value_spin.setDecimals(1)
            self.outlier_value_spin.setRange(0.0, 100.0)
            self.outlier_value_spin.setValue(1.0)
        self._on_reanalyze()

    def _on_reanalyze(self):
        if self.current_recipe:
            self._run_analysis()

    # ─── Analysis ────────────────────────────────────────────

    def _run_analysis(self):
        if not self.current_recipe:
            return

        window_size = self.window_spin.value()
        equipment_type = "iso" if self.radio_iso.isChecked() else "dw"
        # Raw = the true measurement (no outlier exclusion) — the honest headline
        # used by all charts / wafer map / detail dialog.
        self.current_result = analyze_recipe(
            self.current_recipe, window_size=window_size,
            equipment_type=equipment_type,
            outlier_mode="none", outlier_value=0.0)
        # Robust companion = outlier-excluded preset from the UI (default Percentile
        # 1%), shown ALONGSIDE the raw value. "None" -> no companion (mirrors raw).
        robust_mode = self.outlier_mode_combo.currentText().lower()
        if robust_mode == "none":
            self.current_result_robust = None
        else:
            self.current_result_robust = analyze_recipe(
                self.current_recipe, window_size=window_size,
                equipment_type=equipment_type,
                outlier_mode=robust_mode,
                outlier_value=self.outlier_value_spin.value())

        # Time analysis
        self.current_timing = extract_recipe_timing(self.current_recipe)

        self._update_summary_table()
        self._update_spec_display()
        self._update_scan_info()
        self._update_res_slider_range()
        self._update_profile_chart()
        self._update_trend_chart()
        self._update_wafer_map()
        self._update_best5_chart()
        self._update_time_tab()
        self._update_resolution_compare()

        # Reset Ball Screw result when recipe changes (requires explicit Analyze click)
        self.current_bs_result = None
        self._clear_bs_tab()
        # Reset QC result (requires explicit Run QC click)
        self.current_qc_result = None
        self._clear_qc_tab()
        # Reset Compare result
        self.current_compare_result = None
        self._clear_compare_tab()

        self.export_btn.setEnabled(True)

        bw = self.current_result.best_window
        spec_text = ""
        if self.current_result.spec_pass is not None:
            spec_text = "PASS" if self.current_result.spec_pass else "FAIL"
        if bw:
            self.statusBar().showMessage(
                f"Analysis: {self.current_result.range_label} | "
                f"Best: R{bw.repeat_range} | "
                f"Rep.1σ: {bw.mean_rep_1sigma:.3f}nm | {spec_text}")
        else:
            self.statusBar().showMessage(
                f"Analysis: {self.current_result.range_label} | {spec_text}")

    def _update_info_tree(self):
        self.info_tree.clear()
        if not self.current_recipe:
            return
        r = self.current_recipe
        self.info_tree.addTopLevelItem(QTreeWidgetItem(["Range", r.range_label]))
        self.info_tree.addTopLevelItem(QTreeWidgetItem(["Repeats", str(r.repeat_count)]))
        self.info_tree.addTopLevelItem(QTreeWidgetItem([
            "Source", self.source_combo.currentText()]))
        for rep in r.repeats:
            ritem = QTreeWidgetItem([f"Repeat {rep.repeat_no}", rep.directory.name])
            ritem.addChild(QTreeWidgetItem(["Profiles", str(len(rep.profiles))]))
            if rep.lot_id:
                ritem.addChild(QTreeWidgetItem(["Lot ID", rep.lot_id]))
            self.info_tree.addTopLevelItem(ritem)

    # ─── Chart Updates ───────────────────────────────────────

    @staticmethod
    def _update_canvas(canvas: FigureCanvas, new_fig: Figure):
        """Safely replace figure, preventing rendering ghosts."""
        old_fig = canvas.figure
        if old_fig is not new_fig:
            plt.close(old_fig)
        new_fig.set_canvas(canvas)
        canvas.figure = new_fig
        # Sync DPI to the canvas' logical DPI → point-sized fonts render at a
        # consistent physical size across displays with different scaling.
        target_dpi = canvas.logicalDpiX() or new_fig.get_dpi()
        new_fig.set_dpi(target_dpi)
        w, h = canvas.width(), canvas.height()
        if w > 0 and h > 0:
            new_fig.set_size_inches(w / target_dpi, h / target_dpi)
        canvas.draw_idle()
        canvas.update()

    def _update_summary_table(self):
        if not self.current_result:
            return
        rows = get_dual_summary_table(
            self.current_result, self.current_result_robust, use_best_window=True)
        self.summary_table.setRowCount(len(rows))
        metric_keys = ["Rep. Max (nm)", "Rep. 1σ (nm)", "OPM Max (nm)", "OPM 1σ (nm)"]
        cols = ["Range", "Position"] + metric_keys
        for i, row in enumerate(rows):
            is_total = row.get("Range") == "Total"
            is_group = row.get("Range") == "Group"
            for j, key in enumerate(cols):
                differ = False
                if key in metric_keys:
                    raw = row.get(key, "")
                    rob = row.get(f"{key} (rob)", raw)
                    # Show raw on top and robust below only where they differ.
                    differ = (self.current_result_robust is not None
                              and rob != "-" and str(rob) != str(raw))
                    text = f"{raw}\n{rob}" if differ else f"{raw}"
                else:
                    text = str(row.get(key, ""))
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if is_total:
                    item.setBackground(QColor("#313244"))
                    item.setFont(QFont("Segoe UI", 11, QFont.Bold))
                elif is_group:
                    item.setBackground(QColor("#1e1e2e"))
                    item.setForeground(QColor("#f9e2af"))
                    item.setFont(QFont("Segoe UI", 11, QFont.Bold))
                if differ and not is_group:
                    item.setForeground(QColor("#f9e2af"))  # raw≠robust -> highlight
                self.summary_table.setItem(i, j, item)

    def _update_spec_display(self):
        """Update spec judgment with dual-spec values and overall verdict."""
        if not self.current_result:
            return

        r = self.current_result
        equip_label = "Isolated AE" if r.equipment_type == "iso" else "Double Walled AE"
        basis = "Total RMS / Max" if r.equipment_type == "iso" else "Center (5_CM)"

        lines = []
        lines.append(f"<b>Type:</b> {equip_label}")
        lines.append(f"<b>Basis:</b> {basis}")

        if r.best_window:
            lines.append(f"<b>Window:</b> R{r.best_window.repeat_range}")

        rr = self.current_result_robust  # outlier-excluded companion (may be None)

        # OPM Repeatability (Rep. 1\u03c3) \u2014 raw, with robust shown alongside
        if r.spec_limit is not None:
            val_str = f"{r.spec_value:.3f}" if r.spec_value is not None else "N/A"
            pass_icon = "\u2705" if r.spec_pass else "\u274c"
            line = f"{pass_icon} <b>Rep. 1\u03c3:</b> {val_str} / {r.spec_limit} nm"
            if rr is not None and rr.spec_value is not None:
                rob_icon = "\u2705" if rr.spec_pass else "\u274c"
                line += (f"  <span style='color:#a6adc8'>(robust {rob_icon} "
                         f"{rr.spec_value:.3f})</span>")
            lines.append("")
            lines.append(line)

        # Max OPM \u2014 raw, with robust shown alongside
        if r.spec_opm_limit is not None:
            val_str = f"{r.spec_opm_value:.3f}" if r.spec_opm_value is not None else "N/A"
            pass_icon = "\u2705" if r.spec_opm_pass else "\u274c"
            line = f"{pass_icon} <b>OPM Max:</b> {val_str} / {r.spec_opm_limit} nm"
            if rr is not None and rr.spec_opm_value is not None:
                rob_icon = "\u2705" if rr.spec_opm_pass else "\u274c"
                line += (f"  <span style='color:#a6adc8'>(robust {rob_icon} "
                         f"{rr.spec_opm_value:.3f})</span>")
            lines.append(line)

        # Flag when raw vs robust verdict disagrees (= an outlier flipped the result).
        if rr is not None and r.overall_pass is not None and rr.overall_pass is not None \
                and r.overall_pass != rr.overall_pass:
            lines.append("")
            lines.append("<span style='color:#f9e2af'>\u26a0 Raw/Robust \ud310\uc815 \ubd88\uc77c\uce58 "
                         "\u2014 outlier \uc601\ud5a5 (QC-5 \ud655\uc778)</span>")

        self.spec_lines_label.setText("<br>".join(lines))

        # Overall verdict
        overall = r.overall_pass
        if overall is not None:
            if overall:
                self.spec_verdict_label.setText("PASS")
                self.spec_verdict_label.setStyleSheet(
                    "font-size: 18px; font-weight: bold; color: #a6e3a1; padding: 4px;")
            else:
                self.spec_verdict_label.setText("FAIL")
                self.spec_verdict_label.setStyleSheet(
                    "font-size: 18px; font-weight: bold; color: #f38ba8; padding: 4px;")
        else:
            self.spec_verdict_label.setText("\u2014")
            self.spec_verdict_label.setStyleSheet(
                "font-size: 18px; font-weight: bold; padding: 4px;")

    def _update_scan_info(self):
        """Update Scan Parameters panel from current recipe."""
        if not self.current_recipe or not self.current_recipe.repeats:
            self.scan_info_label.setText("—")
            return

        repeat = self.current_recipe.repeats[0]
        recipe_name = getattr(repeat, 'recipe_id', '') or ''
        # Extract short recipe name (last segment of path)
        if '\\' in recipe_name:
            recipe_name = recipe_name.rsplit('\\', 1)[-1]

        # Get scan params from first available profile
        profile = None
        for pos in POSITION_LABELS:
            if pos in repeat.profiles:
                profile = repeat.profiles[pos]
                break

        if profile is None:
            self.scan_info_label.setText(f"<b>Recipe:</b> {recipe_name}")
            return

        px_count = len(profile.raw_data)
        res_nm = profile.scan_size_um * 1000 / px_count if px_count > 0 else 0

        lines = [
            f"<b>Recipe:</b> {recipe_name}",
            f"<b>Size:</b> {profile.scan_size_um:.0f} µm &nbsp; "
            f"<b>Px:</b> {px_count}",
            f"<b>Resolution:</b> {res_nm:.1f} nm/px",
            f"<b>Speed:</b> {profile.scan_speed_mm_s:.3f} mm/s &nbsp; "
            f"<b>SP:</b> {profile.set_point:.1f}",
            f"<b>Z Gain:</b> {profile.z_servo_gain:.1f}",
        ]
        self.scan_info_label.setText("<br>".join(lines))

    def _get_scan_info_dict(self) -> dict:
        """Get scan info dict for chart suptitle."""
        if not self.current_recipe or not self.current_recipe.repeats:
            return {}
        repeat = self.current_recipe.repeats[0]
        for pos in POSITION_LABELS:
            if pos in repeat.profiles:
                p = repeat.profiles[pos]
                px_count = len(p.raw_data)
                return {
                    "range_label": self.current_recipe.range_label,
                    "pixels": px_count,
                    "resolution_nm": p.scan_size_um * 1000 / px_count if px_count else 0,
                    "speed": p.scan_speed_mm_s,
                    "set_point": p.set_point,
                }
        return {}

    def _on_res_slider_changed(self, value):
        """Update label immediately, debounce chart redraw."""
        if not self.current_recipe:
            return
        if value <= 1:
            self.res_slider_label.setText("Original")
        else:
            scan_info = self._get_scan_info_dict()
            orig_res = scan_info.get("resolution_nm", 0)
            sim_res = orig_res * value
            self.res_slider_label.setText(f"{sim_res:.0f} nm/px (×{value})")
        self._res_debounce.start()

    def _update_res_slider_range(self):
        """Update resolution slider range based on current recipe."""
        if not self.current_recipe:
            return
        scan_info = self._get_scan_info_dict()
        orig_res = scan_info.get("resolution_nm", 0)
        if orig_res <= 0:
            return
        # Max factor: simulate up to ~3052 nm/px (25mm equivalent)
        max_factor = max(1, int(3052 / orig_res))
        self.res_slider.blockSignals(True)
        self.res_slider.setMinimum(1)
        self.res_slider.setMaximum(max_factor)
        self.res_slider.setValue(1)
        self.res_slider.blockSignals(False)
        self.res_slider_label.setText("Original")

    def _update_profile_chart(self):
        if not self.current_recipe:
            return
        scan_info = self._get_scan_info_dict()
        y_mode = self.y_scale_combo.currentText().lower()
        sim_factor = self.res_slider.value()
        fig = create_profile_overlay_figure(self.current_recipe, figsize=(12, 9),
                                            scan_info=scan_info,
                                            y_scale_mode=y_mode,
                                            sim_factor=sim_factor)
        self._update_canvas(self.profile_canvas, fig)

        # Store axes → position mapping for double-click
        self._profile_axes_map = {}
        axes = fig.get_axes()
        for pos in POSITION_LABELS:
            r, c = POSITION_GRID[pos]
            idx = r * 3 + c
            if idx < len(axes):
                self._profile_axes_map[id(axes[idx])] = pos

        # Connect double-click (disconnect old handler first to prevent accumulation)
        if hasattr(self, '_profile_cid') and self._profile_cid is not None:
            self.profile_canvas.mpl_disconnect(self._profile_cid)
        self._profile_cid = self.profile_canvas.mpl_connect(
            'button_press_event', self._on_profile_dblclick)

    def _on_profile_dblclick(self, event):
        """Open Position Detail Dialog on double-click."""
        if not event.dblclick or event.inaxes is None:
            return
        if not hasattr(self, '_profile_axes_map'):
            return
        pos = self._profile_axes_map.get(id(event.inaxes))
        if pos and self.current_recipe:
            from .position_detail_dialog import PositionDetailDialog
            dlg = PositionDetailDialog(
                pos, self.current_recipe, self.current_result, parent=self)
            dlg.exec()

    def _update_trend_chart(self):
        if not self.current_result:
            return
        fig = create_saturation_trend_figure(self.current_result, figsize=(10, 6))
        self._update_canvas(self.trend_canvas, fig)

    def _update_wafer_map(self):
        if not self.current_result:
            return
        fig = create_wafer_map_figure(self.current_result, metric="rep_max", figsize=(8, 7))
        self._update_canvas(self.wafer_canvas, fig)

    def _update_best5_chart(self):
        if not self.current_result:
            return
        fig = create_best5_comparison_figure(self.current_result, figsize=(12, 6))
        self._update_canvas(self.best5_canvas, fig)

    # ─── Time Analysis ───────────────────────────────────────

    def _update_time_tab(self):
        """Populate Time Analysis tab with timing data."""
        if not self.current_timing:
            return

        t = self.current_timing

        # Summary labels
        td = t.total_duration
        self.time_total_label.setText(t.total_duration_str if td else "—")

        avg_rep = t.avg_repeat_duration
        if avg_rep:
            from ..core.time_analysis import _fmt_duration
            self.time_avg_repeat_label.setText(_fmt_duration(avg_rep))
        else:
            self.time_avg_repeat_label.setText("—")

        avg_pt = t.avg_per_point_sec
        if avg_pt:
            m, s = divmod(int(avg_pt), 60)
            self.time_avg_point_label.setText(f"{m}m {s:02d}s" if m else f"{s}s")
        else:
            self.time_avg_point_label.setText("—")

        if t.is_continuous:
            self.time_continuous_label.setText("Continuous")
            self.time_continuous_label.setStyleSheet("font-size: 14px; color: #a6e3a1;")
        else:
            self.time_continuous_label.setText("Gaps detected")
            self.time_continuous_label.setStyleSheet("font-size: 14px; color: #f9e2af;")

        # Estimation
        self._update_time_estimate()

        # Table
        rows = format_timing_summary(t)
        self.time_table.setRowCount(len(rows))
        cols = ["Repeat", "Folder", "Start", "End", "Duration", "Per Point", "Gap"]
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                val = row.get(col, "")
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                # Highlight gaps
                if col == "Gap" and val and "⚠" in str(val):
                    item.setForeground(QColor("#f9e2af"))
                self.time_table.setItem(i, j, item)

    def _update_time_estimate(self):
        if not self.current_timing:
            return
        n = self.time_est_spin.value()
        est = self.current_timing.estimate_duration(n)
        self.time_est_result.setText(est)

    # ─── Ball Screw Pitch ─────────────────────────────────────

    def _on_bs_analyze(self):
        """Run Ball Screw Pitch analysis on current recipe."""
        if not self.current_recipe:
            QMessageBox.warning(self, "Warning", "Load data first.")
            return

        material_text = self.bs_material_combo.currentText()
        material = "AL" if material_text.startswith("AL") else "SUS"
        signal_source = self.source_combo.currentText()

        try:
            self.current_bs_result = analyze_ball_screw(
                self.current_recipe, signal_source=signal_source, material=material)
            self._update_bs_tab()
        except Exception as e:
            QMessageBox.critical(self, "Ball Screw Analysis Error", str(e))

    def _on_bs_filter_changed(self):
        """Toggle stabilization point display."""
        if self.current_bs_result:
            self._update_bs_tab()

    def _clear_bs_tab(self):
        """Reset Ball Screw tab to empty state."""
        self.bs_verdict_label.setText("—")
        self.bs_verdict_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; border: 2px solid #45475a;"
            "border-radius: 6px; padding: 4px; color: #a6adc8;")
        self.bs_table.clear()
        self.bs_table.setRowCount(0)
        self.bs_table.setColumnCount(0)
        for canvas in (self.bs_bar_canvas, self.bs_heatmap_canvas):
            old = canvas.figure
            new_fig = Figure(figsize=old.get_size_inches())
            plt.close(old)
            self._update_canvas(canvas, new_fig)

    def _update_bs_tab(self):
        """Refresh all Ball Screw tab visuals from current_bs_result."""
        if not self.current_bs_result:
            return
        bs = self.current_bs_result
        include_stab = self.bs_show_stab_check.isChecked()

        # ── Verdict badge ────────────────────────────────────────────────────
        if bs.overall_pass:
            self.bs_verdict_label.setText("PASS")
            self.bs_verdict_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; border: 2px solid #a6e3a1;"
                "border-radius: 6px; padding: 4px; color: #a6e3a1;")
        else:
            self.bs_verdict_label.setText("FAIL")
            self.bs_verdict_label.setStyleSheet(
                "font-size: 16px; font-weight: bold; border: 2px solid #f38ba8;"
                "border-radius: 6px; padding: 4px; color: #f38ba8;")

        positions, rep_labels, dishing_matrix = get_dishing_matrix(
            bs, include_stabilization=include_stab)
        spec_limit = bs.spec_limit
        n_pos = len(positions)
        n_rep = len(rep_labels)

        # ── Bar chart ────────────────────────────────────────────────────────
        bar_fig = Figure(figsize=(8, 4), facecolor="#1e1e2e")
        ax = bar_fig.add_subplot(111, facecolor="#181825")

        x_pos = np.arange(n_pos)
        bar_w = 0.6
        colors_rep = plt.cm.tab10(np.linspace(0, 0.9, max(n_rep, 1)))

        # Scatter individual repeat values
        for rep_i in range(n_rep):
            vals = dishing_matrix[:, rep_i]
            ax.scatter(x_pos, vals, color=colors_rep[rep_i], s=40, zorder=5,
                       label=rep_labels[rep_i], alpha=0.85)

        # Mean bars
        means = np.nanmean(dishing_matrix, axis=1)
        bar_colors = ["#f38ba8" if v > spec_limit else "#89b4fa" for v in means]
        ax.bar(x_pos, means, width=bar_w, color=bar_colors, alpha=0.35, zorder=3)

        # Spec line
        ax.axhline(spec_limit, color="#f38ba8", linewidth=1.5, linestyle="--",
                   label=f"Spec ({spec_limit} nm)")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(positions, rotation=25, ha="right",
                           color="#cdd6f4", fontsize=9)
        ax.set_ylabel("Dishing (nm)", color="#cdd6f4", fontsize=10)
        ax.set_title(f"Ball Screw Pitch — Dishing per Position [{bs.material}, ≤{spec_limit} nm]",
                     color="#89b4fa", fontsize=11, pad=8)
        ax.tick_params(colors="#cdd6f4", labelsize=9)
        ax.spines[:].set_color("#45475a")
        legend = ax.legend(loc="upper right", fontsize=8,
                           facecolor="#313244", edgecolor="#45475a",
                           labelcolor="#cdd6f4", framealpha=0.85)
        ax.grid(axis="y", color="#313244", linewidth=0.5)
        bar_fig.tight_layout(pad=1.2)
        self._update_canvas(self.bs_bar_canvas, bar_fig)

        # ── Heatmap ──────────────────────────────────────────────────────────
        hm_fig = Figure(figsize=(6, 4), facecolor="#1e1e2e")
        ax2 = hm_fig.add_subplot(111, facecolor="#181825")

        import matplotlib.colors as mcolors
        vmax = max(float(np.nanmax(dishing_matrix)), spec_limit * 1.1)
        vmin = 0.0
        cmap = plt.cm.RdYlGn_r
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        im = ax2.imshow(dishing_matrix.T, aspect="auto", cmap=cmap, norm=norm,
                        origin="upper")
        ax2.set_xticks(range(n_pos))
        ax2.set_xticklabels(positions, rotation=30, ha="right",
                            color="#cdd6f4", fontsize=8)
        ax2.set_yticks(range(n_rep))
        ax2.set_yticklabels(rep_labels, color="#cdd6f4", fontsize=8)
        ax2.tick_params(colors="#cdd6f4")
        ax2.set_title("Dishing Heatmap\n(Position × Repeat)",
                      color="#89b4fa", fontsize=10, pad=6)
        ax2.spines[:].set_color("#45475a")

        # Annotate values + highlight spec failure
        for pos_i in range(n_pos):
            for rep_i in range(n_rep):
                val = dishing_matrix[pos_i, rep_i]
                if not np.isnan(val):
                    txt_color = "white" if val > spec_limit * 0.7 else "black"
                    ax2.text(pos_i, rep_i, f"{val:.2f}",
                             ha="center", va="center",
                             color=txt_color, fontsize=7.5, fontweight="bold")
                    if val > spec_limit:
                        ax2.add_patch(plt.Rectangle(
                            (pos_i - 0.5, rep_i - 0.5), 1, 1,
                            fill=False, edgecolor="#f38ba8", linewidth=2))

        cb = hm_fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="#cdd6f4")
        cb.ax.tick_params(labelcolor="#cdd6f4", labelsize=8)
        cb.set_label("Dishing (nm)", color="#cdd6f4", fontsize=9)
        hm_fig.tight_layout(pad=1.2)
        self._update_canvas(self.bs_heatmap_canvas, hm_fig)

        # ── Summary Table ────────────────────────────────────────────────────
        stat_cols = ["Position"] + rep_labels + ["Mean", "Stdev", "Max", "Spec"]
        self.bs_table.setColumnCount(len(stat_cols))
        self.bs_table.setHorizontalHeaderLabels(stat_cols)
        self.bs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.bs_table.setRowCount(n_pos)
        for row_i, pos in enumerate(positions):
            vals = dishing_matrix[row_i, :]
            valid_vals = vals[~np.isnan(vals)]
            mean_v = float(np.mean(valid_vals)) if len(valid_vals) else float("nan")
            std_v = float(np.std(valid_vals, ddof=0)) if len(valid_vals) else float("nan")
            max_v = float(np.max(valid_vals)) if len(valid_vals) else float("nan")
            is_stab = pos == "1_LT_stab"
            spec_txt = "N/A" if is_stab else ("PASS" if max_v <= spec_limit else "FAIL")

            row_data = [pos]
            for rep_i in range(n_rep):
                v = vals[rep_i]
                row_data.append(f"{v:.3f}" if not np.isnan(v) else "—")
            row_data += [
                f"{mean_v:.3f}" if not np.isnan(mean_v) else "—",
                f"{std_v:.3f}" if not np.isnan(std_v) else "—",
                f"{max_v:.3f}" if not np.isnan(max_v) else "—",
                spec_txt,
            ]

            for col_i, cell_val in enumerate(row_data):
                item = QTableWidgetItem(cell_val)
                item.setTextAlignment(Qt.AlignCenter)
                # Color based on context
                if col_i == 0:  # Position label
                    if is_stab:
                        item.setForeground(QColor("#a6adc8"))
                elif col_i == len(row_data) - 1:  # Spec column
                    if spec_txt == "FAIL":
                        item.setForeground(QColor("#f38ba8"))
                        item.setFont(QFont("Segoe UI", 10, QFont.Bold))
                    elif spec_txt == "PASS":
                        item.setForeground(QColor("#a6e3a1"))
                else:  # Value cells
                    try:
                        fval = float(cell_val)
                        if not is_stab and fval > spec_limit:
                            item.setForeground(QColor("#f38ba8"))
                        elif not is_stab and fval <= spec_limit * 0.8:
                            item.setForeground(QColor("#a6e3a1"))
                    except (ValueError, TypeError):
                        pass
                self.bs_table.setItem(row_i, col_i, item)

    # ─── Flatten ─────────────────────────────────────────────

    def _on_flatten_execute(self):
        if not self.current_recipe:
            QMessageBox.warning(self, "Warning", "Load data first.")
            return

        pos = self.flat_pos_combo.currentText()
        rep_idx = self.flat_rep_combo.currentIndex()
        order = int(self.flat_order_combo.currentText())
        edge_pct = self.flat_edge_spin.value()

        if rep_idx < 0 or rep_idx >= len(self.current_recipe.repeats):
            return

        repeat = self.current_recipe.repeats[rep_idx]
        if pos not in repeat.profiles:
            QMessageBox.warning(self, "Warning",
                                f"No profile for {pos} in Repeat {rep_idx+1}")
            return

        profile = repeat.profiles[pos]
        result = self.flatten_proc.flatten(
            profile.z_nm, profile.x_mm, order=order, edge_percent=edge_pct
        )

        self.flat_stats_label.setText(
            f"Order {order} | OPM: {result.opm_before:.3f} → {result.opm_after:.3f} nm | "
            f"RMS: {result.rms_before:.3f} → {result.rms_after:.3f} nm | Edge: {edge_pct}%"
        )

        fig = create_flatten_preview_figure(result, profile.x_mm, figsize=(10, 8))
        self._update_canvas(self.flatten_canvas, fig)
        self.flat_undo_btn.setEnabled(self.flatten_proc.can_undo)

    def _on_flatten_undo(self):
        prev = self.flatten_proc.undo()
        if prev and self.current_recipe:
            pos = self.flat_pos_combo.currentText()
            rep_idx = self.flat_rep_combo.currentIndex()
            if 0 <= rep_idx < len(self.current_recipe.repeats):
                profile = self.current_recipe.repeats[rep_idx].profiles.get(pos)
                if profile:
                    fig = create_flatten_preview_figure(prev, profile.x_mm)
                    self._update_canvas(self.flatten_canvas, fig)
                    self.flat_stats_label.setText(
                        f"Undo → Order {prev.order} | OPM: {prev.opm_after:.3f} nm")
        self.flat_undo_btn.setEnabled(self.flatten_proc.can_undo)

    # ─── Export ──────────────────────────────────────────────

    def _on_export(self):
        if not self.current_result or not self.current_recipe:
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return

        base = Path(folder)
        rl = self.current_result.range_label
        try:
            export_summary_csv(self.current_result, base / f"summary_{rl}.csv",
                               robust_result=self.current_result_robust)
            export_avg_line_csv(self.current_recipe, base / f"avg_lines_{rl}.csv")
            export_checklist(self.current_result, base / f"checklist_{rl}.txt",
                             robust_result=self.current_result_robust)

            for name, canvas in [("profiles", self.profile_canvas),
                                  ("trend", self.trend_canvas),
                                  ("wafer_map", self.wafer_canvas),
                                  ("best5", self.best5_canvas)]:
                canvas.figure.savefig(str(base / f"{name}_{rl}.png"), dpi=150,
                                      facecolor="#1e1e2e", bbox_inches="tight")

            # Ball Screw export (only if analysis has been run)
            if self.current_bs_result:
                include_stab = self.bs_show_stab_check.isChecked()
                export_ball_screw_csv(self.current_bs_result, base, include_stab)
                # Save chart images
                for name, canvas in [("bs_bar", self.bs_bar_canvas),
                                     ("bs_heatmap", self.bs_heatmap_canvas)]:
                    canvas.figure.savefig(str(base / f"{name}_{rl}.png"), dpi=150,
                                          facecolor="#1e1e2e", bbox_inches="tight")

            QMessageBox.information(self, "Export", f"Exported to:\n{folder}")
            self.statusBar().showMessage(f"Exported {rl} to {folder}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


def run_app():
    """Launch the application."""
    # High-DPI rounding policy must be set before the QApplication is constructed.
    if QApplication.instance() is None:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()

    if not QApplication.instance():
        sys.exit(app.exec())
    else:
        app.exec()
