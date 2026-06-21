"""Admin dialog to create / edit / delete spec presets (core.spec_config).

Child QDialogs do not inherit the MainWindow stylesheet (it is set per-widget, not
app-wide), so this dialog carries its own compact dark theme to match the app.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QComboBox, QDoubleSpinBox, QListWidget, QPushButton, QMessageBox,
)

from ..core import spec_config

_DIALOG_QSS = (
    "QDialog { background-color: #1e1e2e; }"
    "QLabel { color: #cdd6f4; font-size: 15px; }"
    "QLineEdit, QComboBox, QDoubleSpinBox, QListWidget {"
    " background-color: #181825; color: #cdd6f4;"
    " border: 1px solid #45475a; border-radius: 4px; padding: 4px; }"
    "QPushButton { background-color: #313244; color: #cdd6f4;"
    " border: 1px solid #45475a; border-radius: 4px; padding: 6px 14px; }"
    "QPushButton:hover { background-color: #45475a; }"
)

_MODE_TO_COMBO = {"none": "None", "percentile": "Percentile", "pixels": "Pixels"}
_COMBO_TO_MODE = {v: k for k, v in _MODE_TO_COMBO.items()}


class PresetManagerDialog(QDialog):
    """Create/edit/delete named spec presets.

    ``current`` pre-fills the form from the live UI ("save current settings").
    ``range_mm`` scopes the optional spec-limit override fields to the loaded
    recipe's range (disabled when no recipe is loaded). After the dialog closes,
    ``selected_name`` holds the preset the caller should select (or None).
    """

    def __init__(self, parent, current: dict, range_mm: Optional[int]):
        super().__init__(parent)
        self.setWindowTitle("Spec 프리셋 관리")
        self.setMinimumWidth(575)
        self.setStyleSheet(_DIALOG_QSS)
        self._range_mm = range_mm
        self.selected_name: Optional[str] = None

        root = QHBoxLayout(self)

        # Left: saved presets + delete
        left = QVBoxLayout()
        left.addWidget(QLabel("저장된 프리셋"))
        self.list = QListWidget()
        self.list.itemClicked.connect(self._on_list_click)
        left.addWidget(self.list)
        del_btn = QPushButton("삭제")
        del_btn.clicked.connect(self._on_delete)
        left.addWidget(del_btn)
        root.addLayout(left, 1)

        # Right: form
        form = QGridLayout()
        rng_txt = f"{range_mm}mm" if range_mm is not None else "—"
        r = 0
        form.addWidget(QLabel("이름"), r, 0)
        self.name_edit = QLineEdit()
        form.addWidget(self.name_edit, r, 1); r += 1

        form.addWidget(QLabel("장비 유형"), r, 0)
        self.equip_combo = QComboBox()
        self.equip_combo.addItem("Isolated AE (iso)", "iso")
        self.equip_combo.addItem("Double Walled AE (dw)", "dw")
        form.addWidget(self.equip_combo, r, 1); r += 1

        form.addWidget(QLabel("Outlier 모드"), r, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["None", "Percentile", "Pixels"])
        self.mode_combo.currentTextChanged.connect(self._on_mode)
        form.addWidget(self.mode_combo, r, 1); r += 1

        form.addWidget(QLabel("Outlier 값"), r, 0)
        self.value_spin = QDoubleSpinBox()
        self.value_spin.setRange(0.0, 9999.0)
        self.value_spin.setDecimals(1)
        form.addWidget(self.value_spin, r, 1); r += 1

        form.addWidget(QLabel("장비 ID"), r, 0)
        self.equip_id_edit = QLineEdit()
        form.addWidget(self.equip_id_edit, r, 1); r += 1

        form.addWidget(QLabel("작성자"), r, 0)
        self.author_edit = QLineEdit()
        form.addWidget(self.author_edit, r, 1); r += 1

        form.addWidget(QLabel(f"Rep 한계 override ({rng_txt})"), r, 0)
        self.rep_edit = QLineEdit()
        self.rep_edit.setPlaceholderText("비우면 내장 기본")
        form.addWidget(self.rep_edit, r, 1); r += 1

        form.addWidget(QLabel(f"OPM 한계 override ({rng_txt})"), r, 0)
        self.opm_edit = QLineEdit()
        self.opm_edit.setPlaceholderText("비우면 내장 기본")
        form.addWidget(self.opm_edit, r, 1); r += 1

        if range_mm is None:
            self.rep_edit.setEnabled(False)
            self.opm_edit.setEnabled(False)

        right = QVBoxLayout()
        right.addLayout(form)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("저장")
        save_btn.clicked.connect(self._on_save)
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(close_btn)
        right.addLayout(btn_row)
        root.addLayout(right, 2)

        self._prefill(current)
        self._refresh_list()
        self._on_mode(self.mode_combo.currentText())

    # ----- helpers -----
    def _on_mode(self, text: str):
        self.value_spin.setEnabled(text != "None")

    def _prefill(self, src: dict):
        et = src.get("equipment_type", "iso")
        self.equip_combo.setCurrentIndex(0 if et == "iso" else 1)
        ol = src.get("outlier") or {}
        self.mode_combo.setCurrentText(_MODE_TO_COMBO.get(ol.get("mode", "none"), "None"))
        if ol.get("value") is not None:
            self.value_spin.setValue(float(ol["value"]))
        meta = src.get("meta") or {}
        self.equip_id_edit.setText(meta.get("equipment_id", ""))
        self.author_edit.setText(meta.get("author", ""))

    def _refresh_list(self):
        self.list.clear()
        for p in spec_config.list_presets():
            self.list.addItem(p.get("name", ""))

    def _on_list_click(self, item):
        p = spec_config.get_preset(item.text())
        if not p:
            return
        self.name_edit.setText(p.get("name", ""))
        self._prefill(p)
        ov = None
        if self._range_mm is not None:
            ov = (p.get("spec_overrides") or {}).get(str(self._range_mm))
        self.rep_edit.setText("" if not ov or ov.get("rep_limit") is None else str(ov["rep_limit"]))
        self.opm_edit.setText("" if not ov or ov.get("opm_limit") is None else str(ov["opm_limit"]))

    @staticmethod
    def _parse(text: str):
        """(ok, value-or-None). Blank -> (True, None); bad number -> (False, None)."""
        text = text.strip()
        if not text:
            return True, None
        try:
            return True, float(text)
        except ValueError:
            return False, None

    def _on_save(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "프리셋", "이름을 입력하세요.")
            return
        ok_r, rep = self._parse(self.rep_edit.text())
        ok_o, opm = self._parse(self.opm_edit.text())
        if not ok_r or not ok_o:
            QMessageBox.warning(self, "프리셋", "한계값은 숫자여야 합니다 (비우면 내장 기본).")
            return
        if (rep is not None and rep <= 0) or (opm is not None and opm <= 0):
            QMessageBox.warning(self, "프리셋", "한계값은 0보다 큰 값이어야 합니다 (nm).")
            return

        # Preserve other ranges' overrides when editing an existing preset.
        existing = spec_config.get_preset(name)
        spec_overrides = dict((existing or {}).get("spec_overrides") or {})
        if self._range_mm is not None:
            key = str(self._range_mm)
            entry = {}
            if rep is not None:
                entry["rep_limit"] = rep
            if opm is not None:
                entry["opm_limit"] = opm
            if entry:
                spec_overrides[key] = entry
            else:
                spec_overrides.pop(key, None)

        preset = {
            "name": name,
            "equipment_type": self.equip_combo.currentData(),
            "outlier": {
                "mode": _COMBO_TO_MODE[self.mode_combo.currentText()],
                "value": float(self.value_spin.value()),
            },
            "spec_overrides": spec_overrides,
            "meta": {
                "equipment_id": self.equip_id_edit.text().strip(),
                "author": self.author_edit.text().strip(),
            },
        }
        spec_config.save_preset(preset)
        self.selected_name = name
        self._refresh_list()
        QMessageBox.information(self, "프리셋", f"'{name}' 저장됨.")

    def _on_delete(self):
        item = self.list.currentItem()
        if not item:
            return
        name = item.text()
        if QMessageBox.question(self, "삭제", f"'{name}' 프리셋을 삭제할까요?") != QMessageBox.Yes:
            return
        spec_config.delete_preset(name)
        if self.selected_name == name:
            self.selected_name = None
        self._refresh_list()
