"""Admin PIN dialogs: authenticate to enter admin mode, and change the PIN.

Kept UI-only; all hashing/persistence lives in ``core.app_config``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QMessageBox,
)

from ..core import app_config

# Child QDialogs don't inherit the MainWindow stylesheet (set per-widget, not
# app-wide), so carry a compact dark theme to match the app.
_DIALOG_QSS = (
    "QDialog { background-color: #1e1e2e; }"
    "QLabel { color: #cdd6f4; font-size: 12px; }"
    "QLineEdit { background-color: #181825; color: #cdd6f4;"
    " border: 1px solid #45475a; border-radius: 4px; padding: 4px; }"
    "QPushButton { background-color: #313244; color: #cdd6f4;"
    " border: 1px solid #45475a; border-radius: 4px; padding: 6px 14px; }"
    "QPushButton:hover { background-color: #45475a; }"
)


class _PinDialog(QDialog):
    """Single password-style PIN entry with an inline error line."""

    def __init__(self, parent, title: str, prompt: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(280)
        self.setStyleSheet(_DIALOG_QSS)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(prompt))

        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.Password)
        self.edit.returnPressed.connect(self.accept)
        lay.addWidget(self.edit)

        self.msg = QLabel("")
        self.msg.setStyleSheet("color: #f38ba8; font-size: 11px;")
        lay.addWidget(self.msg)

        row = QHBoxLayout()
        row.addStretch()
        ok = QPushButton("확인")
        cancel = QPushButton("취소")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        row.addWidget(ok)
        row.addWidget(cancel)
        lay.addLayout(row)

    def value(self) -> str:
        return self.edit.text()


def prompt_admin_pin(parent) -> bool:
    """Prompt for the admin PIN, re-asking on a wrong entry. True if verified."""
    dlg = _PinDialog(parent, "Admin 인증", "Admin PIN을 입력하세요:")
    while dlg.exec() == QDialog.Accepted:
        if app_config.verify_admin_pin(dlg.value()):
            return True
        dlg.msg.setText("PIN이 올바르지 않습니다.")
        dlg.edit.clear()
    return False


def prompt_change_pin(parent) -> bool:
    """Ask for a new PIN twice and persist it. True if changed."""
    while True:
        first = _PinDialog(parent, "Admin PIN 변경", "새 PIN을 입력하세요:")
        if first.exec() != QDialog.Accepted:
            return False
        new_pin = first.value().strip()
        if not new_pin:
            QMessageBox.warning(parent, "Admin PIN 변경", "빈 PIN은 사용할 수 없습니다.")
            continue

        confirm = _PinDialog(parent, "Admin PIN 변경", "새 PIN을 다시 입력하세요:")
        if confirm.exec() != QDialog.Accepted:
            return False
        if confirm.value().strip() != new_pin:
            QMessageBox.warning(
                parent, "Admin PIN 변경", "PIN이 일치하지 않습니다. 다시 입력하세요.")
            continue

        app_config.set_admin_pin(new_pin)
        return True
