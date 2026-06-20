"""Application config: admin access (PIN) persistence.

Stores a single JSON file at ``<project>/config/app_config.json``. The admin PIN
is kept only as a SHA-256 hash, never in plaintext. When no config file exists
the tool falls back to a shipped default PIN so a fresh checkout is still usable;
the admin can change it from the UI, which writes the (hashed) value to disk.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CONFIG_PATH = CONFIG_DIR / "app_config.json"

# Shipped default until the admin sets a site-specific PIN.
DEFAULT_PIN = "0000"


def _hash(pin: str) -> str:
    return hashlib.sha256(pin.encode("utf-8")).hexdigest()


def load_config() -> dict:
    """Read the config JSON, or an empty dict if missing/unreadable."""
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_config(cfg: dict) -> None:
    # Atomic write: serialize to a sibling temp file then os.replace, so an
    # interrupted/concurrent write can never leave a half-written config that
    # would silently fail the PIN gate open to the shipped default.
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
    tmp.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)


def get_admin_pin_hash() -> str:
    """Hash of the active admin PIN (shipped default until one is set)."""
    return load_config().get("admin_pin_hash") or _hash(DEFAULT_PIN)


def is_default_pin() -> bool:
    """True while the active PIN equals the shipped default (incl. no config,
    a falsy/corrupt stored hash, or an admin who re-set the PIN back to it)."""
    return get_admin_pin_hash() == _hash(DEFAULT_PIN)


def verify_admin_pin(pin: str) -> bool:
    return _hash(pin) == get_admin_pin_hash()


def set_admin_pin(pin: str) -> None:
    cfg = load_config()
    cfg["admin_pin_hash"] = _hash(pin)
    save_config(cfg)
