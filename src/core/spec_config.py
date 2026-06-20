"""Named spec / recipe presets (judgment configuration).

A preset bundles the per-recipe judgment setup so an operator does not re-enter it
each run: equipment type, the robust-outlier preset, optional per-range spec-limit
overrides, and report metadata. Presets persist in the same JSON as the admin PIN
(``core.app_config``) under the ``presets`` key.

Preset shape::

    {
      "name": "XE-7 ISO 25mm",
      "equipment_type": "iso" | "dw",
      "outlier": {"mode": "none"|"percentile"|"pixels", "value": 1.0},
      "spec_overrides": {"25": {"rep_limit": 12.9, "opm_limit": 200.0}, ...},
      "meta": {"equipment_id": "XE7-01", "author": "백광림"}
    }

``spec_overrides`` is keyed by range_mm as a string; an absent range (or absent
rep_limit/opm_limit) means the built-in ``analyzer.SPEC_*`` table is used.
"""
from __future__ import annotations

from typing import Optional

from .app_config import load_config, save_config

_KEY = "presets"


def list_presets() -> list[dict]:
    """All saved presets (empty list if none)."""
    presets = load_config().get(_KEY, [])
    return presets if isinstance(presets, list) else []


def get_preset(name: str) -> Optional[dict]:
    for p in list_presets():
        if p.get("name") == name:
            return p
    return None


def save_preset(preset: dict) -> None:
    """Insert or replace a preset by name."""
    cfg = load_config()
    presets = cfg.get(_KEY, [])
    if not isinstance(presets, list):
        presets = []
    presets = [p for p in presets
               if isinstance(p, dict) and p.get("name") != preset.get("name")]
    presets.append(preset)
    cfg[_KEY] = presets
    save_config(cfg)


def delete_preset(name: str) -> None:
    cfg = load_config()
    presets = cfg.get(_KEY, [])
    if not isinstance(presets, list):
        presets = []
    cfg[_KEY] = [p for p in presets
                 if isinstance(p, dict) and p.get("name") != name]
    save_config(cfg)


def resolve_overrides(preset: Optional[dict], range_mm: int) -> Optional[dict]:
    """The {"rep_limit", "opm_limit"} override for this range, or None if the
    preset has no override for it (so the built-in spec table applies)."""
    if not preset:
        return None
    ov = (preset.get("spec_overrides") or {}).get(str(range_mm))
    return ov or None
