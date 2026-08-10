"""Which trained variant of each model family do we actually use?

The RevIN ablation was strong enough to be worth re-running at the full training
budget, so each family has two candidate runs. The winner is decided by
**validation MAE only** -- the test regime never participates in a choice. Report
tables, figures and the submission archive all import this so they cannot disagree
about which checkpoint is "the" model.
"""

from __future__ import annotations

import json
from pathlib import Path

#: ``family -> candidate run tags``, in the order they were introduced.
FAMILIES: dict[str, list[str]] = {
    "patchtst": ["patchtst_main", "patchtst_norevin"],
    "lstm_attention": ["lstm_main", "lstm_norevin"],
}

#: Human-readable names used in the report tables.
DISPLAY_NAMES: dict[str, str] = {
    "patchtst_main": "PatchTST-cov",
    "patchtst_norevin": "PatchTST-cov, no RevIN",
    "lstm_main": "LSTM+Attention",
    "lstm_norevin": "LSTM+Attention, no RevIN",
}


def load_runs(results_dir: Path) -> dict[str, dict]:
    """Every candidate run log that exists, keyed by tag."""
    runs = {}
    for candidates in FAMILIES.values():
        for tag in candidates:
            path = results_dir / f"{tag}.json"
            if path.exists():
                runs[tag] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def select_best(results_dir: Path) -> dict[str, str]:
    """``family -> winning tag``, chosen by validation MAE."""
    runs = load_runs(results_dir)
    chosen: dict[str, str] = {}
    for family, candidates in FAMILIES.items():
        available = [tag for tag in candidates if tag in runs]
        if not available:
            continue
        chosen[family] = min(available, key=lambda tag: runs[tag]["val"]["mae"])
    return chosen


def checkpoint_for(tag: str, root: Path, full_history: bool = False) -> Path:
    """Path to a run's checkpoint; ``full_history`` picks the retrained twin."""
    return root / "runs" / (f"{tag}_full" if full_history else tag) / "checkpoint.pt"
