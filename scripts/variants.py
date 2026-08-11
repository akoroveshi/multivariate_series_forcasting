"""Which trained run of each model family do we actually use?

Every training run drops a JSON into ``results/``. Rather than hard-coding a list
of candidate names -- which went stale every time we added a round of experiments
-- this module discovers them: any ``results/*.json`` carrying a ``config`` with a
``model_type`` and a scored ``val`` block is a candidate.

The winner per family is decided by **validation MAE only**. The test regime never
participates in a choice. Report tables, figures and the submission archive all
import this so they cannot disagree about which checkpoint is "the" model.
"""

from __future__ import annotations

import json
from pathlib import Path

#: Aggregate files in results/ that are not individual training runs.
NOT_A_RUN = {"ablations.json", "jena.json", "ensemble.json", "baselines_local.json",
             "permutation_importance.json", "selection.json"}


def load_runs(results_dir: Path) -> dict[str, dict]:
    """Every training-run log in ``results_dir``, keyed by tag."""
    runs: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name in NOT_A_RUN:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if "config" in payload and "val" in payload and payload.get("val"):
            runs[path.stem] = payload
    return runs


def family_of(run: dict) -> str:
    """The model family a run belongs to."""
    return str(run["config"]["model_type"])


def select_best(results_dir: Path) -> dict[str, str]:
    """``family -> winning tag``, chosen by validation MAE."""
    runs = load_runs(results_dir)
    best: dict[str, str] = {}
    for tag, run in runs.items():
        family = family_of(run)
        if family not in best or run["val"]["mae"] < runs[best[family]]["val"]["mae"]:
            best[family] = tag
    return best


def describe(tag: str, run: dict) -> str:
    """Readable table label: family plus the settings that actually differ."""
    config = run["config"]
    base = "PatchTST-cov" if config["model_type"] == "patchtst" else "LSTM+Attention"
    bits = []
    if not config.get("use_revin", True):
        bits.append("no RevIN")
    windows = run.get("train_windows")
    if windows:
        bits.append(f"{windows / 1000:.0f}k win.")
    if config.get("history_len", 168) != 168:
        bits.append(f"$L{{=}}{config['history_len']}$")
    return f"{base}" + (f" ({', '.join(bits)})" if bits else "")


def checkpoint_for(tag: str, root: Path, full_history: bool = False) -> Path:
    """Path to a run's checkpoint; ``full_history`` picks the retrained twin."""
    return root / "runs" / (f"{tag}_full" if full_history else tag) / "checkpoint.pt"


def summarise(results_dir: Path) -> str:
    """One line per run, best per family marked -- used by the CLI helpers."""
    runs = load_runs(results_dir)
    best = select_best(results_dir)
    lines = [f"{'val MAE':>8s} {'val WAPE':>9s} {'test WAPE':>10s} {'ep':>4s}  tag"]
    for tag, run in sorted(runs.items(), key=lambda kv: kv[1]["val"]["mae"]):
        mark = " *" if best.get(family_of(run)) == tag else "  "
        lines.append(
            f"{run['val']['mae']:8.4f} {run['val']['wape']:9.3f} "
            f"{run['test']['wape']:10.3f} {run['best_epoch']:4d}{mark} {tag}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[1]
    print(summarise(Path(sys.argv[1]) if len(sys.argv) > 1 else root / "results"))
