from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_wavepacket_shared_spectral_ltpb import (
    COLORS,
    FIGURES,
    MODEL_LABELS,
    SOURCE_SCHEMAS,
    WAVEPACKET_MODELS,
    WAVEPACKET_SEEDS,
    _panel_labels,
    _style,
)


FIGURE_ID = "per_seed_results"
OUTPUT_SUFFIXES = ("png", "svg", "pdf")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair(figure_dir: Path = FIGURES) -> Path:
    """Repair only the frozen per-seed figure layout from its unchanged source CSV.

    Formal numerical data, model outputs, metrics, thresholds and the frozen
    plotter remain untouched.  Originals are retained for a complete audit trail.
    """

    correction_path = figure_dir / "POSTFREEZE_FIGURE_CORRECTION.json"
    if correction_path.exists():
        raise FileExistsError(f"Refusing to repeat recorded correction: {correction_path}")

    source_path = figure_dir / "source_data" / f"{FIGURE_ID}.csv"
    manifest_path = figure_dir / "figure_manifest.json"
    source = pd.read_csv(source_path)
    if source.columns.tolist() != list(SOURCE_SCHEMAS[FIGURE_ID]) or len(source) != 75:
        raise ValueError("Per-seed source data differ from the frozen 75-row contract")
    numeric = source.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("Per-seed source contains non-finite values")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["figures"] if item["id"] == FIGURE_ID)
    if entry["source_sha256"] != _sha256(source_path):
        raise ValueError("Per-seed source SHA-256 no longer matches the formal manifest")

    backup_dir = figure_dir / "pre_layout_correction"
    if backup_dir.exists():
        expected_backup = {manifest_path.name} | {
            f"{FIGURE_ID}.{suffix}" for suffix in OUTPUT_SUFFIXES
        }
        observed_backup = {path.name for path in backup_dir.iterdir() if path.is_file()}
        if observed_backup != expected_backup:
            raise RuntimeError(
                f"Incomplete pre-correction backup: {sorted(observed_backup)}"
            )
    else:
        backup_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(manifest_path, backup_dir / manifest_path.name)
        for suffix in OUTPUT_SUFFIXES:
            original = figure_dir / f"{FIGURE_ID}.{suffix}"
            shutil.copy2(original, backup_dir / original.name)
    original_hashes: dict[str, str] = {}
    for suffix in OUTPUT_SUFFIXES:
        original_hashes[suffix] = _sha256(backup_dir / f"{FIGURE_ID}.{suffix}")

    _style()
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.75), sharex=True)
    specs = (
        ("spinor_relative_l2", "Spinor relative L2", True),
        ("current_relative_l2", "Current relative L2", True),
        ("fidelity", "Mean fidelity", False),
    )
    for axis, (metric, title, logarithmic) in zip(axes, specs, strict=True):
        for model in WAVEPACKET_MODELS:
            subset = source[(source.model == model) & (source.metric == metric)].sort_values(
                "seed"
            )
            axis.plot(
                subset.seed,
                np.maximum(subset.value, 1.0e-15),
                marker="o",
                markersize=3.2,
                color=COLORS[model],
                label=MODEL_LABELS[model],
            )
        if logarithmic:
            axis.set_yscale("log")
        else:
            axis.set_ylim(-0.03, 1.03)
        axis.set_title(title)
        axis.set_xlabel("Paired seed")
        axis.set_xticks(WAVEPACKET_SEEDS)
        axis.grid(axis="y", which="both", alpha=0.16)
    axes[0].set_ylabel("Metric value")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
    )
    _panel_labels(axes)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.84, bottom=0.36, wspace=0.34)

    new_hashes: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="wavepacket-layout-") as temporary:
        temporary_dir = Path(temporary)
        with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.04}):
            for suffix in OUTPUT_SUFFIXES:
                staged = temporary_dir / f"{FIGURE_ID}.{suffix}"
                kwargs: dict[str, object] = {
                    "facecolor": "white",
                    "metadata": {
                        "Creator": "Python/matplotlib post-freeze layout-only repair"
                    },
                }
                if suffix == "png":
                    kwargs["dpi"] = 600
                fig.savefig(staged, **kwargs)
                if not staged.is_file() or staged.stat().st_size == 0:
                    raise RuntimeError(f"Failed to stage repaired {suffix}")
            plt.close(fig)
        for suffix in OUTPUT_SUFFIXES:
            staged = temporary_dir / f"{FIGURE_ID}.{suffix}"
            destination = figure_dir / staged.name
            same_volume_staging = figure_dir / f".{staged.name}.layout-staging"
            shutil.copy2(staged, same_volume_staging)
            same_volume_staging.replace(destination)
            new_hashes[suffix] = _sha256(destination)

    entry["output_sha256"] = new_hashes
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    correction = {
        "experiment": "wavepacket_shared_spectral_ltpb",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "layout_only",
        "figure": FIGURE_ID,
        "reason": (
            "Formal visual QA found that the axes-level external legend and tight "
            "save boundary compressed the three data axes. The unchanged 75-row "
            "source CSV was redrawn with fixed subplot margins and a figure legend."
        ),
        "source_data": source_path.relative_to(figure_dir.parent.parent.parent).as_posix(),
        "source_sha256": _sha256(source_path),
        "original_outputs_sha256": original_hashes,
        "repaired_outputs_sha256": new_hashes,
        "backup_directory": backup_dir.relative_to(figure_dir.parent.parent.parent).as_posix(),
        "unchanged": [
            "formal learned run artifacts",
            "oracle artifacts",
            "models and checkpoints",
            "physics and residual",
            "metrics and aggregate CSVs",
            "preregistered decision JSON and thresholds",
            "all figure source data",
            "frozen formal code and its SHA-256 manifest",
        ],
    }
    correction_path.write_text(
        json.dumps(correction, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return correction_path


def main() -> None:
    print(repair())


if __name__ == "__main__":
    main()
