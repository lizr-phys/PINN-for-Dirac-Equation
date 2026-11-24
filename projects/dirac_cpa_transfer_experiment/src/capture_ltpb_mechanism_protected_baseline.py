from __future__ import annotations

"""Capture an explicit per-file SHA-256 baseline for protected prior outputs.

This utility is intentionally separate from the experiment runner.  It records
old artifacts only and excludes both new mechanism-ablation raw roots and the
new figure root.  The audit later requires the exact same path set, sizes, and
hashes.  It refuses to overwrite an existing baseline.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .config import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_ROOT / "ltpb_mechanism_ablation_protected_baseline.json"
EXCLUDED_PREFIXES = (
    "results/raw/ltpb_mechanism_ablation/",
    "results/raw/ltpb_mechanism_ablation_smoke/",
    "results/figures/ltpb_mechanism_ablation/",
)
PROTECTED_REPORTS = (
    PROJECT_ROOT / "EXPERIMENT_REPORT.md",
    PROJECT_ROOT / "LEARNED_TWO_PHASE_PILOT_REPORT.md",
)


def relative_posix(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def is_excluded(relative: str) -> bool:
    return any(relative.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def selected_files() -> list[Path]:
    paths: set[Path] = set()
    for root in (RESULTS_ROOT / "raw", RESULTS_ROOT / "figures"):
        for path in root.rglob("*"):
            if path.is_file() and not is_excluded(relative_posix(path)):
                paths.add(path.resolve())
    for path in RESULTS_ROOT.iterdir():
        if path.is_file() and path.resolve() != OUTPUT_PATH.resolve():
            relative = relative_posix(path)
            if not relative.startswith("results/ltpb_mechanism_ablation_"):
                paths.add(path.resolve())
    for path in PROTECTED_REPORTS:
        if not path.is_file():
            raise FileNotFoundError(f"Missing protected report: {path}")
        paths.add(path.resolve())
    return sorted(paths, key=relative_posix)


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite protected baseline: {OUTPUT_PATH}")
    entries: dict[str, dict[str, int | str]] = {}
    aggregate = hashlib.sha256()
    for path in selected_files():
        relative = relative_posix(path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        entries[relative] = {"size_bytes": len(content), "sha256": digest}
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(digest))
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Protect all prior raw results, figures, summaries, audits, and formal reports from the isolated LTPB mechanism ablation.",
        "selection": {
            "recursive_roots": ["results/raw", "results/figures"],
            "top_level_results_files": True,
            "protected_reports": [relative_posix(path) for path in PROTECTED_REPORTS],
            "excluded_prefixes": list(EXCLUDED_PREFIXES),
            "excluded_top_level_prefix": "results/ltpb_mechanism_ablation_",
        },
        "file_count": len(entries),
        "aggregate_algorithm": "sorted relative UTF-8 path + NUL + binary per-file SHA-256",
        "aggregate_sha256": aggregate.hexdigest(),
        "files": entries,
        "legacy_pre_turn_aggregate_records": {
            "algorithm": "unspecified; retained as provenance only",
            "results/raw/main": "9B9808481529CC320949F3D095B81EACA7FDCD1D7D089FDE96FAF7A68BFCCEF7",
            "results/raw/sensitivity": "F64F0E0BA4327A51C864D2884AF960414CF516F3E191BA1ECF96080648BE8D2E",
            "results/raw/learned_two_phase_pilot": "9541B41DDB0C956230A2621D3BF8EC6727F2B81392E9A97C311029CE72455090",
            "results/figures/learned_two_phase_pilot": "DB20697EDE956EA571887B3EBB0D1B1FDBD7DC3EC456B59B27978B20D9697917",
        },
    }
    write_json(OUTPUT_PATH, payload)
    print(
        f"Captured {len(entries)} protected files in {OUTPUT_PATH}; "
        f"aggregate={payload['aggregate_sha256']}"
    )


if __name__ == "__main__":
    main()
