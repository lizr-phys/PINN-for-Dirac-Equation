from __future__ import annotations

"""Freeze every pre-wavepacket research artifact as a SHA-256 manifest."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .config import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_ROOT / "wavepacket_shared_spectral_ltpb_protected_baseline.json"
EXCLUDED_PREFIXES = (
    "results/raw/wavepacket_shared_spectral_ltpb/",
    "results/raw/wavepacket_shared_spectral_ltpb_smoke/",
    "results/figures/wavepacket_shared_spectral_ltpb/",
    "results/logs/wavepacket_shared_spectral_ltpb/",
    "results/wavepacket_shared_spectral_ltpb_",
)
EXCLUDED_REPORT_PREFIX = "WAVEPACKET_SHARED_SPECTRAL_LTPB_"


def relative_posix(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def is_excluded(relative: str) -> bool:
    return any(relative.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def selected_files() -> list[Path]:
    paths: set[Path] = set()
    for root_name in ("raw", "figures", "logs"):
        root = RESULTS_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and not is_excluded(relative_posix(path)):
                paths.add(path.resolve())
    if RESULTS_ROOT.is_dir():
        for path in RESULTS_ROOT.iterdir():
            if path.is_file() and not is_excluded(relative_posix(path)):
                paths.add(path.resolve())
    for path in PROJECT_ROOT.glob("*.md"):
        if not path.name.startswith(EXCLUDED_REPORT_PREFIX):
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
        "purpose": (
            "Protect all earlier raw results, figures, logs, aggregates, and top-level "
            "research Markdown from the isolated shared-spectral wavepacket experiment."
        ),
        "selection": {
            "recursive_roots": ["results/raw", "results/figures", "results/logs"],
            "top_level_results_files": True,
            "top_level_markdown_records": True,
            "excluded_prefixes": list(EXCLUDED_PREFIXES),
            "excluded_report_prefix": EXCLUDED_REPORT_PREFIX,
        },
        "file_count": len(entries),
        "aggregate_algorithm": "sorted relative UTF-8 path + NUL + binary per-file SHA-256",
        "aggregate_sha256": aggregate.hexdigest(),
        "files": entries,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_PATH, payload)
    print(
        f"Captured {len(entries)} protected files in {OUTPUT_PATH}; "
        f"aggregate={payload['aggregate_sha256']}"
    )


if __name__ == "__main__":
    main()
