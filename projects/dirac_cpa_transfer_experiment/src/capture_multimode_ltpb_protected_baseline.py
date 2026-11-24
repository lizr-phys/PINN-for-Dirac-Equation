from __future__ import annotations

"""Freeze all pre-Multi-LTPB research artifacts as a per-file SHA-256 manifest.

The experiment and its final audit use isolated ``multimode_ltpb`` paths.  This
capture refuses to overwrite an existing manifest and deliberately excludes
only those new paths, so every earlier raw result, figure, log, aggregate and
top-level Markdown research record remains byte-for-byte protected.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from .config import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_PATH = RESULTS_ROOT / "multimode_ltpb_protected_baseline.json"
EXCLUDED_PREFIXES = (
    "results/raw/multimode_ltpb/",
    "results/raw/multimode_ltpb_smoke/",
    "results/figures/multimode_ltpb/",
    "results/logs/multimode_ltpb/",
    "results/multimode_ltpb_",
)
EXCLUDED_REPORT_PREFIX = "MULTIMODE_LTPB_"


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
            "Protect every prior raw result, figure, log, aggregate and Markdown "
            "research record from the isolated two-wave-number Multi-LTPB experiment."
        ),
        "selection": {
            "recursive_roots": ["results/raw", "results/figures", "results/logs"],
            "top_level_results_files": True,
            "top_level_markdown_records": True,
            "excluded_prefixes": list(EXCLUDED_PREFIXES),
            "excluded_report_prefix": EXCLUDED_REPORT_PREFIX,
        },
        "file_count": len(entries),
        "aggregate_algorithm": (
            "sorted relative UTF-8 path + NUL + binary per-file SHA-256"
        ),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": entries,
    }
    write_json(OUTPUT_PATH, payload)
    print(
        f"Captured {len(entries)} protected files in {OUTPUT_PATH}; "
        f"aggregate={payload['aggregate_sha256']}"
    )


if __name__ == "__main__":
    main()
