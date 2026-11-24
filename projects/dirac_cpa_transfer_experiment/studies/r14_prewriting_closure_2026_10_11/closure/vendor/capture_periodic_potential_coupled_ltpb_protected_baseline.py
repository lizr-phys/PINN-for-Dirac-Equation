from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "periodic_potential_coupled_ltpb"
DEFAULT_OUTPUT = (
    ROOT / "results" / "raw" / EXPERIMENT / "protection" /
    "preexisting_results_sha256.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_files(root: Path = ROOT) -> list[Path]:
    """Return existing research artifacts outside the new experiment namespace."""

    files: set[Path] = set()
    for parent in (root / "results" / "raw", root / "results" / "figures"):
        if not parent.is_dir():
            continue
        for path in parent.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(parent)
            if relative.parts and relative.parts[0].startswith(EXPERIMENT):
                continue
            files.add(path)

    for path in (root / "results").glob("*"):
        if path.is_file() and not path.name.startswith(EXPERIMENT):
            files.add(path)
    for pattern in ("*REPORT.md", "*PREREGISTRATION.md"):
        for path in root.glob(pattern):
            if not path.name.startswith("PERIODIC_POTENTIAL_COUPLED_LTPB"):
                files.add(path)
    return sorted(files, key=lambda value: value.as_posix().lower())


def capture(output: Path = DEFAULT_OUTPUT, *, verify_only: bool = False) -> dict[str, object]:
    output = Path(output)
    if verify_only:
        if not output.is_file():
            raise FileNotFoundError(output)
        with output.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        changed: list[str] = []
        missing: list[str] = []
        for row in rows:
            path = ROOT / row["relative_path"]
            if not path.is_file():
                missing.append(row["relative_path"])
            elif sha256_file(path) != row["sha256"]:
                changed.append(row["relative_path"])
        return {
            "status": "PASS" if not changed and not missing else "FAIL",
            "file_count": len(rows),
            "changed": changed,
            "missing": missing,
        }

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite protected baseline: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "relative_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in protected_files()
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("relative_path", "size_bytes", "sha256")
        )
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "experiment": EXPERIMENT,
                "purpose": "immutable evidence that earlier formal artifacts were not modified",
                "manifest": output.relative_to(ROOT).as_posix(),
                "file_count": len(rows),
                "manifest_sha256": sha256_file(output),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {"status": "CAPTURED", "file_count": len(rows), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture or verify prior-result hashes")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(capture(args.output, verify_only=args.verify_only), indent=2))


if __name__ == "__main__":
    main()
