from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "results"
    / "wavepacket_shared_spectral_ltpb_preregistration_sha256.json"
)
FORMAL = (
    ROOT / "results" / "raw" / "wavepacket_shared_spectral_ltpb"
)
FILES = (
    "WAVEPACKET_SHARED_SPECTRAL_LTPB_PREREGISTRATION.md",
    "WAVEPACKET_SHARED_SPECTRAL_LTPB_FIGURE_CONTRACT.md",
    "src/physics.py",
    "src/wavepacket_config.py",
    "src/wavepacket_physics.py",
    "src/wavepacket_models.py",
    "src/wavepacket_evaluate.py",
    "src/train_wavepacket_shared_spectral_ltpb.py",
    "src/run_wavepacket_shared_spectral_ltpb.py",
    "src/wavepacket_shared_spectral_ltpb_decision.py",
    "src/aggregate_wavepacket_shared_spectral_ltpb.py",
    "src/plot_wavepacket_shared_spectral_ltpb.py",
    "src/qa_wavepacket_shared_spectral_ltpb_figures.py",
    "src/build_wavepacket_shared_spectral_ltpb_report.py",
    "src/audit_wavepacket_shared_spectral_ltpb.py",
    "src/capture_wavepacket_shared_spectral_ltpb_protected_baseline.py",
    "tests/test_wavepacket_shared_spectral_ltpb.py",
    "tests/test_wavepacket_evaluation_decision.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_sha256(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(hashes.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    formal_files = [
        path for path in FORMAL.rglob("*") if path.is_file()
    ] if FORMAL.exists() else []
    if formal_files:
        shown = ", ".join(
            str(path.relative_to(ROOT)) for path in formal_files[:8]
        )
        raise RuntimeError(
            "Refusing to freeze preregistration after formal artifacts exist: "
            + shown
        )
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite freeze manifest: {OUTPUT}")
    missing = [relative for relative in FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing formal files: {missing}")

    hashes = {relative: sha256(ROOT / relative) for relative in FILES}
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "wavepacket_shared_spectral_ltpb",
        "frozen_before_formal_results": True,
        "formal_files_present_at_freeze": 0,
        "smoke_scope": (
            "dimension, hard-initial-condition, gradient, finite-loss and "
            "atomic-artifact checks only; no tuning from formal results"
        ),
        "full_pytest_before_freeze": "104 passed",
        "preformal_pipeline_note": (
            "A first preformal manifest was superseded before any formal artifact "
            "after an isolated synthetic pipeline smoke found and corrected only "
            "figure/QA plumbing defects: the rho-to-density source-schema key, "
            "the training-phase palette alias, wide/short PDF preview dimensions, "
            "and a pypdf text-extraction fallback when pdftotext is unavailable. "
            "No model, physics, optimizer, sampling, metric or decision threshold "
            "changed. The superseded manifest is retained under "
            "results/logs/wavepacket_shared_spectral_ltpb/."
        ),
        "hash_algorithm": "SHA-256",
        "aggregate_algorithm": (
            "sorted relative UTF-8 path + NUL + ASCII file hash + newline"
        ),
        "file_count": len(hashes),
        "files": hashes,
        "aggregate_sha256": aggregate_sha256(hashes),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(OUTPUT)
    print(payload["aggregate_sha256"])


if __name__ == "__main__":
    main()


__all__ = ["FILES", "FORMAL", "OUTPUT", "sha256", "aggregate_sha256"]
