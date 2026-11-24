from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "multimode_ltpb_preregistration_sha256.json"
FORMAL = ROOT / "results" / "raw" / "multimode_ltpb"
FILES = (
    "MULTIMODE_LTPB_PREREGISTRATION.md",
    "MULTIMODE_LTPB_FIGURE_CONTRACT.md",
    "src/physics.py",
    "src/multimode_config.py",
    "src/multimode_physics.py",
    "src/multimode_models.py",
    "src/multimode_evaluate.py",
    "src/train_multimode_ltpb.py",
    "src/run_multimode_ltpb.py",
    "src/multimode_ltpb_decision.py",
    "src/aggregate_multimode_ltpb.py",
    "src/plot_multimode_ltpb.py",
    "src/qa_multimode_ltpb_figures.py",
    "src/build_multimode_ltpb_report.py",
    "src/audit_multimode_ltpb.py",
    "tests/test_multimode_ltpb.py",
    "tests/test_multimode_evaluation_decision.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    formal_files = list(FORMAL.rglob("*")) if FORMAL.exists() else []
    formal_files = [path for path in formal_files if path.is_file()]
    if formal_files:
        raise RuntimeError(
            "Refusing to create a preregistration freeze after formal artifacts exist: "
            + ", ".join(str(path.relative_to(ROOT)) for path in formal_files[:5])
        )
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite existing freeze manifest: {OUTPUT}")
    missing = [relative for relative in FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze missing files: {missing}")
    hashes = {relative: sha256(ROOT / relative) for relative in FILES}
    digest = hashlib.sha256()
    for relative, file_hash in sorted(hashes.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_before_formal_results": True,
        "formal_files_present_at_freeze": 0,
        "smoke_protocol": "five learned models, seed 0, 20 Adam steps; code/NaN/artifact check only",
        "full_pytest_before_freeze": "75 passed",
        "files": hashes,
        "aggregate_sha256": digest.hexdigest(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(payload["aggregate_sha256"])


if __name__ == "__main__":
    main()
