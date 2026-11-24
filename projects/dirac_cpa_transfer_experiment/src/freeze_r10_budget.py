from __future__ import annotations
import json
from datetime import datetime, timezone
from .r10_config import ROOT, RAW, DOCS, source_directory, SEEDS
from .r9_config import RAW as R9_RAW
from .freeze_r7_phase_release import write_json, sha256


def science_files():
    from .freeze_r9_wavepacket import science_files as r9_science
    names = ["r10_config.py", "r10_norm_diagnostics.py", "r10_state.py", "r10_evaluate.py",
             "train_r10_budget.py", "freeze_r10_budget.py", "run_r10_budget.py"]
    files = [ROOT/"src"/p for p in names]+r9_science()
    files += [ROOT/"tests/test_r10_budget.py", DOCS/"preregistration.md", DOCS/"theory_check.md", DOCS/"figure_contract.md",
              ROOT/"NEXT_STEP_PLAN_AFTER_R9.md", RAW/"pre_freeze_test_result.json", RAW/"smoke_validation.json",
              R9_RAW/"audit.json", R9_RAW/"artifact_sha256.json", R9_RAW/"decision.json"]
    for seed in SEEDS:
        files += [source_directory(seed)/name for name in ("config.json", "model_state.pt", "recovery_state.pt", "metrics.json")]
    return sorted(set(files))


def preserve_sources():
    path = RAW/"protected_prior_results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from .freeze_r9_wavepacket import verify_history, verify_freeze
    verify_freeze()
    if not verify_history()["pass_check"]:
        raise RuntimeError("Historical source integrity failed")
    audit = json.loads((R9_RAW/"audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS":
        raise RuntimeError("R9 must complete its engineering audit before R10")
    files = json.loads((R9_RAW/"artifact_sha256.json").read_text(encoding="utf-8"))["files"]
    files |= json.loads((R9_RAW/"protected_history.json").read_text(encoding="utf-8"))["files"]
    files |= {str((R9_RAW/"audit.json").relative_to(ROOT)).replace("\\", "/"): sha256(R9_RAW/"audit.json"),
              str((R9_RAW/"artifact_sha256.json").relative_to(ROOT)).replace("\\", "/"): sha256(R9_RAW/"artifact_sha256.json")}
    result = dict(captured_utc=datetime.now(timezone.utc).isoformat(), files=files,
                  all_r9_and_historical_results_preserved=True)
    write_json(path, result)
    return result


def verify_prior_results():
    record = json.loads((RAW/"protected_prior_results.json").read_text(encoding="utf-8"))
    changed = [p for p, digest in record["files"].items()
               if not (ROOT/p).is_file() or sha256(ROOT/p) != digest]
    return dict(status="PASS" if not changed else "FAIL", checked_files=len(record["files"]), changed=changed)


def freeze():
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW/"frozen_protocol_and_code.json"
    if path.exists():
        return verify_freeze()
    preserve_sources()
    if list((RAW/"formal").rglob("recovery_state.pt")):
        raise RuntimeError("Scientific freeze must precede formal continuation")
    for name in ("pre_freeze_test_result.json", "smoke_validation.json"):
        if json.loads((RAW/name).read_text(encoding="utf-8"))["status"] != "PASS":
            raise RuntimeError("R10 engineering gate failed")
    record = dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
                  files={p.relative_to(ROOT).as_posix(): sha256(p) for p in science_files()})
    write_json(path, record)
    return sha256(path)


def verify_freeze():
    path = RAW/"frozen_protocol_and_code.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    for name, digest in record["files"].items():
        if not (ROOT/name).is_file() or sha256(ROOT/name) != digest:
            raise RuntimeError("R10 frozen file changed: "+name)
    return sha256(path)
