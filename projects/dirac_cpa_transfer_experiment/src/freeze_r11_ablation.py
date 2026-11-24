from __future__ import annotations
import json
from datetime import datetime, timezone
from .r11_config import ROOT, RAW, DOCS
from .freeze_r7_phase_release import write_json, sha256

def preserve_sources():
    path = RAW/"protected_prior_results.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from .freeze_r10_budget import verify_freeze, verify_prior_results
    verify_freeze()
    if verify_prior_results()["status"] != "PASS":
        raise RuntimeError("Prior history integrity failed")
    prior = ROOT/"results/raw/r10_wavepacket_budget_diagnosis"
    audit = json.loads((prior/"audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "PASS":
        raise RuntimeError("R10 engineering audit must pass")
    files = json.loads((prior/"protected_prior_results.json").read_text(encoding="utf-8"))["files"]
    files |= json.loads((prior/"artifact_sha256.json").read_text(encoding="utf-8"))["files"]
    # Include all R10 output files, including prior audit snapshots and its own manifests.
    for directory in ("results/raw", "results/figures", "results/logs", "docs/experiments"):
        for p in (ROOT/directory/"r10_wavepacket_budget_diagnosis").rglob("*"):
            if p.is_file():
                files[p.relative_to(ROOT).as_posix()] = sha256(p)
    for p in (ROOT/"R10_WAVEPACKET_BUDGET_DIAGNOSIS_REPORT.md",
              ROOT/"results/raw/r9_wavepacket_periodic_potential/frozen_protocol_and_code.json"):
        files[p.relative_to(ROOT).as_posix()] = sha256(p)
    result = dict(captured_utc=datetime.now(timezone.utc).isoformat(), files=files,
                  r1_through_r10_preserved=True)
    write_json(path, result)
    return result

def verify_prior_results():
    record = json.loads((RAW/"protected_prior_results.json").read_text(encoding="utf-8"))
    changed = [p for p,d in record["files"].items() if not (ROOT/p).is_file() or sha256(ROOT/p) != d]
    return dict(status="PASS" if not changed else "FAIL", checked_files=len(record["files"]), changed=changed)

def science_files():
    from .freeze_r9_wavepacket import science_files as inherited
    names = ("r11_config.py", "r11_models.py", "r11_exact_carrier_diagnostic.py",
        "r11_residual.py", "r11_reference.py", "r11_evaluate.py", "r11_decision.py",
        "train_r11_ablation.py", "freeze_r11_ablation.py", "run_r11_ablation.py")
    files = inherited()+[ROOT/"src"/n for n in names]+[ROOT/"src/r10_norm_diagnostics.py",
        ROOT/"tests/test_r11_ablation.py"]
    files += [DOCS/p for p in ("preregistration.md", "theory_check.md", "figure_contract.md")]
    files += [RAW/p for p in ("protocol.json", "pre_freeze_test_result.json", "smoke_validation.json",
        "reference_validation.json", "reference_test_grid.npz", "reference_fine_grid.npz",
        "history_review.json", "protected_prior_results.json")]
    return sorted(set(files))

def freeze():
    path = RAW/"frozen_protocol_and_code.json"
    if path.exists():
        return verify_freeze()
    preserve_sources()
    if list((RAW/"formal").rglob("recovery_state.pt")):
        raise RuntimeError("Freeze must precede formal training")
    for name in ("reference_validation.json", "pre_freeze_test_result.json", "smoke_validation.json"):
        if json.loads((RAW/name).read_text(encoding="utf-8"))["status"] != "PASS":
            raise RuntimeError("Pre-freeze gate failed: "+name)
    write_json(path, dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        files={p.relative_to(ROOT).as_posix():sha256(p) for p in science_files()}))
    return sha256(path)

def verify_freeze():
    path = RAW/"frozen_protocol_and_code.json"
    for p,d in json.loads(path.read_text(encoding="utf-8"))["files"].items():
        if not (ROOT/p).is_file() or sha256(ROOT/p) != d:
            raise RuntimeError("R11 frozen file changed: "+p)
    return sha256(path)
