from __future__ import annotations
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from .r11_config import RAW, ROOT, configuration
from .r9_config import RAW as R9_RAW
from .r10_config import RAW as R10_RAW
from .r9_reference import compare_pair, solve
from .freeze_r7_phase_release import write_json, sha256

def prepare():
    RAW.mkdir(parents=True, exist_ok=True)
    if (RAW/"reference_validation.json").exists():
        return json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    cfg = configuration()
    source = json.loads((R9_RAW/"reference_validation.json").read_text(encoding="utf-8"))
    if source["status"] != "PASS" or source["selected_truncation"] != [-48, 80]:
        raise RuntimeError("R9 verified reference not available")
    times = np.linspace(0, 5, 4001)
    row, small, _ = compare_pair(cfg, replace(cfg, n_min=-64, n_max=96), times)
    with np.load(R9_RAW/"reference_n-48_80.npz") as old:
        source_difference = np.linalg.norm(small["state"]-old["state"])/np.linalg.norm(old["state"])
    if not row["pass_check"] or source_difference > 1e-12:
        raise RuntimeError("Inherited reference revalidation failed; no formal training")
    np.savez_compressed(RAW/"reference_fine_grid.npz", **small)
    np.savez_compressed(RAW/"reference_test_grid.npz", **solve(cfg, np.linspace(0, 5, 201)))
    record = dict(status="PASS", inherited_truncation=True, selected_truncation=[-48, 80],
        revalidation=row, relative_difference_from_r9=float(source_difference),
        source_reference_sha256=sha256(R9_RAW/"reference_n-48_80.npz"),
        source_calibration_sha256=sha256(R9_RAW/"reference_validation.json"),
        before_formal_training=True, no_pinn_based_selection=True)
    write_json(RAW/"reference_validation.json", record)
    reviews = {}
    for tag, filename, data_dir in (
        ("R8", "R8_COUPLING_RECOVERY_REPORT.md", ROOT/"results/raw/r8_coupling_recovery"),
        ("R9", "R9_WAVEPACKET_PERIODIC_POTENTIAL_REPORT.md", R9_RAW),
        ("R10", "R10_WAVEPACKET_BUDGET_DIAGNOSIS_REPORT.md", R10_RAW)):
        path = ROOT/filename
        content = path.read_text(encoding="utf-8")
        csvs = {p.name:dict(rows=len(pd.read_csv(p)), sha256=sha256(p))
                for p in data_dir.glob("*.csv")}
        reviews[tag] = dict(report_sha256=sha256(path), report_characters=len(content),
            raw_csv_tables=csvs, raw_decision=json.loads((data_dir/"decision.json").read_text(encoding="utf-8"))
            if (data_dir/"decision.json").exists() else None)
    write_json(RAW/"history_review.json", reviews)
    return record
