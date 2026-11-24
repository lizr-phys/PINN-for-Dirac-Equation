from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .r8_config import (RAW, CANDIDATES, CONTROLS, EXPLORATION_SEEDS,
                        VALIDATION_SEEDS, THRESHOLDS)
from .freeze_r7_phase_release import write_json


def read_stage(stage):
    records = []
    for p in sorted((RAW/stage).rglob("metrics.json")):
        if "checkpoints" in p.parts:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        r["run_path"] = p.parent.relative_to(RAW).as_posix()
        records.append(r)
    return pd.DataFrame(records)


def joint_mask(frame):
    result = np.ones(len(frame), dtype=bool)
    for metric, value in THRESHOLDS.items():
        result &= (frame[metric] >= value if metric == "fidelity_mean" else frame[metric] <= value).to_numpy()
    return result


def absolute_success(frame, seeds=VALIDATION_SEEDS, minimum=4):
    if tuple(sorted(frame.seed)) != tuple(seeds):
        return dict(pass_check=False, complete=False, joint_pass_count=0)
    checks = {m: bool(frame[m].mean() >= v if m == "fidelity_mean" else frame[m].mean() <= v)
              for m, v in THRESHOLDS.items()}
    passed = joint_mask(frame)
    return dict(pass_check=all(checks.values()) and int(passed.sum()) >= minimum,
                complete=True, mean_checks=checks, joint_pass_count=int(passed.sum()),
                per_seed_joint_pass={str(s): bool(ok) for s, ok in zip(frame.seed, passed)})


def score(frame):
    return float((frame.spinor_rel_l2_gauge_aligned/.10+
                  frame.current_rel_l2/.15+frame.sideband_complex_coefficient_rel_l2/.10).mean()/3)


def select_candidates():
    destination = RAW/"exploration_selection.json"
    if destination.exists():
        return json.loads(destination.read_text(encoding="utf-8"))
    frame = read_stage("exploration")
    if len(frame) != 18:
        raise RuntimeError("Selection requires all 18 exploration runs")
    ranks = []
    for name in CANDIDATES:
        group = frame[frame.model == name]
        if tuple(sorted(group.seed)) != EXPLORATION_SEEDS:
            raise RuntimeError("Exploration seeds incomplete")
        ranks.append(dict(model=name, score=score(group),
                          exploration_absolute=absolute_success(group, EXPLORATION_SEEDS, 2)))
    ranks.sort(key=lambda row: (row["score"], CANDIDATES.index(row["model"])))
    result = dict(stage="EXPLORATORY_SELECTION_ONLY", selection_formula="mean((spinor/.10 + current/.15 + sideband_complex/.10)/3)",
                  ranking=ranks, selected_candidates=[r["model"] for r in ranks[:2]],
                  validation_seeds=list(VALIDATION_SEEDS), thresholds_unchanged=THRESHOLDS)
    write_json(destination, result)
    return result


def validation_decision():
    selection = json.loads((RAW/"exploration_selection.json").read_text(encoding="utf-8"))
    frame = read_stage("validation")
    expected = selection["selected_candidates"]+list(CONTROLS)
    if len(frame) != 5*len(expected) or set(frame.model) != set(expected):
        raise RuntimeError("Independent validation cohort is incomplete")
    groups = {name: absolute_success(frame[frame.model == name]) for name in expected}
    successful = [n for n in selection["selected_candidates"] if groups[n]["pass_check"]]
    successful.sort(key=lambda n: (score(frame[frame.model == n]), CANDIDATES.index(n)))
    result = dict(stage="INDEPENDENT_VALIDATION", seeds=list(VALIDATION_SEEDS),
                  classification="COUPLING_RECOVERY_SUCCESS" if successful else "NO_VALIDATED_RECOVERY",
                  groups=groups, successful_candidates=successful,
                  exploratory_selection_does_not_imply_validation=True)
    write_json(RAW/"decision.json", result)
    return result


def conditional_plan():
    decision = validation_decision()
    selected = json.loads((RAW/"exploration_selection.json").read_text(encoding="utf-8"))["selected_candidates"]
    ablations = []
    mapping = {"A_carrier_complex": "A_identity_carrier", "B_free_correction": "B_static_free",
               "C_release_windows": "window_complex", "D_carrier_windows": "D_identity_carrier"}
    for method in decision["successful_candidates"]:
        ablations.append(mapping[method])
        if method == "D_carrier_windows" and "A_carrier_complex" not in selected:
            ablations.append("A_carrier_complex")
    plan = dict(ablations=list(dict.fromkeys(ablations)),
                extension_models=decision["successful_candidates"]+["complex"] if decision["successful_candidates"] else [],
                extension_cases=["T8", "V01", "V04", "free_packet"] if decision["successful_candidates"] else [],
                rescue_model=None)
    if not decision["successful_candidates"]:
        frame = read_stage("validation")
        plan["rescue_model"] = min(selected, key=lambda n: (score(frame[frame.model == n]), CANDIDATES.index(n)))
    write_json(RAW/"conditional_execution_plan.json", plan)
    return plan
