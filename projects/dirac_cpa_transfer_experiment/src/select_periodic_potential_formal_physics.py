from __future__ import annotations

import json
from pathlib import Path

from .train_periodic_potential_coupled_ltpb import RAW, write_json


def select_authorized_amendment() -> dict:
    original = json.loads((RAW/"calibration/calibration.json").read_text(encoding="utf-8"))
    verified = json.loads((RAW/"reference_verification.json").read_text(encoding="utf-8"))
    rows = [r for r in original["candidates"] if r["t_max"] == 5.0]
    choices = []
    for r in rows:
        diagnostic = next(d for d in verified["independent_matrix_exponential_checks"]
                          if d["t_max"] == 5.0 and d["potential_strength"] == r["potential_strength"])
        checks = dict(r["checks"])
        checks.pop("strict_complex_spinor_truncation_below_threshold")
        error = diagnostic["J6_vs_J8_diagnostic_only"]["padded_full_trajectory_spinor_relative_l2"]
        checks["J6_J8_full_spinor_difference_lt_1e_minus_6"] = error < 1e-6
        if all(checks.values()):
            choices.append((r["potential_strength"], checks, error))
    if not choices:
        raise RuntimeError("Authorized amended calibration also has no candidate")
    strength, checks, error = min(choices, key=lambda item: item[0])
    if strength != 0.2:
        raise RuntimeError("The independently selected strength differs from user-approved V0=0.2")
    result = {"status": "SELECTED", "all_selection_gates_passed": True,
              "authorization": "2026-09-29: 用户明确回复‘允许这一处修订，继续正式实验’",
              "amendment": "PERIODIC_POTENTIAL_COUPLED_LTPB_AMENDMENT_01.md",
              "selected": {"potential_strength": strength, "t_max": 5.0, "ladder_half_width": 6},
              "original_calibration_status": original["calibration_status"],
              "revised_truncation_pair": [6, 8], "revised_truncation_time_points": 8001,
              "truncation_relative_l2": error, "checks": checks}
    destination = RAW/"formal_selection.json"
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, result)
    return result


if __name__ == "__main__":
    print(json.dumps(select_authorized_amendment(), ensure_ascii=False, indent=2))
