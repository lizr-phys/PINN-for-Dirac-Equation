from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .periodic_potential_config import (
    PERIODIC_POTENTIAL_EXPERIMENT,
    PeriodicPotentialConfig,
    ReferenceCalibrationConfig,
)
from .periodic_potential_physics import (
    ReferenceSolution,
    calibration_observables,
    compare_reference_truncations,
    reference_solution_checks,
    solve_reference_eigendecomposition,
)


STRICT_CONVERGENCE_METRIC = "padded_full_trajectory_spinor_relative_l2"


def evaluate_reference_candidate(
    base_cfg: PeriodicPotentialConfig,
    calibration_cfg: ReferenceCalibrationConfig,
    *,
    potential_strength: float,
    t_max: float,
) -> tuple[dict[str, Any], ReferenceSolution, ReferenceSolution]:
    """Evaluate one candidate using reference information only."""

    times = np.linspace(0.0, float(t_max), calibration_cfg.time_points)
    small = solve_reference_eigendecomposition(
        base_cfg,
        times,
        half_width=calibration_cfg.small_half_width,
        potential_strength=potential_strength,
    )
    large = solve_reference_eigendecomposition(
        base_cfg,
        times,
        half_width=calibration_cfg.large_half_width,
        potential_strength=potential_strength,
    )
    small_checks = reference_solution_checks(small)
    large_checks = reference_solution_checks(large)
    small_observables = calibration_observables(small)
    large_observables = calibration_observables(large)
    truncation = compare_reference_truncations(small, large)

    sideband = large_observables["max_total_occupation_outside_center"]
    outer = large_observables["max_outer_layer_total_occupation"]
    truncation_error = truncation[STRICT_CONVERGENCE_METRIC]
    checks = {
        "sideband_occupation_in_closed_interval": bool(
            calibration_cfg.sideband_occupation_min
            <= sideband
            <= calibration_cfg.sideband_occupation_max
        ),
        "large_ladder_outer_occupation_below_threshold": bool(
            outer < calibration_cfg.outer_mode_occupation_max
        ),
        "strict_complex_spinor_truncation_below_threshold": bool(
            truncation_error
            < calibration_cfg.spinor_truncation_relative_l2_max
        ),
        "small_hamiltonian_hermitian": bool(
            small_checks["hermiticity_max_abs"]
            <= calibration_cfg.hermiticity_tolerance
        ),
        "large_hamiltonian_hermitian": bool(
            large_checks["hermiticity_max_abs"]
            <= calibration_cfg.hermiticity_tolerance
        ),
        "small_norm_conserved": bool(
            small_checks["norm_max_drift"] <= calibration_cfg.norm_drift_tolerance
        ),
        "large_norm_conserved": bool(
            large_checks["norm_max_drift"] <= calibration_cfg.norm_drift_tolerance
        ),
        "small_time_reversible": bool(
            small_checks["time_reversal_relative_l2"]
            <= calibration_cfg.time_reversal_tolerance
        ),
        "large_time_reversible": bool(
            large_checks["time_reversal_relative_l2"]
            <= calibration_cfg.time_reversal_tolerance
        ),
    }
    candidate_pass = all(checks.values())
    payload: dict[str, Any] = {
        "potential_strength": float(potential_strength),
        "t_max": float(t_max),
        "small_half_width": calibration_cfg.small_half_width,
        "large_half_width": calibration_cfg.large_half_width,
        "strict_convergence_metric": STRICT_CONVERGENCE_METRIC,
        "strict_convergence_threshold": (
            calibration_cfg.spinor_truncation_relative_l2_max
        ),
        "candidate_pass": candidate_pass,
        "checks": checks,
        "small_reference_checks": small_checks,
        "large_reference_checks": large_checks,
        "small_ladder_observables": small_observables,
        "large_ladder_observables": large_observables,
        "truncation_comparison": truncation,
    }
    return payload, small, large


def calibrate_reference_problem(
    base_cfg: PeriodicPotentialConfig | None = None,
    calibration_cfg: ReferenceCalibrationConfig | None = None,
) -> tuple[dict[str, Any], dict[tuple[float, float], ReferenceSolution]]:
    """Run the preregistered primary/fallback sequence.

    The strict selection metric is the full complex spinor after embedding the
    J=4 trajectory in the J=6 state space.  Modal occupations are recorded as
    diagnostics but are never substituted for the spinor convergence gate.
    """

    cfg = PeriodicPotentialConfig() if base_cfg is None else base_cfg
    protocol = (
        ReferenceCalibrationConfig() if calibration_cfg is None else calibration_cfg
    )
    records: list[dict[str, Any]] = []
    large_references: dict[tuple[float, float], ReferenceSolution] = {}
    selected: dict[str, Any] | None = None
    horizons_evaluated: list[float] = []

    for t_max in (protocol.primary_t_max, protocol.fallback_t_max):
        horizons_evaluated.append(float(t_max))
        horizon_records: list[dict[str, Any]] = []
        for potential_strength in protocol.candidate_potential_strengths:
            record, _small, large = evaluate_reference_candidate(
                cfg,
                protocol,
                potential_strength=potential_strength,
                t_max=t_max,
            )
            records.append(record)
            horizon_records.append(record)
            large_references[(float(t_max), float(potential_strength))] = large
        passing = [record for record in horizon_records if record["candidate_pass"]]
        if passing:
            selected = min(passing, key=lambda item: item["potential_strength"])
            break

    status = "SELECTED" if selected is not None else "NO_STRICT_CANDIDATE"
    payload = {
        "experiment": PERIODIC_POTENTIAL_EXPERIMENT,
        "calibration_status": status,
        "selection_rule": (
            "At T=5 choose the smallest V0 satisfying all gates; only when none "
            "passes, repeat the same gates at T=8. Strict truncation uses the "
            "zero-padded full complex-spinor trajectory relative L2."
        ),
        "strict_convergence_metric": STRICT_CONVERGENCE_METRIC,
        "base_physics": {
            "length": cfg.length,
            "mass": cfg.mass,
            "k0": cfg.k0,
            "coupling_wave_number": cfg.coupling_wave_number,
        },
        "protocol": protocol.as_dict(),
        "horizons_evaluated": horizons_evaluated,
        "selected": None
        if selected is None
        else {
            "potential_strength": selected["potential_strength"],
            "t_max": selected["t_max"],
            "ladder_half_width": protocol.large_half_width,
        },
        "candidates": records,
    }
    return payload, large_references


def _flatten_candidate(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        "potential_strength": record["potential_strength"],
        "t_max": record["t_max"],
        "candidate_pass": record["candidate_pass"],
        "small_half_width": record["small_half_width"],
        "large_half_width": record["large_half_width"],
    }
    for group in (
        "checks",
        "small_reference_checks",
        "large_reference_checks",
        "small_ladder_observables",
        "large_ladder_observables",
        "truncation_comparison",
    ):
        for key, value in record[group].items():
            flat[f"{group}__{key}"] = value
    return flat


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_calibration_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    references: dict[tuple[float, float], ReferenceSolution],
) -> dict[str, Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite calibration evidence: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "calibration.json"
    summary_csv = output_dir / "calibration_summary.csv"
    source_csv = output_dir / "calibration_mode_occupation_source.csv"
    summary_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(summary_csv, [_flatten_candidate(row) for row in payload["candidates"]])

    source_rows: list[dict[str, Any]] = []
    for (t_max, potential_strength), reference in sorted(references.items()):
        small = solve_reference_eigendecomposition(
            PeriodicPotentialConfig(),
            reference.times,
            half_width=int(payload["protocol"]["small_half_width"]),
            potential_strength=potential_strength,
        )
        np.savez_compressed(
            output_dir / f"reference_T{t_max:g}_V{potential_strength:g}.npz",
            t=reference.times,
            small_wave_numbers=small.wave_numbers,
            small_state=small.state,
            small_time_derivative=small.time_derivative,
            large_wave_numbers=reference.wave_numbers,
            large_state=reference.state,
            large_time_derivative=reference.time_derivative,
            large_modal_occupation=reference.modal_occupation,
            large_branch_occupation=reference.branch_occupation,
        )
        populations = reference.modal_occupation
        for time_index, time in enumerate(reference.times):
            for mode_index, ladder_index in enumerate(reference.ladder_indices):
                source_rows.append(
                    {
                        "t_max": t_max,
                        "potential_strength": potential_strength,
                        "t": float(time),
                        "ladder_index": int(ladder_index),
                        "wave_number": float(reference.wave_numbers[mode_index]),
                        "modal_occupation": float(populations[time_index, mode_index]),
                    }
                )
    _write_csv(source_csv, source_rows)
    return {
        "calibration_json": summary_json,
        "calibration_summary_csv": summary_csv,
        "calibration_source_csv": source_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reference-only weak-periodic-potential calibration"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/raw/periodic_potential_coupled_ltpb/calibration"),
    )
    parser.add_argument("--time-points", type=int, default=4001)
    args = parser.parse_args()
    protocol = replace(ReferenceCalibrationConfig(), time_points=args.time_points)
    payload, references = calibrate_reference_problem(
        PeriodicPotentialConfig(), protocol
    )
    paths = write_calibration_outputs(args.output_dir, payload, references)
    print(json.dumps(payload["selected"], sort_keys=True))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
