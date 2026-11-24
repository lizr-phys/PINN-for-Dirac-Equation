from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from .aggregate_wavepacket_shared_spectral_ltpb import (
    build_wavepacket_comparisons,
    collect_wavepacket_runs,
    summarize_wavepacket_runs,
)
from .freeze_wavepacket_shared_spectral_ltpb_preregistration import (
    aggregate_sha256,
)
from .wavepacket_config import WAVEPACKET_MODELS, WAVEPACKET_SEEDS, WavepacketConfig
from .wavepacket_models import (
    PhysicalComponentPolar,
    PhysicalSinglePhaseCPA,
    PhysicalVanilla,
    SharedSpectralComplex,
    SharedSpectralLTPB,
    build_wavepacket_model,
    count_trainable_parameters,
    trainable_state_sha256,
)
from .wavepacket_shared_spectral_ltpb_decision import (
    evaluate_wavepacket_preregistered_decision,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw" / "wavepacket_shared_spectral_ltpb"
LEARNED = RAW / "learned"
FIGURES = RESULTS / "figures" / "wavepacket_shared_spectral_ltpb"
LOGS = RESULTS / "logs" / "wavepacket_shared_spectral_ltpb"
REPORT = ROOT / "WAVEPACKET_SHARED_SPECTRAL_LTPB_REPORT.md"
OUTPUT = RESULTS / "wavepacket_shared_spectral_ltpb_audit.json"
RUN_ARTIFACTS = {
    "config.json",
    "metrics.json",
    "loss_history.csv",
    "time_series.csv",
    "dense_grid.npz",
    "model_state.pt",
}
FIGURE_NAMES = (
    "primary_error_comparison",
    "per_seed_results",
    "wavepacket_density_spacetime",
    "wavepacket_current_spacetime",
    "branch_center_width",
    "phase_slope_train_midpoint",
    "spectral_coefficient_errors",
    "training_curves",
    "accuracy_time_memory",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_l2(predicted: np.ndarray, exact: np.ndarray) -> float:
    return float(
        np.linalg.norm((predicted - exact).ravel())
        / np.linalg.norm(exact.ravel())
    )


def _numeric_close(left: Any, right: Any, *, atol: float = 2.0e-11) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), rtol=2.0e-10, atol=atol))
    except (TypeError, ValueError):
        return left == right


def _check_frame_close(actual: pd.DataFrame, expected: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if actual.columns.tolist() != expected.columns.tolist() or actual.shape != expected.shape:
        return [
            f"frame schema mismatch actual={actual.shape}/{actual.columns.tolist()} "
            f"expected={expected.shape}/{expected.columns.tolist()}"
        ]
    for column in actual.columns:
        if pd.api.types.is_numeric_dtype(actual[column]) and pd.api.types.is_numeric_dtype(
            expected[column]
        ):
            left = actual[column].to_numpy(dtype=np.float64)
            right = expected[column].to_numpy(dtype=np.float64)
            if not np.allclose(left, right, rtol=2.0e-10, atol=2.0e-11, equal_nan=True):
                problems.append(column)
        elif actual[column].fillna("<NA>").tolist() != expected[column].fillna(
            "<NA>"
        ).tolist():
            problems.append(column)
    return problems


def _write(payload: dict[str, Any]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_audit() -> dict[str, Any]:
    # Ensure the report's link to the audit target resolves during this same run.
    _write({"audit": "wavepacket_shared_spectral_ltpb", "status": "RUNNING"})
    checks: list[dict[str, Any]] = []

    def perform(name: str, callback: Callable[[], Any]) -> None:
        try:
            detail = callback()
            checks.append({"name": name, "pass": True, "detail": detail})
        except Exception as exc:  # audit must preserve all failures in one JSON
            checks.append(
                {
                    "name": name,
                    "pass": False,
                    "detail": {"error_type": type(exc).__name__, "error": str(exc)},
                }
            )

    def inventory() -> dict[str, Any]:
        expected = {
            (model, seed)
            for model in WAVEPACKET_MODELS
            for seed in WAVEPACKET_SEEDS
        }
        actual: set[tuple[str, int]] = set()
        bad: dict[str, Any] = {}
        if LEARNED.is_dir():
            for model_dir in LEARNED.iterdir():
                if not model_dir.is_dir():
                    bad[str(model_dir.relative_to(ROOT))] = "not directory"
                    continue
                for seed_dir in model_dir.iterdir():
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                        bad[str(seed_dir.relative_to(ROOT))] = "unexpected entry"
                        continue
                    seed = int(seed_dir.name.removeprefix("seed_"))
                    actual.add((model_dir.name, seed))
                    names = {path.name for path in seed_dir.iterdir()}
                    if names != RUN_ARTIFACTS:
                        bad[str(seed_dir.relative_to(ROOT))] = {
                            "missing": sorted(RUN_ARTIFACTS - names),
                            "extra": sorted(names - RUN_ARTIFACTS),
                        }
        if actual != expected or bad:
            raise AssertionError(
                f"inventory mismatch missing={sorted(expected-actual)} "
                f"extra={sorted(actual-expected)} bad={bad}"
            )
        staging = [str(path.relative_to(ROOT)) for path in RAW.rglob(".*.staging-*")]
        if staging:
            raise AssertionError(f"staging paths remain: {staging}")
        return {"runs": len(actual), "artifacts": len(actual) * len(RUN_ARTIFACTS)}

    perform("formal_inventory_25_runs_exactly_six_artifacts", inventory)

    def configs_and_leakage() -> dict[str, Any]:
        torch.set_default_dtype(torch.float64)
        expected_counts = {
            "physical_vanilla": 12996,
            "physical_sp_cpa": 13061,
            "physical_component_polar": 12996,
            "shared_spectral_complex": 12932,
            "shared_spectral_ltpb": 12932,
        }
        problems: list[str] = []
        shared_hashes: dict[int, list[str]] = {seed: [] for seed in WAVEPACKET_SEEDS}
        for model in WAVEPACKET_MODELS:
            for seed in WAVEPACKET_SEEDS:
                path = LEARNED / model / f"seed_{seed}" / "config.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                cfg_data = payload["experiment_config"]
                spec = payload["run_spec"]
                streams = payload["random_streams"]
                expected = {
                    "x_min": -8.0 * np.pi,
                    "x_max": 8.0 * np.pi,
                    "t_max": 5.0,
                    "mass": 5.0,
                    "n_min": 0,
                    "n_max": 32,
                    "k0": 2.0,
                    "sigma_k": 0.5,
                    "hidden_layers": 4,
                    "hidden_width": 64,
                    "physical_collocation_points": 1024,
                    "spectral_time_points": 32,
                    "adam_steps": 5000,
                    "learning_rate": 1.0e-3,
                    "test_x_points": 256,
                    "test_t_points": 201,
                }
                for key, value in expected.items():
                    if not _numeric_close(cfg_data.get(key), value):
                        problems.append(f"{model}/{seed} config {key}")
                if spec.get("model") != model or spec.get("seed") != seed or spec.get("steps") != 5000:
                    problems.append(f"{model}/{seed} run spec")
                if payload["training_objective"].get("terms") != ["strong_dirac_pde_residual"]:
                    problems.append(f"{model}/{seed} loss terms")
                if payload["training_objective"].get("supervised_labels") is not False:
                    problems.append(f"{model}/{seed} supervised flag")
                if streams.get("model_initialization_seed") != 200000 + seed:
                    problems.append(f"{model}/{seed} init seed")
                if streams.get("shared_time_seed") != 300000 + seed:
                    problems.append(f"{model}/{seed} time seed")
                expected_x = 400000 + seed if model.startswith("physical_") else None
                if streams.get("physical_x_seed") != expected_x:
                    problems.append(f"{model}/{seed} x seed")
                if int(payload["parameter_count"]) != expected_counts[model]:
                    problems.append(f"{model}/{seed} parameter count")
                rebuilt = build_wavepacket_model(model, WavepacketConfig(), 200000 + seed)
                rebuilt_hash = trainable_state_sha256(rebuilt)
                if rebuilt_hash != payload["trainable_initial_state_sha256"]:
                    problems.append(f"{model}/{seed} initial hash")
                if model.startswith("shared_spectral_"):
                    shared_hashes[seed].append(payload["trainable_initial_state_sha256"])
                serialized = json.dumps(payload).lower()
                forbidden = (
                    "reference_energies",
                    "exact_spectral_state",
                    "phase_slope_target",
                    "exact_time_carrier",
                )
                if any(token in serialized for token in forbidden):
                    problems.append(f"{model}/{seed} forbidden config token")
        for seed, hashes in shared_hashes.items():
            if len(hashes) != 2 or len(set(hashes)) != 1:
                problems.append(f"shared initial pairing seed {seed}")
        learned_classes = (
            PhysicalVanilla,
            PhysicalSinglePhaseCPA,
            PhysicalComponentPolar,
            SharedSpectralComplex,
            SharedSpectralLTPB,
        )
        forbidden_source = (
            "reference_energies",
            "exact_spectral_state",
            "wavepacket_exact_solution",
        )
        for cls in learned_classes:
            source = inspect.getsource(cls).lower()
            for token in forbidden_source:
                if token in source:
                    problems.append(f"{cls.__name__} source token {token}")
        if problems:
            raise AssertionError(problems)
        return {"configs": 25, "shared_pair_initial_hashes": 5}

    perform("formal_configs_random_streams_parameter_budget_and_no_leakage", configs_and_leakage)

    def checkpoints() -> dict[str, Any]:
        problems: list[str] = []
        for model in WAVEPACKET_MODELS:
            for seed in WAVEPACKET_SEEDS:
                path = LEARNED / model / f"seed_{seed}" / "model_state.pt"
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if payload.get("model") != model or payload.get("seed") != seed or payload.get("steps") != 5000:
                    problems.append(f"{model}/{seed} metadata")
                state = payload.get("state_dict", {})
                if not state or any(
                    token in key.lower()
                    for key in state
                    for token in ("energy", "frequency", "carrier", "phase_slope")
                ):
                    problems.append(f"{model}/{seed} state")
                rebuilt = build_wavepacket_model(model, WavepacketConfig(), 200000 + seed)
                if set(state) != set(rebuilt.state_dict()):
                    problems.append(f"{model}/{seed} state schema")
        if problems:
            raise AssertionError(problems)
        return {"checkpoints": 25}

    perform("checkpoint_metadata_state_and_no_frequency_buffers", checkpoints)

    def histories() -> dict[str, Any]:
        expected_steps = sorted({1, 5000} | set(range(10, 5001, 10)))
        problems: list[str] = []
        for model in WAVEPACKET_MODELS:
            for seed in WAVEPACKET_SEEDS:
                frame = pd.read_csv(LEARNED / model / f"seed_{seed}" / "loss_history.csv")
                if frame.columns.tolist() != [
                    "step", "total", "pde", "elapsed_seconds", "collocation_batch_index"
                ]:
                    problems.append(f"{model}/{seed} columns")
                if frame.step.astype(int).tolist() != expected_steps:
                    problems.append(f"{model}/{seed} cadence")
                if not np.all(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy())):
                    problems.append(f"{model}/{seed} finite")
                if not np.allclose(frame.total, frame.pde, rtol=0.0, atol=0.0):
                    problems.append(f"{model}/{seed} pde only")
        if problems:
            raise AssertionError(problems)
        return {"rows_per_run": len(expected_steps), "runs": 25}

    perform("loss_history_501_rows_pde_only", histories)

    def dense_contract() -> dict[str, Any]:
        problems: list[str] = []
        for model in WAVEPACKET_MODELS:
            for seed in WAVEPACKET_SEEDS:
                run = LEARNED / model / f"seed_{seed}"
                with np.load(run / "dense_grid.npz") as dense:
                    expected_shapes = {
                        "x": (256,),
                        "t": (201,),
                        "training_wave_numbers": (33,),
                        "psi_predicted_raw": (201, 256, 2),
                        "psi_exact": (201, 256, 2),
                        "rho_predicted": (201, 256),
                        "current_predicted": (201, 256),
                        "branch_coefficients_predicted_gauge_aligned": (201, 33, 2),
                        "phase_slopes": (33, 2),
                        "phase_slope_relative_errors": (33, 2),
                        "branch_plus_density": (201, 256),
                        "branch_minus_density": (201, 256),
                    }
                    for key, shape in expected_shapes.items():
                        if key not in dense or dense[key].shape != shape:
                            problems.append(f"{model}/{seed} {key} shape")
                        elif not np.all(np.isfinite(dense[key])):
                            problems.append(f"{model}/{seed} {key} finite")
                    ic = np.max(np.abs(dense["psi_predicted_raw"][0] - dense["psi_exact"][0]))
                    if ic > 2.0e-11:
                        problems.append(f"{model}/{seed} hard initial {ic}")
                    if model.startswith("shared_spectral_"):
                        if (
                            "heldout_wave_numbers" not in dense.files
                            or dense["heldout_wave_numbers"].shape != (32,)
                        ):
                            problems.append(f"{model}/{seed} heldout shape")
                        elif np.intersect1d(
                            dense["training_wave_numbers"], dense["heldout_wave_numbers"]
                        ).size:
                            problems.append(f"{model}/{seed} heldout overlap")
                series = pd.read_csv(run / "time_series.csv")
                if len(series) != 201 or not np.all(
                    np.isfinite(series.select_dtypes(include=[np.number]).to_numpy())
                ):
                    problems.append(f"{model}/{seed} time series")
        if problems:
            raise AssertionError(problems)
        return {"runs": 25, "dense_grid": [201, 256], "training_modes": 33, "midpoints": 32}

    perform("dense_shapes_finite_hard_initial_and_heldout_isolation", dense_contract)

    def metric_recompute() -> dict[str, Any]:
        problems: list[str] = []
        maximum_difference = 0.0
        for model in WAVEPACKET_MODELS:
            for seed in WAVEPACKET_SEEDS:
                run = LEARNED / model / f"seed_{seed}"
                metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
                with np.load(run / "dense_grid.npz") as dense:
                    predicted = dense["psi_predicted_gauge_aligned"]
                    exact = dense["psi_exact"]
                    rho = dense["rho_predicted"]
                    rho_exact = dense["rho_exact"]
                    current = dense["current_predicted"]
                    current_exact = dense["current_exact"]
                    coeff = dense["branch_coefficients_predicted_gauge_aligned"]
                    coeff_exact = dense["branch_coefficients_exact"]
                    occ = dense["branch_occupations_predicted"]
                    occ_exact = dense["branch_occupations_exact"]
                    residual = dense["dirac_residual_physical"]
                    overlap = WavepacketConfig().length * np.mean(
                        np.sum(np.conj(exact) * predicted, axis=2), axis=1
                    )
                    pred_norm = WavepacketConfig().length * np.mean(
                        np.sum(np.abs(predicted) ** 2, axis=2), axis=1
                    )
                    exact_norm = WavepacketConfig().length * np.mean(
                        np.sum(np.abs(exact) ** 2, axis=2), axis=1
                    )
                    fidelity = np.clip(
                        np.abs(overlap) ** 2 / np.maximum(pred_norm * exact_norm, 1e-30),
                        0.0,
                        1.0,
                    )
                    recomputed = {
                        "spinor_rel_l2_gauge_aligned": _relative_l2(predicted, exact),
                        "density_rel_l2": _relative_l2(rho, rho_exact),
                        "current_rel_l2": _relative_l2(current, current_exact),
                        "fidelity_mean": float(np.mean(fidelity)),
                        "spectral_coefficient_rel_l2_gauge_aligned": _relative_l2(coeff, coeff_exact),
                        "spectral_occupation_rel_l2": _relative_l2(occ, occ_exact),
                        "dense_physical_pde_residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
                        "max_norm_drift": float(np.max(np.abs(pred_norm - 1.0))),
                    }
                    for key, value in recomputed.items():
                        difference = abs(float(metrics[key]) - value)
                        maximum_difference = max(maximum_difference, difference)
                        if not _numeric_close(metrics[key], value, atol=3.0e-11):
                            problems.append(f"{model}/{seed} {key}: {metrics[key]} vs {value}")
                for memory_key in ("peak_rss_mb", "peak_rss_increment_mb", "rss_peak_training_bytes"):
                    if not np.isfinite(float(metrics[memory_key])) or float(metrics[memory_key]) < 0:
                        problems.append(f"{model}/{seed} {memory_key}")
        if problems:
            raise AssertionError(problems)
        return {"runs": 25, "maximum_absolute_difference": maximum_difference}

    perform("metrics_independently_recomputed_from_dense_raw", metric_recompute)

    def parseval() -> dict[str, Any]:
        problems: list[str] = []
        max_difference = 0.0
        cfg = WavepacketConfig()
        for model in ("shared_spectral_complex", "shared_spectral_ltpb"):
            for seed in WAVEPACKET_SEEDS:
                run = LEARNED / model / f"seed_{seed}"
                with np.load(run / "dense_grid.npz") as dense:
                    spectral = dense["dirac_residual_spectral_or_inband_projection"]
                    physical = dense["dirac_residual_physical"]
                    spectral_rms = float(
                        np.sqrt(np.mean(np.sum(np.abs(spectral) ** 2, axis=1)) / cfg.length)
                    )
                    physical_rms = float(np.sqrt(np.mean(np.abs(physical) ** 2)))
                    difference = abs(spectral_rms - physical_rms)
                    max_difference = max(max_difference, difference)
                    if difference > 2.0e-11:
                        problems.append(f"{model}/{seed}: {difference}")
        if problems:
            raise AssertionError(problems)
        return {"spectral_runs": 10, "maximum_rms_difference": max_difference}

    perform("spectral_physical_parseval_residual_equivalence", parseval)

    def aggregates() -> dict[str, Any]:
        runs = collect_wavepacket_runs()
        summary = summarize_wavepacket_runs(runs)
        comparisons = build_wavepacket_comparisons(runs)
        actual_runs = pd.read_csv(RESULTS / "wavepacket_shared_spectral_ltpb_runs.csv")
        actual_summary = pd.read_csv(RESULTS / "wavepacket_shared_spectral_ltpb_summary.csv")
        actual_comparisons = pd.read_csv(RESULTS / "wavepacket_shared_spectral_ltpb_comparisons.csv")
        problems = {
            "runs": _check_frame_close(actual_runs, runs),
            "summary": _check_frame_close(actual_summary, summary),
            "comparisons": _check_frame_close(actual_comparisons, comparisons),
        }
        if any(problems.values()):
            raise AssertionError(problems)
        expected_decision = evaluate_wavepacket_preregistered_decision(runs)
        actual_decision = json.loads(
            (RESULTS / "wavepacket_shared_spectral_ltpb_decision.json").read_text(encoding="utf-8")
        )
        if expected_decision != actual_decision:
            raise AssertionError("decision JSON differs from recomputation")
        target = runs[runs.model == "shared_spectral_ltpb"]
        required_finite = [
            "spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean",
            "training_phase_slope_weighted_relative_error",
            "heldout_phase_slope_weighted_relative_error",
            "branch_plus_center_rmse", "branch_minus_center_rmse",
            "branch_plus_width_rel_l2", "branch_minus_width_rel_l2",
        ]
        if not np.all(np.isfinite(target[required_finite].to_numpy(dtype=np.float64))):
            raise AssertionError("target finite-value contract failed")
        return {
            "runs": 25,
            "population_sd_ddof": 0,
            "overall_pass": actual_decision["overall_pass"],
        }

    perform("aggregates_population_sd_and_decision_recomputed", aggregates)

    def oracle() -> dict[str, Any]:
        expected = {"config.json", "metrics.json", "time_series.csv", "dense_grid.npz"}
        names = {path.name for path in (RAW / "oracle").iterdir()}
        if names != expected:
            raise AssertionError(f"oracle files {names}")
        config = json.loads((RAW / "oracle" / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((RAW / "oracle" / "metrics.json").read_text(encoding="utf-8"))
        if not config.get("uses_exact_dispersion") or not metrics.get("excluded_from_learned_ranking"):
            raise AssertionError("oracle role flags")
        for key in ("spinor_rel_l2_gauge_aligned", "density_rel_l2", "current_rel_l2"):
            if abs(float(metrics[key])) > 2.0e-12:
                raise AssertionError(f"oracle {key}={metrics[key]}")
        if float(metrics["dense_physical_pde_residual_rms"]) > 2.0e-12:
            raise AssertionError("oracle residual")
        return {
            "spinor": metrics["spinor_rel_l2_gauge_aligned"],
            "pde": metrics["dense_physical_pde_residual_rms"],
        }

    perform("spectral_oracle_exact_and_excluded_from_ranking", oracle)

    def figures() -> dict[str, Any]:
        manifest = json.loads((FIGURES / "figure_manifest.json").read_text(encoding="utf-8"))
        qa = json.loads((FIGURES / "figure_qa.json").read_text(encoding="utf-8"))
        problems: list[str] = []
        for name in FIGURE_NAMES:
            for suffix in (".png", ".svg", ".pdf"):
                path = FIGURES / f"{name}{suffix}"
                if not path.is_file() or path.stat().st_size == 0:
                    problems.append(str(path.relative_to(ROOT)))
            source = FIGURES / "source_data" / f"{name}.csv"
            if not source.is_file() or source.stat().st_size == 0:
                problems.append(str(source.relative_to(ROOT)))
        manifest_entries = manifest.get("figures", [])
        if isinstance(manifest_entries, dict):
            manifest_names = set(manifest_entries)
        else:
            manifest_names = {
                str(item.get("id")) for item in manifest_entries if isinstance(item, dict)
            }
        if manifest_names != set(FIGURE_NAMES):
            problems.append(f"manifest names {manifest_names}")
        if qa.get("status") != "PASS" or not (FIGURES / "FIGURE_QA.md").is_file():
            problems.append("figure QA")
        if problems:
            raise AssertionError(problems)
        return {"figures": len(FIGURE_NAMES), "formats": ["PNG", "SVG", "PDF"], "qa": "PASS"}

    perform("figure_manifest_source_formats_and_visual_qa", figures)

    def protected() -> dict[str, Any]:
        manifest = json.loads(
            (RESULTS / "wavepacket_shared_spectral_ltpb_protected_baseline.json").read_text(
                encoding="utf-8"
            )
        )
        problems: list[str] = []
        aggregate = hashlib.sha256()
        for relative, record in sorted(manifest["files"].items()):
            path = ROOT / relative
            if not path.is_file():
                problems.append(f"missing {relative}")
                continue
            digest = _sha256(path)
            if digest != record["sha256"] or path.stat().st_size != record["size_bytes"]:
                problems.append(f"changed {relative}")
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(bytes.fromhex(digest))
        if aggregate.hexdigest() != manifest["aggregate_sha256"]:
            problems.append("aggregate")
        if len(manifest["files"]) != 670:
            problems.append(f"file count {len(manifest['files'])}")
        if problems:
            raise AssertionError(problems[:30])
        return {"checked": len(manifest["files"]), "aggregate_sha256": manifest["aggregate_sha256"]}

    perform("all_670_protected_prior_files_unchanged", protected)

    def freeze() -> dict[str, Any]:
        path = RESULTS / "wavepacket_shared_spectral_ltpb_preregistration_sha256.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        problems: list[str] = []
        observed: dict[str, str] = {}
        for relative, expected_hash in manifest["files"].items():
            target = ROOT / relative
            if not target.is_file():
                problems.append(f"missing {relative}")
                continue
            observed[relative] = _sha256(target)
            if observed[relative] != expected_hash:
                problems.append(f"changed {relative}")
        if aggregate_sha256(observed) != manifest["aggregate_sha256"]:
            problems.append("aggregate")
        if not manifest.get("frozen_before_formal_results") or manifest.get("formal_files_present_at_freeze") != 0:
            problems.append("freeze timing")
        if problems:
            raise AssertionError(problems)
        return {"checked": len(observed), "aggregate_sha256": manifest["aggregate_sha256"]}

    perform("preregistration_and_formal_code_freeze_hashes", freeze)

    def logs() -> dict[str, Any]:
        problems: list[str] = []
        paths = sorted(LOGS.glob("*.log"))
        if len(paths) < 5:
            problems.append(f"only {len(paths)} logs")
        text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in paths)
        if "Traceback (most recent call last)" in text or "Non-finite" in text:
            problems.append("failure text in logs")
        for seed in WAVEPACKET_SEEDS:
            if not any(f"seed_{seed}" in path.name for path in paths):
                problems.append(f"missing seed {seed} log")
        if problems:
            raise AssertionError(problems)
        return {"log_files": len(paths)}

    perform("formal_logs_complete_without_tracebacks", logs)

    def report_links() -> dict[str, Any]:
        text = REPORT.read_text(encoding="utf-8")
        links = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
        missing: list[str] = []
        for raw_link in links:
            link = raw_link.strip().split("#", 1)[0]
            if not link or "://" in link:
                continue
            target = (REPORT.parent / link).resolve()
            if not target.exists():
                missing.append(raw_link)
        expected_raw = {
            f"results/raw/wavepacket_shared_spectral_ltpb/learned/{model}/seed_{seed}/{artifact}"
            for model in WAVEPACKET_MODELS
            for seed in WAVEPACKET_SEEDS
            for artifact in RUN_ARTIFACTS
        }
        missing_raw = sorted(link for link in expected_raw if link not in text)
        if missing or missing_raw:
            raise AssertionError({"missing_targets": missing, "missing_raw_links": missing_raw})
        return {"checked_links": len(links), "expected_raw_links": len(expected_raw)}

    perform("report_links_exist_including_all_150_raw_artifacts", report_links)

    passed = sum(int(check["pass"]) for check in checks)
    payload = {
        "audit": "wavepacket_shared_spectral_ltpb",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
    }
    _write(payload)
    return payload


def main() -> None:
    result = run_audit()
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["checks_passed"],
                "total": result["checks_total"],
            }
        )
    )


if __name__ == "__main__":
    main()
