from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from .multimode_config import MODE_KEYS, MULTIMODE_MODELS, MULTIMODE_SEEDS, MultimodeConfig
from .multimode_models import build_multimode_model, count_trainable_parameters, trainable_state_sha256
from .train_multimode_ltpb import (
    RUN_ARTIFACTS,
    paired_collocation_seed,
    paired_initialization_seed,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw" / "multimode_ltpb"
FIGURES = RESULTS / "figures" / "multimode_ltpb"
OUTPUT = RESULTS / "multimode_ltpb_audit.json"
EXPECTED_PARAMETERS = {
    "vanilla": 12_996,
    "sp_cpa": 13_061,
    "component_polar": 12_996,
    "multi_branch_complex": 12_860,
    "multi_ltpb": 12_860,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(array: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(array))))


def run_audit(
    *,
    root: Path = ROOT,
    output: Path = OUTPUT,
) -> dict[str, Any]:
    results = root / "results"
    raw = results / "raw" / "multimode_ltpb"
    figures = results / "figures" / "multimode_ltpb"
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    def guarded(name: str, fn: Callable[[], tuple[bool, Any] | bool]) -> None:
        try:
            result = fn()
            if isinstance(result, tuple):
                record(name, bool(result[0]), result[1])
            else:
                record(name, bool(result))
        except Exception as exc:  # audit should produce evidence, not abort
            record(name, False, f"{type(exc).__name__}: {exc}")

    expected_dirs = [raw / "learned" / model / f"seed_{seed}" for model in MULTIMODE_MODELS for seed in MULTIMODE_SEEDS]

    def inventory() -> tuple[bool, Any]:
        actual_dirs = sorted(path for path in (raw / "learned").glob("*/*") if path.is_dir())
        expected_set = {path.resolve() for path in expected_dirs}
        actual_set = {path.resolve() for path in actual_dirs}
        details: dict[str, Any] = {
            "expected_runs": 25,
            "actual_runs": len(actual_dirs),
            "missing": [str(path.relative_to(root)) for path in expected_dirs if path.resolve() not in actual_set],
            "extra": [str(path.relative_to(root)) for path in actual_dirs if path.resolve() not in expected_set],
            "bad_artifacts": {},
        }
        for path in expected_dirs:
            if path.is_dir():
                names = {item.name for item in path.iterdir()}
                if names != set(RUN_ARTIFACTS):
                    details["bad_artifacts"][str(path.relative_to(root))] = sorted(names)
        passed = actual_set == expected_set and not details["bad_artifacts"]
        return passed, details

    guarded("formal_inventory_25_runs_exactly_six_artifacts", inventory)

    def learned_configs() -> tuple[bool, Any]:
        problems: list[str] = []
        initial_hashes: dict[str, str] = {}
        for model in MULTIMODE_MODELS:
            for seed in MULTIMODE_SEEDS:
                path = raw / "learned" / model / f"seed_{seed}" / "config.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                cfg = payload["experiment_config"]
                spec = payload["run_spec"]
                expected = {
                    "mass": 5.0,
                    "wave_numbers": [2.0, 5.0],
                    "t_max": 5.0,
                    "hidden_layers": 4,
                    "collocation_points": 1024,
                    "adam_steps": 5000,
                    "learning_rate": 1e-3,
                    "test_x_points": 128,
                    "test_t_points": 201,
                    "dtype": "float64",
                }
                for key, value in expected.items():
                    if cfg.get(key) != value:
                        problems.append(f"{model}/seed_{seed}: {key}={cfg.get(key)!r}")
                if spec.get("model") != model or spec.get("seed") != seed or spec.get("steps") != 5000:
                    problems.append(f"{model}/seed_{seed}: run_spec mismatch")
                streams = payload["random_streams"]
                if streams.get("model_initialization_seed") != paired_initialization_seed(seed):
                    problems.append(f"{model}/seed_{seed}: initialization seed")
                if streams.get("collocation_seed") != paired_collocation_seed(seed):
                    problems.append(f"{model}/seed_{seed}: collocation seed")
                if payload.get("parameter_count") != EXPECTED_PARAMETERS[model]:
                    problems.append(f"{model}/seed_{seed}: parameter count")
                lower = path.read_text(encoding="utf-8").lower()
                if any(token in lower for token in ('"energy"', '"energies"', '"frequency"', '"carrier"', '"phase_slope"')):
                    problems.append(f"{model}/seed_{seed}: exact-frequency field leaked into config")
                rebuilt = build_multimode_model(model, MultimodeConfig(), paired_initialization_seed(seed))
                rebuilt_hash = trainable_state_sha256(rebuilt)
                if payload.get("trainable_initial_state_sha256") != rebuilt_hash:
                    problems.append(f"{model}/seed_{seed}: initial-state hash mismatch")
                initial_hashes[f"{model}/seed_{seed}"] = rebuilt_hash
        return not problems, {"problems": problems, "initial_hashes": initial_hashes}

    guarded("formal_configs_and_no_exact_frequency_leakage", learned_configs)

    def checkpoints() -> tuple[bool, Any]:
        problems: list[str] = []
        for model in MULTIMODE_MODELS:
            for seed in MULTIMODE_SEEDS:
                path = raw / "learned" / model / f"seed_{seed}" / "model_state.pt"
                payload = torch.load(path, map_location="cpu", weights_only=False)
                keys_text = "\n".join(str(key).lower() for key in payload.keys())
                state_keys = "\n".join(str(key).lower() for key in payload["state_dict"].keys())
                if any(token in keys_text + state_keys for token in ("energy", "frequency", "carrier", "phase_slope")):
                    problems.append(f"{model}/seed_{seed}: forbidden checkpoint key")
                if payload["model"] != model or payload["seed"] != seed or payload["steps"] != 5000:
                    problems.append(f"{model}/seed_{seed}: checkpoint metadata")
                if any(not torch.all(torch.isfinite(value)) for value in payload["state_dict"].values()):
                    problems.append(f"{model}/seed_{seed}: nonfinite state")
        return not problems, problems

    guarded("checkpoint_metadata_state_and_no_frequency_buffers", checkpoints)

    def histories() -> tuple[bool, Any]:
        problems: list[str] = []
        expected_steps = [1] + list(range(10, 5001, 10))
        for model in MULTIMODE_MODELS:
            for seed in MULTIMODE_SEEDS:
                frame = pd.read_csv(raw / "learned" / model / f"seed_{seed}" / "loss_history.csv")
                if len(frame) != 501 or frame.step.tolist() != expected_steps:
                    problems.append(f"{model}/seed_{seed}: cadence rows={len(frame)}")
                if not np.array_equal(frame.total.to_numpy(), frame.pde.to_numpy()):
                    problems.append(f"{model}/seed_{seed}: total != pde")
                if not _finite(frame.select_dtypes(include=[np.number]).to_numpy()):
                    problems.append(f"{model}/seed_{seed}: nonfinite history")
        return not problems, problems

    guarded("loss_history_501_rows_pde_only", histories)

    def dense_artifacts() -> tuple[bool, Any]:
        problems: list[str] = []
        required = {
            "x": (128,),
            "t": (201,),
            "psi_predicted_raw": (201, 128, 2),
            "psi_predicted_gauge_aligned": (201, 128, 2),
            "psi_exact": (201, 128, 2),
            "rho_predicted": (201, 128),
            "rho_exact": (201, 128),
            "current_predicted": (201, 128),
            "current_exact": (201, 128),
            "dirac_residual": (201, 128, 2),
            "mode_coefficients_predicted_gauge_aligned": (201, 4),
            "mode_coefficients_exact": (201, 4),
            "mode_occupations_predicted": (201, 4),
            "mode_occupations_exact": (201, 4),
        }
        for model in MULTIMODE_MODELS:
            for seed in MULTIMODE_SEEDS:
                path = raw / "learned" / model / f"seed_{seed}" / "dense_grid.npz"
                with np.load(path) as data:
                    for key, shape in required.items():
                        if key not in data or data[key].shape != shape:
                            problems.append(f"{model}/seed_{seed}: {key} shape")
                        elif not _finite(data[key]):
                            problems.append(f"{model}/seed_{seed}: {key} nonfinite")
                    if "psi_predicted_raw" in data and "psi_exact" in data:
                        initial_error = float(np.max(np.abs(data["psi_predicted_raw"][0] - data["psi_exact"][0])))
                        if initial_error > 2e-12:
                            problems.append(f"{model}/seed_{seed}: hard IC error={initial_error}")
                    if "mode_occupations_exact" in data:
                        if float(np.max(np.abs(data["mode_occupations_exact"] - 0.25))) > 2e-12:
                            problems.append(f"{model}/seed_{seed}: exact occupations")
                    if model == "multi_ltpb":
                        for key in ("internal_mode_amplitudes", "internal_mode_phases"):
                            if key not in data or data[key].shape != (201, 128, 4):
                                problems.append(f"{model}/seed_{seed}: {key}")
        return not problems, problems

    guarded("dense_grid_shapes_dtypes_finite_hard_initial_condition", dense_artifacts)

    def metrics_consistency() -> tuple[bool, Any]:
        problems: list[str] = []
        metric_keys = (
            "spinor_rel_l2_gauge_aligned",
            "current_rel_l2",
            "fidelity_mean",
            "density_rel_l2",
            "dense_pde_residual_rms",
        )
        for model in MULTIMODE_MODELS:
            for seed in MULTIMODE_SEEDS:
                run_dir = raw / "learned" / model / f"seed_{seed}"
                metric = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                history = pd.read_csv(run_dir / "loss_history.csv")
                with np.load(run_dir / "dense_grid.npz") as data:
                    pred = data["psi_predicted_gauge_aligned"]
                    exact = data["psi_exact"]
                    current_pred = data["current_predicted"]
                    current_exact = data["current_exact"]
                    rho_pred = data["rho_predicted"]
                    rho_exact = data["rho_exact"]
                    residual = data["dirac_residual"]
                    overlap = 2 * math.pi * np.mean(np.sum(np.conj(exact) * pred, axis=2), axis=1)
                    pred_norm = 2 * math.pi * np.mean(np.sum(np.abs(pred) ** 2, axis=2), axis=1)
                    exact_norm = 2 * math.pi * np.mean(np.sum(np.abs(exact) ** 2, axis=2), axis=1)
                    recomputed = {
                        "spinor_rel_l2_gauge_aligned": np.linalg.norm((pred - exact).ravel()) / np.linalg.norm(exact.ravel()),
                        "current_rel_l2": np.linalg.norm((current_pred - current_exact).ravel()) / np.linalg.norm(current_exact.ravel()),
                        "fidelity_mean": np.mean(np.abs(overlap) ** 2 / (pred_norm * exact_norm)),
                        "density_rel_l2": np.linalg.norm((rho_pred - rho_exact).ravel()) / np.linalg.norm(rho_exact.ravel()),
                        "dense_pde_residual_rms": np.sqrt(np.mean(np.abs(residual) ** 2)),
                    }
                for key in metric_keys:
                    if not np.isclose(float(metric[key]), float(recomputed[key]), rtol=2e-12, atol=2e-14):
                        problems.append(f"{model}/seed_{seed}: {key}")
                if not np.isclose(float(metric["final_loss_pde"]), float(history.pde.iloc[-1]), rtol=0, atol=0):
                    problems.append(f"{model}/seed_{seed}: final loss")
        return not problems, problems

    guarded("metrics_independently_recomputed_from_dense_raw", metrics_consistency)

    def aggregates_and_decision() -> tuple[bool, Any]:
        problems: list[str] = []
        runs = pd.read_csv(results / "multimode_ltpb_runs.csv")
        summary = pd.read_csv(results / "multimode_ltpb_summary.csv").set_index("model")
        if len(runs) != 25 or len(summary) != 5:
            problems.append("row counts")
        for row in runs.itertuples():
            raw_metric = json.loads((raw / "learned" / row.model / f"seed_{int(row.seed)}" / "metrics.json").read_text(encoding="utf-8"))
            for key in ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean"):
                if not np.isclose(float(getattr(row, key)), float(raw_metric[key]), rtol=0, atol=0):
                    problems.append(f"runs csv {row.model}/{row.seed}/{key}")
        for model in MULTIMODE_MODELS:
            group = runs[runs.model == model]
            for key in ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean"):
                mean = float(group[key].mean())
                sd = float(np.std(group[key].to_numpy(), ddof=0))
                if not np.isclose(float(summary.loc[model, f"{key}_mean"]), mean, rtol=2e-14, atol=2e-16):
                    problems.append(f"summary mean {model}/{key}")
                if not np.isclose(float(summary.loc[model, f"{key}_std"]), sd, rtol=2e-14, atol=2e-16):
                    problems.append(f"summary sd {model}/{key}")

        decision = json.loads((results / "multimode_ltpb_decision.json").read_text(encoding="utf-8"))
        means = runs.groupby("model").mean(numeric_only=True)
        baselines = [model for model in MULTIMODE_MODELS if model != "multi_ltpb"]
        target = means.loc["multi_ltpb"]
        conditions = [
            float(target.spinor_rel_l2_gauge_aligned) <= 0.05,
            float(target.current_rel_l2) <= 0.10,
            float(target.fidelity_mean) >= 0.99,
            float(target.spinor_rel_l2_gauge_aligned) <= 0.1 * float(means.loc[baselines, "spinor_rel_l2_gauge_aligned"].min()),
            float(target.current_rel_l2) <= 0.1 * float(means.loc[baselines, "current_rel_l2"].min()),
        ]
        paired = 0
        for seed in MULTIMODE_SEEDS:
            subset = runs[runs.seed == seed].set_index("model")
            trow = subset.loc["multi_ltpb"]
            if (
                float(trow.spinor_rel_l2_gauge_aligned) <= 0.1 * float(subset.loc[baselines, "spinor_rel_l2_gauge_aligned"].min())
                and float(trow.current_rel_l2) <= 0.1 * float(subset.loc[baselines, "current_rel_l2"].min())
            ):
                paired += 1
        phase_pass = 0
        target_runs = runs[runs.model == "multi_ltpb"]
        for mode in MODE_KEYS:
            valid = target_runs[f"{mode}_phase_valid_fraction"].to_numpy(float)
            errors = target_runs[f"{mode}_phase_slope_relative_error"].to_numpy(float)
            if np.all(valid >= 0.90) and np.all(np.isfinite(errors)) and float(np.mean(errors)) < 0.01:
                phase_pass += 1
        independent_overall = bool(all(conditions) and paired >= 4 and phase_pass >= 3)
        if int(decision["paired_joint_pass_count"]) != paired:
            problems.append("paired decision")
        if int(decision["phase_slope_pass_count"]) != phase_pass:
            problems.append("phase decision")
        if bool(decision["overall_pass"]) != independent_overall:
            problems.append("overall decision")
        return not problems, {"problems": problems, "independent_overall": independent_overall, "paired": paired, "phase_pass": phase_pass}

    guarded("aggregates_population_sd_and_decision_independently_recomputed", aggregates_and_decision)

    def oracle_isolation() -> tuple[bool, Any]:
        oracle = raw / "oracle"
        expected = {"config.json", "metrics.json", "time_series.csv", "dense_grid.npz"}
        names = {path.name for path in oracle.iterdir()} if oracle.is_dir() else set()
        metrics = json.loads((oracle / "metrics.json").read_text(encoding="utf-8"))
        passed = (
            names == expected
            and metrics.get("excluded_from_learned_ranking") is True
            and float(metrics["spinor_rel_l2_gauge_aligned"]) < 1e-12
            and float(metrics["dense_pde_residual_rms"]) < 1e-12
            and not any((raw / "learned").glob("oracle/*"))
        )
        return passed, {"files": sorted(names), "spinor": metrics.get("spinor_rel_l2_gauge_aligned"), "pde": metrics.get("dense_pde_residual_rms")}

    guarded("oracle_exact_and_excluded_from_learned_ranking", oracle_isolation)

    def figures_check() -> tuple[bool, Any]:
        manifest = json.loads((figures / "figure_manifest.json").read_text(encoding="utf-8"))["figures"]
        qa = json.loads((figures / "figure_qa.json").read_text(encoding="utf-8"))
        problems: list[str] = []
        if len(manifest) != 7:
            problems.append(f"manifest count={len(manifest)}")
        for item in manifest:
            for path_text in [item["source_data"], *item["outputs"].values()]:
                path = root / path_text
                if not path.is_file() or path.stat().st_size == 0:
                    problems.append(f"missing {path_text}")
            source = pd.read_csv(root / item["source_data"])
            if len(source) != int(item["source_rows"]):
                problems.append(f"source rows {item['id']}")
        if qa.get("status") != "PASS" or not qa.get("visual_inspection_recorded"):
            problems.append("figure QA not visually passed")
        if not (figures / "FIGURE_QA.md").is_file():
            problems.append("FIGURE_QA.md missing")
        return not problems, problems

    guarded("figure_manifest_source_formats_and_visual_qa", figures_check)

    def protected_baseline() -> tuple[bool, Any]:
        manifest = json.loads((results / "multimode_ltpb_protected_baseline.json").read_text(encoding="utf-8"))
        problems: list[str] = []
        for relative, info in manifest["files"].items():
            path = root / relative
            if not path.is_file():
                problems.append(f"missing {relative}")
            elif path.stat().st_size != info["size_bytes"] or _sha256(path) != info["sha256"]:
                problems.append(f"changed {relative}")
        return not problems and len(manifest["files"]) == manifest["file_count"], {"checked": len(manifest["files"]), "problems": problems}

    guarded("all_438_protected_prior_files_unchanged", protected_baseline)

    def prereg_hashes() -> tuple[bool, Any]:
        path = results / "multimode_ltpb_preregistration_sha256.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        problems: list[str] = []
        for relative, expected in payload["files"].items():
            candidate = root / relative
            if not candidate.is_file() or _sha256(candidate) != expected:
                problems.append(relative)
        return not problems, {"checked": len(payload["files"]), "problems": problems, "frozen_before_formal_results": payload.get("frozen_before_formal_results")}

    guarded("preregistration_and_code_freeze_hashes", prereg_hashes)

    def report_links() -> tuple[bool, Any]:
        report = root / "MULTIMODE_LTPB_EXPERIMENT_REPORT.md"
        text = report.read_text(encoding="utf-8")
        targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)|!\[[^\]]*\]\(([^)]+)\)", text)
        missing: list[str] = []
        checked = 0
        for pair in targets:
            target = pair[0] or pair[1]
            target = target.split("#", 1)[0]
            if not target or re.match(r"^[a-zA-Z]+://", target):
                continue
            candidate = (report.parent / target).resolve()
            checked += 1
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                missing.append(f"outside project: {target}")
                continue
            if not candidate.exists():
                missing.append(target)
        return not missing and checked >= 175, {"checked": checked, "missing": missing}

    guarded("report_relative_links_exist_within_project", report_links)

    status = "PASS" if checks and all(item["pass"] for item in checks) else "FAIL"
    payload = {
        "audit": "multimode_ltpb",
        "status": status,
        "checks_passed": sum(int(item["pass"]) for item in checks),
        "checks_total": len(checks),
        "checks": checks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the formal Multi-LTPB experiment")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    result = run_audit(root=args.root, output=args.output)
    print(json.dumps({"status": result["status"], "passed": result["checks_passed"], "total": result["checks_total"]}))
    if result["status"] != "PASS" and not args.allow_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
