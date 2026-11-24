from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .audit_multimode_ltpb import OUTPUT, ROOT, run_audit
from .multimode_config import MODE_KEYS, MULTIMODE_MODELS, MULTIMODE_SEEDS, MultimodeConfig
from .multimode_models import build_multimode_model, trainable_state_sha256
from .train_multimode_ltpb import paired_initialization_seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _correct_config_check(root: Path) -> tuple[bool, dict[str, Any]]:
    torch.set_default_dtype(torch.float64)
    raw = root / "results" / "raw" / "multimode_ltpb" / "learned"
    problems: list[str] = []
    for model in MULTIMODE_MODELS:
        for seed in MULTIMODE_SEEDS:
            path = raw / model / f"seed_{seed}" / "config.json"
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
                    problems.append(f"{model}/seed_{seed}: {key}")
            if spec.get("model") != model or spec.get("seed") != seed or spec.get("steps") != 5000:
                problems.append(f"{model}/seed_{seed}: run_spec")
            rebuilt = build_multimode_model(model, MultimodeConfig(), paired_initialization_seed(seed))
            observed_hash = trainable_state_sha256(rebuilt)
            if payload.get("trainable_initial_state_sha256") != observed_hash:
                problems.append(f"{model}/seed_{seed}: float64 initial hash")
            lower = path.read_text(encoding="utf-8").lower()
            if any(token in lower for token in ('"energy"', '"energies"', '"frequency"', '"carrier"', '"phase_slope"')):
                problems.append(f"{model}/seed_{seed}: exact frequency leakage")
    return not problems, {
        "problems": problems,
        "correction": "Set torch default dtype to float64 before deterministic initialization rebuild, matching the trainer.",
    }


def _correct_metrics_check(root: Path) -> tuple[bool, dict[str, Any]]:
    raw = root / "results" / "raw" / "multimode_ltpb" / "learned"
    problems: list[str] = []
    maximum_roundtrip_difference = 0.0
    for model in MULTIMODE_MODELS:
        for seed in MULTIMODE_SEEDS:
            run_dir = raw / model / f"seed_{seed}"
            metric = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            history = pd.read_csv(run_dir / "loss_history.csv")
            difference = abs(float(metric["final_loss_pde"]) - float(history.pde.iloc[-1]))
            maximum_roundtrip_difference = max(maximum_roundtrip_difference, difference)
            if not np.isclose(float(metric["final_loss_pde"]), float(history.pde.iloc[-1]), rtol=2e-14, atol=2e-16):
                problems.append(f"{model}/seed_{seed}: final loss")
            with np.load(run_dir / "dense_grid.npz") as data:
                pred = data["psi_predicted_gauge_aligned"]
                exact = data["psi_exact"]
                current_pred = data["current_predicted"]
                current_exact = data["current_exact"]
                rho_pred = data["rho_predicted"]
                rho_exact = data["rho_exact"]
                residual = data["dirac_residual"]
                overlap = 2 * np.pi * np.mean(np.sum(np.conj(exact) * pred, axis=2), axis=1)
                pred_norm = 2 * np.pi * np.mean(np.sum(np.abs(pred) ** 2, axis=2), axis=1)
                exact_norm = 2 * np.pi * np.mean(np.sum(np.abs(exact) ** 2, axis=2), axis=1)
                recomputed = {
                    "spinor_rel_l2_gauge_aligned": np.linalg.norm((pred - exact).ravel()) / np.linalg.norm(exact.ravel()),
                    "current_rel_l2": np.linalg.norm((current_pred - current_exact).ravel()) / np.linalg.norm(current_exact.ravel()),
                    "fidelity_mean": np.mean(np.abs(overlap) ** 2 / (pred_norm * exact_norm)),
                    "density_rel_l2": np.linalg.norm((rho_pred - rho_exact).ravel()) / np.linalg.norm(rho_exact.ravel()),
                    "dense_pde_residual_rms": np.sqrt(np.mean(np.abs(residual) ** 2)),
                }
            for key, value in recomputed.items():
                if not np.isclose(float(metric[key]), float(value), rtol=2e-12, atol=2e-14):
                    problems.append(f"{model}/seed_{seed}: {key}")
    return not problems, {
        "problems": problems,
        "maximum_json_csv_float_roundtrip_difference": maximum_roundtrip_difference,
        "correction": "Use explicit floating-point tolerances instead of bitwise equality after CSV parsing.",
    }


def _correct_aggregate_check(root: Path) -> tuple[bool, dict[str, Any]]:
    results = root / "results"
    raw = results / "raw" / "multimode_ltpb" / "learned"
    runs = pd.read_csv(results / "multimode_ltpb_runs.csv")
    summary = pd.read_csv(results / "multimode_ltpb_summary.csv").set_index("model")
    decision = json.loads((results / "multimode_ltpb_decision.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    maximum_roundtrip_difference = 0.0
    for row in runs.itertuples():
        raw_metric = json.loads((raw / row.model / f"seed_{int(row.seed)}" / "metrics.json").read_text(encoding="utf-8"))
        for key in ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean"):
            difference = abs(float(getattr(row, key)) - float(raw_metric[key]))
            maximum_roundtrip_difference = max(maximum_roundtrip_difference, difference)
            if not np.isclose(float(getattr(row, key)), float(raw_metric[key]), rtol=2e-14, atol=2e-16):
                problems.append(f"runs csv {row.model}/{row.seed}/{key}")
    for model in MULTIMODE_MODELS:
        group = runs[runs.model == model]
        for key in ("spinor_rel_l2_gauge_aligned", "current_rel_l2", "fidelity_mean"):
            if not np.isclose(float(summary.loc[model, f"{key}_mean"]), float(group[key].mean()), rtol=2e-14, atol=2e-16):
                problems.append(f"summary mean {model}/{key}")
            if not np.isclose(float(summary.loc[model, f"{key}_std"]), float(np.std(group[key], ddof=0)), rtol=2e-14, atol=2e-16):
                problems.append(f"summary sd {model}/{key}")

    means = runs.groupby("model").mean(numeric_only=True)
    baselines = [model for model in MULTIMODE_MODELS if model != "multi_ltpb"]
    target = means.loc["multi_ltpb"]
    conditions = [
        target.spinor_rel_l2_gauge_aligned <= 0.05,
        target.current_rel_l2 <= 0.10,
        target.fidelity_mean >= 0.99,
        target.spinor_rel_l2_gauge_aligned <= 0.1 * means.loc[baselines, "spinor_rel_l2_gauge_aligned"].min(),
        target.current_rel_l2 <= 0.1 * means.loc[baselines, "current_rel_l2"].min(),
    ]
    paired = 0
    for seed in MULTIMODE_SEEDS:
        subset = runs[runs.seed == seed].set_index("model")
        row = subset.loc["multi_ltpb"]
        if (
            row.spinor_rel_l2_gauge_aligned <= 0.1 * subset.loc[baselines, "spinor_rel_l2_gauge_aligned"].min()
            and row.current_rel_l2 <= 0.1 * subset.loc[baselines, "current_rel_l2"].min()
        ):
            paired += 1
    phase_pass = 0
    target_runs = runs[runs.model == "multi_ltpb"]
    for mode in MODE_KEYS:
        errors = target_runs[f"{mode}_phase_slope_relative_error"].to_numpy(float)
        valid = target_runs[f"{mode}_phase_valid_fraction"].to_numpy(float)
        if np.all(valid >= 0.9) and np.all(np.isfinite(errors)) and float(np.mean(errors)) < 0.01:
            phase_pass += 1
    overall = bool(all(conditions) and paired >= 4 and phase_pass >= 3)
    if decision["overall_pass"] != overall or decision["paired_joint_pass_count"] != paired or decision["phase_slope_pass_count"] != phase_pass:
        problems.append("decision mismatch")
    return not problems, {
        "problems": problems,
        "maximum_json_csv_float_roundtrip_difference": maximum_roundtrip_difference,
        "independent_overall": overall,
        "paired_joint_pass_count": paired,
        "phase_slope_pass_count": phase_pass,
        "correction": "Use round-trip floating tolerances for JSON-to-CSV comparisons; decision remains independently recomputed.",
    }


def _correct_link_check(root: Path) -> tuple[bool, dict[str, Any]]:
    report = root / "MULTIMODE_LTPB_EXPERIMENT_REPORT.md"
    text = report.read_text(encoding="utf-8")
    pairs = re.findall(r"\[[^\]]*\]\(([^)]+)\)|!\[[^\]]*\]\(([^)]+)\)", text)
    targets = [(left or right).split("#", 1)[0] for left, right in pairs]
    missing: list[str] = []
    for target in targets:
        if not target or re.match(r"^[a-zA-Z]+://", target):
            continue
        candidate = (report.parent / target).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            missing.append(f"outside project: {target}")
            continue
        if not candidate.exists():
            missing.append(target)
    expected_raw = {
        f"results/raw/multimode_ltpb/learned/{model}/seed_{seed}/{name}"
        for model in MULTIMODE_MODELS
        for seed in MULTIMODE_SEEDS
        for name in ("config.json", "metrics.json", "loss_history.csv", "time_series.csv", "dense_grid.npz", "model_state.pt")
    }
    missing_raw_links = sorted(expected_raw.difference(targets))
    return not missing and not missing_raw_links, {
        "checked_links": len([target for target in targets if target]),
        "expected_raw_links": len(expected_raw),
        "missing_targets": missing,
        "missing_raw_links": missing_raw_links,
        "correction": "Verify the complete 150-link raw artifact set directly instead of an unrelated hard-coded total-link count.",
    }


def main() -> None:
    result = run_audit(root=ROOT, output=OUTPUT)
    replacements = {
        "formal_configs_and_no_exact_frequency_leakage": _correct_config_check(ROOT),
        "metrics_independently_recomputed_from_dense_raw": _correct_metrics_check(ROOT),
        "aggregates_population_sd_and_decision_independently_recomputed": _correct_aggregate_check(ROOT),
        "report_relative_links_exist_within_project": _correct_link_check(ROOT),
    }
    for check in result["checks"]:
        if check["name"] in replacements:
            passed, detail = replacements[check["name"]]
            check["pass"] = passed
            check["detail"] = detail
            check["postfreeze_audit_false_positive_corrected"] = True
    result["audit_finalizer"] = "src/finalize_multimode_ltpb_audit.py"
    result["audit_finalizer_sha256"] = _sha256(Path(__file__))
    result["frozen_experiment_code_modified"] = False
    result["correction_scope"] = [
        "audit rebuild dtype",
        "floating point comparison tolerance",
        "direct report raw-link inventory",
    ]
    result["checks_passed"] = sum(int(check["pass"]) for check in result["checks"])
    result["checks_total"] = len(result["checks"])
    result["status"] = "PASS" if all(check["pass"] for check in result["checks"]) else "FAIL"
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed": result["checks_passed"], "total": result["checks_total"]}))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
