from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PRINCIPAL_MODELS, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"

EXPECTED_ORIGINAL_HASHES = {
    "main.py": "A5201276944CD5707B0FDC75BCD5887494BD91F7D04AC48F667A00F2928AA67E",
    "README.md": "3473A1932AD49A0D9D34A57405E4B95C67E7E6A78BD3782D0A7F890C082E64BE",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main_run_dir(case: str, model: str, seed: int) -> Path:
    return RESULTS_ROOT / "raw" / "main" / f"case_{case}" / model / f"seed_{seed}"


def main() -> None:
    failures: list[str] = []
    summary = pd.read_csv(RESULTS_ROOT / "summary.csv")
    main_rows = summary[summary.experiment.eq("main")]
    sensitivity_rows = summary[summary.experiment.eq("sensitivity")]
    if len(summary) != 23:
        failures.append(f"summary rows: expected 23, found {len(summary)}")
    if len(main_rows) != 20:
        failures.append(f"main rows: expected 20, found {len(main_rows)}")
    if sorted(sensitivity_rows.lambda_struct.tolist()) != [0.1, 1.0, 10.0]:
        failures.append("sensitivity lambda set is not exactly [0.1, 1, 10]")

    required_run_files = [
        "config.json",
        "metrics.json",
        "loss_history.csv",
        "time_series.csv",
        "dense_grid.npz",
        "model_state.pt",
    ]
    run_specs = []
    for case in ("P", "M"):
        for model in PRINCIPAL_MODELS:
            run_specs.extend((case, model, seed) for seed in (0, 1, 2))
        run_specs.append((case, "two_branch", 0))
    config_checks = []
    for case, model, seed in run_specs:
        run_dir = main_run_dir(case, model, seed)
        missing = [name for name in required_run_files if not (run_dir / name).exists()]
        if missing:
            failures.append(f"{run_dir}: missing {missing}")
            continue
        with (run_dir / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        exp = config["experiment_config"]
        run = config["run_spec"]
        checks = {
            "case": run["case"] == case,
            "model": run["model"] == model,
            "seed": run["seed"] == seed,
            "steps": run["steps"] == 5000,
            "collocation": exp["collocation_points"] == 1024,
            "hidden_layers": exp["hidden_layers"] == 4,
            "hidden_width": exp["hidden_width"] == 64,
            "dtype": exp["dtype"] == "float64",
            "m": bool(np.isclose(exp["mass"], 5.0)),
            "k": bool(np.isclose(exp["wave_number"], 2.0)),
            "T": bool(np.isclose(exp["t_max"], 5.0)),
        }
        if not all(checks.values()):
            failures.append(f"{run_dir}: config mismatch {checks}")
        config_checks.append({"case": case, "model": model, "seed": seed, **checks})

    numeric = main_rows.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        failures.append("main summary contains non-finite numeric values")

    figure_names = [
        "01_main_metric_comparison",
        "02_case_M_current_x0",
        "03_case_M_relative_phase_x0",
        "04_case_M_spacetime_error_maps",
        "05_case_M_conservation_diagnostics",
        "06_training_loss_curves",
        "07_lambda_struct_sensitivity",
        "08_gradient_cosine_similarity",
    ]
    figure_checks = []
    for name in figure_names:
        present = {}
        for extension in ("svg", "pdf", "png"):
            path = RESULTS_ROOT / "figures" / f"{name}.{extension}"
            present[extension] = path.exists() and path.stat().st_size > 0
        svg_path = RESULTS_ROOT / "figures" / f"{name}.svg"
        svg_text_nodes = svg_path.read_text(encoding="utf-8").count("<text") if svg_path.exists() else 0
        present["svg_text_nodes"] = svg_text_nodes
        if not all(present[ext] for ext in ("svg", "pdf", "png")) or svg_text_nodes == 0:
            failures.append(f"figure {name}: failed export/editable-text check {present}")
        figure_checks.append({"name": name, **present})

    original_dir = PROJECT_ROOT.parent / "dirac_pinn_simple"
    original_hashes = {
        name: sha256(original_dir / name) for name in EXPECTED_ORIGINAL_HASHES
    }
    if original_hashes != EXPECTED_ORIGINAL_HASHES:
        failures.append(f"dirac_pinn_simple hashes changed: {original_hashes}")

    analytic = json.loads(
        (RESULTS_ROOT / "raw" / "analytic_validation.json").read_text(encoding="utf-8")
    )
    analytic_thresholds = {
        "eigen_equation": analytic["eigen_equation_max_error"] < 1.0e-10,
        "pde": analytic["exact_pde_residual_max_abs"] < 1.0e-10,
        "continuity": analytic["exact_continuity_max_abs"] < 1.0e-10,
        "norm": analytic["exact_norm_max_drift"] < 1.0e-10,
        "energy": analytic["exact_energy_max_drift"] < 1.0e-10,
        "current_oscillation": analytic["case_m_current_peak_to_peak"] > 0.25,
    }
    if not all(analytic_thresholds.values()):
        failures.append(f"analytic validation threshold failed: {analytic_thresholds}")

    report = (PROJECT_ROOT / "EXPERIMENT_REPORT.md").read_text(encoding="utf-8")
    forbidden_placeholders = [
        "<!-- PER_SEED_RESULTS -->",
        "<!-- AGGREGATE_RESULTS -->",
        "<!-- SENSITIVITY_RESULTS -->",
        "<!-- GRADIENT_RESULTS -->",
        "<!-- INTERPRETATION -->",
        "<!-- HYPOTHESIS_DECISION -->",
        "<!-- NEXT_STEPS -->",
    ]
    report_complete = not any(marker in report for marker in forbidden_placeholders)
    if not report_complete:
        failures.append("report contains unfilled placeholders")

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "counts": {
            "summary_rows": len(summary),
            "main_runs": len(main_rows),
            "sensitivity_rows_including_reuse": len(sensitivity_rows),
            "main_config_checks": len(config_checks),
            "figure_groups": len(figure_checks),
            "figure_exports": len(figure_checks) * 3,
        },
        "analytic_thresholds": analytic_thresholds,
        "dirac_pinn_simple_sha256": original_hashes,
        "expected_dirac_pinn_simple_sha256": EXPECTED_ORIGINAL_HASHES,
        "report_complete": report_complete,
        "config_checks": config_checks,
        "figure_checks": figure_checks,
    }
    write_json(RESULTS_ROOT / "reproducibility_audit.json", payload)
    print(json.dumps(payload["counts"], indent=2))
    print(f"status={payload['status']}")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
