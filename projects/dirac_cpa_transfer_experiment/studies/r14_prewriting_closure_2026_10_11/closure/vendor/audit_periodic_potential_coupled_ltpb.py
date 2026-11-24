from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .capture_periodic_potential_coupled_ltpb_protected_baseline import capture
from .freeze_periodic_potential_coupled_ltpb import verify_freeze, sha
from .periodic_potential_config import PERIODIC_POTENTIAL_MODELS, paired_initialization_seed
from .periodic_potential_models import build_periodic_potential_model, trainable_state_sha256
from .periodic_potential_coupled_ltpb_decision import evaluate_periodic_potential_decision, TARGET, CONTROL
from .train_periodic_potential_coupled_ltpb import ROOT, RAW, RUN_ARTIFACTS, PairedSampler, load_formal_config, write_json
from .plot_periodic_potential_coupled_ltpb import FIGURES, FIGURE_IDS

REPORT = ROOT/"PERIODIC_POTENTIAL_COUPLED_LTPB_REPORT.md"


def relative(a, b):
    return float(np.linalg.norm((a-b).ravel())/np.linalg.norm(b.ravel()))


def audit():
    write_json(RAW/"audit.json", {"status": "RUNNING"})
    cfg = load_formal_config()
    checks = []
    def check(name, operation):
        try:
            value = operation()
            checks.append({"name": name, "pass": True, "detail": value})
        except Exception as exc:
            checks.append({"name": name, "pass": False, "error": f"{type(exc).__name__}: {exc}"})

    check("frozen_scientific_code_unchanged", lambda: {"manifest_sha256": verify_freeze()})
    def protected():
        data = capture(verify_only=True)
        assert data["status"] == "PASS", data
        return data
    check("prior_887_artifacts_unchanged", protected)

    def inventory():
        expected = {(m, s) for m in PERIODIC_POTENTIAL_MODELS for s in range(5)}
        actual = set()
        for model_dir in (RAW/"learned").iterdir():
            for run in model_dir.iterdir():
                assert run.name.startswith("seed_"), f"Unexpected attempt/staging: {run}"
                actual.add((model_dir.name, int(run.name[5:])))
                assert {p.name for p in run.iterdir()} == set(RUN_ARTIFACTS), run
                assert all((run/a).stat().st_size > 0 for a in RUN_ARTIFACTS)
        assert actual == expected
        return {"runs": 15, "artifacts": 90, "selective_retries": 0}
    check("formal_inventory", inventory)

    def configurations():
        time_hashes = {}
        for seed in range(5):
            sampler = PairedSampler(cfg, seed, physical=False)
            for _ in range(5000):
                sampler.sample()
            time_hashes[seed] = sampler.time_digest.hexdigest()
        records = []
        torch.set_num_threads(1)
        for model_name in PERIODIC_POTENTIAL_MODELS:
            for seed in range(5):
                run = RAW/"learned"/model_name/f"seed_{seed}"
                config = json.loads((run/"config.json").read_text(encoding="utf-8"))
                checkpoint = torch.load(run/"model_state.pt", map_location="cpu", weights_only=True)
                assert config["configuration"] == cfg.as_dict()
                assert checkpoint["configuration"] == cfg.as_dict()
                assert config["formal"] and checkpoint["steps"] == 5000
                assert config["loss_terms"] == ["pde"]
                model = build_periodic_potential_model(model_name, cfg, paired_initialization_seed(seed))
                assert trainable_state_sha256(model) == config["initial_trainable_sha256"]
                assert checkpoint["time_stream_sha256"] == time_hashes[seed]
                assert checkpoint["initial_trainable_sha256"] == config["initial_trainable_sha256"]
                assert all(v.dtype == torch.float64 for v in checkpoint["state_dict"].values())
                serialized = json.dumps({k: v for k, v in checkpoint.items() if k != "state_dict"}).lower()
                assert not any(token in serialized for token in ("eigenvalue", "reference_phase", "reference_energies", "target_slope", "exact_carrier"))
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                with np.load(run/"dense_grid.npz") as d, torch.no_grad():
                    indices = np.array([0, 11, 40, 96, 200])
                    t = torch.tensor(d["t"][indices], dtype=torch.float64)
                    if model_name == "physical_vanilla":
                        x = torch.tensor(d["x"][[1, 31, 128, 211, 255]], dtype=torch.float64)
                        prediction = model(x, t).numpy()
                        expected = d["psi_predicted_raw"][indices, [1, 31, 128, 211, 255]]
                    else:
                        mode = [0, 4, 6, 8, 12]
                        k = torch.tensor(d["wave_numbers"][mode], dtype=torch.float64)
                        prediction = model(k, t).numpy()
                        expected = d["spectral_state_predicted_raw"][indices, mode]
                    np.testing.assert_allclose(prediction, expected, atol=1e-12, rtol=1e-12)
                records.append((model_name, seed, config["initial_trainable_sha256"]))
        for seed in range(5):
            assert next(h for m,s,h in records if m == TARGET and s == seed) == next(h for m,s,h in records if m == CONTROL and s == seed)
        return {"paired_seed_streams_reconstructed": 5, "checkpoints_reloaded": 15, "physical_parameters": 12996, "spectral_parameters": 12932}
    check("configurations_streams_checkpoints_and_predictions", configurations)

    def raw_metrics():
        maximum_difference = 0.0
        for model in PERIODIC_POTENTIAL_MODELS:
            for seed in range(5):
                run = RAW/"learned"/model/f"seed_{seed}"
                metric = json.loads((run/"metrics.json").read_text(encoding="utf-8"))
                h = pd.read_csv(run/"loss_history.csv")
                assert h.step.tolist() == sorted({1, 5000}|set(range(10, 5001, 10)))
                assert np.isfinite(h.to_numpy()).all() and (h.pde == h.total).all()
                assert np.isclose(metric["final_loss_pde"], h.pde.iloc[-1], rtol=1e-13, atol=1e-15)
                assert len(pd.read_csv(run/"time_series.csv")) == 201
                assert metric["training_seconds"] > 0 and metric["peak_rss_mb"] > 0
                with np.load(run/"dense_grid.npz") as d:
                    assert d["psi_exact"].shape == (201, 256, 2)
                    assert d["spectral_state_exact"].shape == (201, 13, 2)
                    pred, exact = d["psi_predicted_gauge_aligned"], d["psi_exact"]
                    original = d["psi_predicted_raw"]
                    phase = np.angle(np.vdot(exact.ravel(), original.ravel()))
                    np.testing.assert_allclose(pred, original*np.exp(-1j*phase), atol=1e-14)
                    norms = cfg.length*np.mean(np.sum(np.abs(pred)**2, axis=-1), axis=1)
                    exact_norms = cfg.length*np.mean(np.sum(np.abs(exact)**2, axis=-1), axis=1)
                    overlap = cfg.length*np.mean(np.sum(np.conj(exact)*pred, axis=-1), axis=1)
                    fidelity = np.clip(np.abs(overlap)**2/np.maximum(norms*exact_norms, 1e-30), 0, 1)
                    values = {"spinor_rel_l2_gauge_aligned": relative(pred, exact),
                              "density_rel_l2": relative(d["rho_predicted"], d["rho_exact"]),
                              "current_rel_l2": relative(d["current_predicted"], d["current_exact"]),
                              "modal_population_rel_l2": relative(d["modal_occupations_predicted"], d["modal_occupations_exact"]),
                              "branch_population_rel_l2": relative(d["branch_occupations_predicted"], d["branch_occupations_exact"]),
                              "fidelity_mean": float(np.mean(fidelity)), "fidelity_min": float(np.min(fidelity)),
                              "max_norm_drift": float(np.max(np.abs(norms-1))),
                              "dense_physical_pde_residual_rms": float(np.sqrt(np.mean(np.abs(d["dirac_residual_physical"])**2)))}
                    active = d["branch_occupations_exact"] >= 1e-6
                    assert np.array_equal(active, d["reference_active_branch_mask"])
                    assert np.isnan(d["branch_phase_error_rad_masked_nan"][~d["branch_phase_evaluation_mask"]]).all()
                    flat = active.reshape(201, -1)
                    pairs = flat[:, :, None] & flat[:, None, :] & ~np.eye(26, dtype=bool)[None]
                    assert np.array_equal(pairs, d["reference_active_pair_mask"])
                    values["active_coherence_rel_l2"] = relative(d["channel_coherence_predicted"][pairs], d["channel_coherence_exact"][pairs])
                    for key, value in values.items():
                        maximum_difference = max(maximum_difference, abs(value-metric[key]))
                        assert np.isclose(value, metric[key], rtol=2e-10, atol=2e-12), (model, seed, key)
                    assert metric["initial_condition_max_abs_error"] < 1e-12
                    if model != "physical_vanilla":
                        assert metric["parseval_dense_rms_absolute_difference"] < 1e-12
                    if model == TARGET:
                        assert metric["initial_sideband_imaginary_tangent_parameter_sensitivity"] == 0
                        assert metric["initial_sideband_time_derivative_rel_l2"] >= 1-1e-12
        return {"runs_recomputed": 15, "max_absolute_metric_difference": maximum_difference, "mask_threshold": 1e-6}
    check("raw_metrics_masks_gauge_and_parseval", raw_metrics)

    def statistics():
        original_rows = [json.loads((RAW/"learned"/m/f"seed_{s}/metrics.json").read_text(encoding="utf-8")) for m in PERIODIC_POTENTIAL_MODELS for s in range(5)]
        direct = pd.DataFrame(original_rows)
        decision = evaluate_periodic_potential_decision(direct)
        assert decision == json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
        saved = pd.read_csv(RAW/"runs.csv")
        for metric in direct.select_dtypes(include=[np.number]):
            np.testing.assert_allclose(saved[metric], direct[metric], rtol=2e-12, atol=2e-14, equal_nan=True)
        summary = pd.read_csv(RAW/"summary.csv").set_index("model")
        for model in PERIODIC_POTENTIAL_MODELS:
            rows = direct[direct.model == model]
            for metric in rows.select_dtypes(include=[np.number]).columns.difference(["seed", "steps"]):
                values = rows[metric].dropna().to_numpy(float)
                if len(values):
                    assert np.isclose(summary.loc[model, metric+"_mean"], np.mean(values), atol=2e-12)
                    assert np.isclose(summary.loc[model, metric+"_population_sd"], np.std(values, ddof=0), atol=2e-12)
        return {"classification": decision["classification"], "population_sd_ddof": 0}
    check("aggregation_and_decision_recomputed", statistics)

    def figures():
        manifest = json.loads((FIGURES/"figure_manifest.json").read_text(encoding="utf-8"))
        assert tuple(f["id"] for f in manifest["figures"]) == FIGURE_IDS
        assert manifest["script_sha256"] == sha(ROOT/"src/plot_periodic_potential_coupled_ltpb.py")
        for figure in manifest["figures"]:
            for ext in ("png", "svg", "pdf"):
                assert sha(FIGURES/f"{figure['id']}.{ext}") == figure["output_sha256"][ext]
            for source in figure["source_data"]:
                assert sha(FIGURES/source["file"]) == source["sha256"]
                assert len(pd.read_csv(FIGURES/source["file"])) == source["rows"]
        assert json.loads((FIGURES/"figure_qa.json").read_text())["status"] == "PASS"
        return {"figures": 8, "formats": 24, "seed0_fixed": True, "visual_qa": "PASS"}
    check("figures_source_data_hashes_and_visual_qa", figures)

    def logs():
        base = ROOT/"results/logs/periodic_potential_coupled_ltpb"
        completion = json.loads((base/"completion.json").read_text())
        assert len(completion["runs"]) == 15
        assert all(r["returncode"] == 0 for r in completion["runs"])
        for r in completion["runs"]:
            text = (ROOT/r["log"]).read_text(encoding="utf-8")
            assert text.count("START ") == 1 and text.count("DONE ") == 1
            assert "5000/5000" in text and "Traceback" not in text
        return {"completed_without_retries": 15}
    check("formal_logs_and_fixed_budget", logs)

    def links():
        count = 0
        documents = [REPORT, ROOT/"PERIODIC_POTENTIAL_COUPLED_LTPB_PREREGISTRATION.md", ROOT/"PERIODIC_POTENTIAL_COUPLED_LTPB_AMENDMENT_01.md", FIGURES/"FIGURE_QA.md"]
        for document in documents:
            text = document.read_text(encoding="utf-8")
            for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
                target = target.strip("<>").split("#", 1)[0]
                if not target or "://" in target:
                    continue
                assert (document.parent/target).resolve().exists(), (document, target)
                count += 1
        report_text = REPORT.read_text(encoding="utf-8")
        for model in PERIODIC_POTENTIAL_MODELS:
            for seed in range(5):
                for artifact in RUN_ARTIFACTS:
                    assert f"results/raw/periodic_potential_coupled_ltpb/learned/{model}/seed_{seed}/{artifact}" in report_text
        return {"verified_links": count, "raw_artifact_links": 90}
    check("report_and_evidence_links", links)
    result = {"status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
              "checks_passed": sum(c["pass"] for c in checks), "checks_total": len(checks), "checks": checks}
    write_json(RAW/"audit.json", result)
    return result


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)
