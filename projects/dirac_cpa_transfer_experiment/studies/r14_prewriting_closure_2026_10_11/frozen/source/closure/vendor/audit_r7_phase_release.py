from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pypdf import PdfReader

from .freeze_r7_phase_release import ROOT, RAW, DOCS, sha256, verify_freeze, verify_history, write_json
from .r7_config import EXPERIMENT, MODELS, TRAINING_MODES
from .r7_decision import decide
from .periodic_potential_physics import energy_basis_numpy
from .train_r7_phase_release import ARTIFACTS
from .aggregate_r7_phase_release import stage_runs


def relative(predicted, exact):
    return float(np.linalg.norm((predicted-exact).ravel())/np.linalg.norm(exact.ravel()))


def independent_metrics(data, cfg):
    raw, exact = data["spectral_state_predicted_raw"], data["spectral_state_exact"]
    overlap = np.vdot(exact.ravel(), raw.ravel())
    phase = float(np.angle(overlap)) if abs(overlap) > 1e-30 else 0
    predicted = raw*np.exp(-1j*phase)
    pnorm = np.sum(np.abs(predicted)**2, axis=(1, 2))
    enorm = np.sum(np.abs(exact)**2, axis=(1, 2))
    fidelity = np.clip(np.abs(np.sum(np.conj(exact)*predicted, axis=(1, 2)))**2/
                       np.maximum(pnorm*enorm, 1e-30), 0, 1)
    population_p, population_e = np.sum(np.abs(predicted)**2, axis=2), np.sum(np.abs(exact)**2, axis=2)
    psi_p, psi_e = data["psi_predicted_gauge_aligned"], data["psi_exact"]
    rho_p, rho_e = np.sum(np.abs(psi_p)**2, axis=2), np.sum(np.abs(psi_e)**2, axis=2)
    current_p = 2*np.real(np.conj(psi_p[:, :, 0])*psi_p[:, :, 1])
    current_e = 2*np.real(np.conj(psi_e[:, :, 0])*psi_e[:, :, 1])
    basis_p, basis_m = energy_basis_numpy(data["wave_numbers"], cfg["mass"])
    cp = np.stack((np.sum(predicted*basis_p, axis=2), np.sum(predicted*basis_m, axis=2)), axis=2)
    ce = np.stack((np.sum(exact*basis_p, axis=2), np.sum(exact*basis_m, axis=2)), axis=2)
    active = np.abs(ce)**2 >= 1e-6
    defined = np.abs(cp)**2 >= 1e-12
    valid = active & defined
    af = active.reshape(len(active), -1)
    mask = af[:, :, None]&af[:, None, :]
    mask &= ~np.eye(af.shape[1], dtype=bool)[None]
    cpf, cef = cp.reshape(len(cp), -1), ce.reshape(len(ce), -1)
    coh_p, coh_e = cpf[:, :, None]*cpf[:, None, :].conj(), cef[:, :, None]*cef[:, None, :].conj()
    coherence_error = relative(coh_p[mask], coh_e[mask])
    masks_ok = (np.array_equal(active, data["reference_active_branch_mask"])
                and np.array_equal(valid, data["branch_phase_evaluation_mask"])
                and np.array_equal(mask, data["reference_active_pair_mask"])
                and np.isnan(data["branch_phase_error_rad_masked_nan"][~valid]).all())
    result = dict(spinor_rel_l2_gauge_aligned=relative(predicted, exact),
                  spinor_rel_l2_raw=relative(raw, exact), current_rel_l2=relative(current_p, current_e),
                  density_rel_l2=relative(rho_p, rho_e), fidelity_mean=float(fidelity.mean()),
                  fidelity_min=float(fidelity.min()), max_norm_drift=float(np.max(np.abs(pnorm-1))),
                  modal_population_rel_l2=relative(population_p, population_e),
                  branch_population_rel_l2=relative(np.abs(cp)**2, np.abs(ce)**2),
                  active_coherence_rel_l2=coherence_error,
                  active_branch_phase_coverage=float(valid.sum()/active.sum()))
    return result, bool(masks_ok)


def link_check(path):
    text = path.read_text(encoding="utf-8")
    targets = re.findall(r"\]\((?:<([^>]+)>|([^\s)]+))\)", text)
    missing, checked = [], 0
    for angled, plain in targets:
        target = angled or plain
        if target.startswith(("https://", "http://", "#")):
            continue
        target = target.split("#")[0]
        candidate = Path(target)
        if not candidate.is_absolute(): candidate = path.parent/candidate
        checked += 1
        if not candidate.exists(): missing.append(target)
    return dict(pass_check=not missing, checked_links=checked, missing=missing)


def figure_qa():
    folder = ROOT/"results/figures"/EXPERIMENT
    manifest = json.loads((folder/"figure_manifest.json").read_text(encoding="utf-8"))
    renders = json.loads((folder/"qa/pdf_render_manifest.json").read_text(encoding="utf-8"))
    render_lookup = {entry["figure_id"]: entry for entry in renders["figures"]}
    review = (folder/"visual_review.md").read_text(encoding="utf-8")
    rows = []
    for figure in manifest["figures"]:
        checks = []
        for entry in figure["source_data"]+figure["exports"]:
            path = ROOT/entry["path"]
            checks.append(path.is_file() and sha256(path) == entry["sha256"])
        png = folder/(figure["figure_id"]+".png")
        with Image.open(png) as image:
            dimensions = image.size
            checks.append(min(dimensions) >= 200)
        svg = (folder/(figure["figure_id"]+".svg")).read_text(encoding="utf-8")
        svg_texts = svg.count("<text")
        checks.append(svg_texts > 0)
        pdf = PdfReader(folder/(figure["figure_id"]+".pdf"))
        pdf_text = pdf.pages[0].extract_text()
        checks.append(len(pdf.pages) == 1 and len(pdf_text) > 30)
        render = render_lookup[figure["figure_id"]]
        checks += [render["pdf_sha256"] == sha256(folder/(figure["figure_id"]+".pdf")),
                   render["rendered_sha256"] == sha256(ROOT/render["rendered_path"]),
                   f"| {figure['figure_id']} | PASS |" in review]
        rows.append(dict(figure=figure["figure_id"], pass_check=all(checks), png_dimensions=dimensions,
                         svg_editable_text_count=svg_texts, pdf_extracted_characters=len(pdf_text),
                         seed_scope=figure["seed_scope"], source_count=len(figure["source_data"])))
    notes = ["# R7 Figure QA", "", "统计：全部种子；均值±population SD，ddof=0；固定动态seed0。无显著性推断。",
             "", "程序检查源表与导出哈希、PNG尺寸、SVG可编辑文本和PDF可提取文字；核对最终PDF与渲染证据哈希。全部最终PDF已实际目视检查，详见 [目视记录](visual_review.md) 和 [渲染记录](qa/pdf_render_manifest.json)；不能把文件完整性冒充目视通过。", "",
             "| Figure | Checks | PNG | SVG text | Source tables | Seed scope |", "| --- | --- | --- | ---: | ---: | --- |"]
    notes += [f"| {r['figure']} | {'PASS' if r['pass_check'] else 'FAIL'} | {r['png_dimensions']} | {r['svg_editable_text_count']} | {r['source_count']} | {r['seed_scope']} |" for r in rows]
    (folder/"FIGURE_QA.md").write_text("\n".join(notes)+"\n", encoding="utf-8")
    return dict(pass_check=all(r["pass_check"] for r in rows), figures=rows)


def audit():
    freeze_digest = verify_freeze()
    history = verify_history()
    references = json.loads((RAW/"reference_validation.json").read_text(encoding="utf-8"))
    checks = {"freeze_and_history": history["pass"], "references": references["status"] == "PASS"}
    smoke = json.loads((RAW/"smoke_validation.json").read_text(encoding="utf-8"))
    theory = json.loads((RAW/"theory_validation.json").read_text(encoding="utf-8"))
    smoke_paths = list((RAW/"smoke").glob("*/*/seed_*/metrics.json"))
    checks["theory_and_smoke"] = (theory["status"] == "PASS" and smoke["status"] == "PASS" and len(smoke_paths) == 6
        and all(all((p.parent/name).is_file() for name in ARTIFACTS) and json.loads(p.read_text(encoding="utf-8"))["steps"] == 20 for p in smoke_paths))
    reporting = json.loads((RAW/"reporting_code_sha256.json").read_text(encoding="utf-8"))
    checks["reporting_code_hashes"] = all(sha256(ROOT/name) == value for name, value in reporting["files"].items())
    test_root = ET.parse(ROOT/"results/logs"/EXPERIMENT/"final_pytest.xml").getroot()
    suites = [test_root] if test_root.tag == "testsuite" else list(test_root.findall("testsuite"))
    checks["complete_pytest"] = (sum(int(s.attrib.get("tests", 0)) for s in suites) >= 160
        and all(sum(int(s.attrib.get(key, 0)) for key in ("failures", "errors", "skipped")) == 0 for s in suites))
    errors, maximum_difference, records = [], 0.0, []
    expected = [("short", m, "ordinary", s) for m in MODELS for s in range(3)]
    expected += [("main", m, mode, s) for m in MODELS for mode in TRAINING_MODES for s in range(5)]
    # Rebuild from lossless per-run JSON, rather than re-rounding CSV floats.
    main_frame = stage_runs("main")
    decision = decide(main_frame)
    saved_decision = json.loads((RAW/"decision.json").read_text(encoding="utf-8"))
    checks["decision_recomputed"] = decision == saved_decision
    if decision["long_training_triggered_by_main"] and references["long_reference_pass"]:
        expected += [("long", m, decision["long_training_mode_selected"], s) for m in MODELS for s in range(5)]
    stage_seed_hashes, stage_seed_weights = {}, {}
    for stage, model, mode, seed in expected:
        directory = RAW/stage/model/mode/f"seed_{seed}"
        missing = [name for name in ARTIFACTS if not (directory/name).is_file()]
        if missing:
            errors.append(f"{directory}: {missing}")
            continue
        metrics = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
        config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        cfg = config["configuration"]
        with np.load(directory/"dense_grid.npz") as dense:
            rebuilt, masks_ok = independent_metrics(dense, cfg)
            fourier = np.exp(1j*np.outer(dense["x"], dense["wave_numbers"]))/np.sqrt(cfg["length"])
            reconstructed = np.einsum("xk,tks->txs", fourier, dense["spectral_state_predicted_raw"])
            fourier_error = float(np.max(np.abs(reconstructed-dense["psi_predicted_raw"])))
            estimate_violation = float(np.max(dense["raw_spinor_error_norm_fine"]-dense["residual_error_estimate"]))
            raw_ic_error = float(np.max(np.abs(dense["spectral_state_predicted_raw"][0]-dense["spectral_state_exact"][0])))
        differences = {key: abs(value-float(metrics[key])) for key, value in rebuilt.items()}
        maximum_difference = max(maximum_difference, max(differences.values()))
        if max(differences.values()) > 2e-11 or not masks_ok or fourier_error > 2e-12 or raw_ic_error > 2e-12 or estimate_violation > 2e-6:
            errors.append(f"metric or mask mismatch: {stage}/{model}/{mode}/{seed}")
        trace = torch.load(directory/"recovery_state.pt", weights_only=False)["sampler_state"]["time_trace"]
        import hashlib
        trace_hash = hashlib.sha256(trace.numpy().tobytes()).hexdigest()
        stage_seed_hashes.setdefault((stage, seed), set()).add(trace_hash)
        stage_seed_weights.setdefault((stage, seed), set()).add(config["initial_trainable_sha256"])
        history_csv = pd.read_csv(directory/"loss_history.csv")
        states = torch.load(directory/"model_state.pt", weights_only=False)
        actual_parameter_count = sum(tensor.numel() for tensor in states["state_dict"].values())
        allowed_checkpoint_keys = {"model", "training_mode", "seed", "steps", "configuration", "state_dict",
                                   "initial_trainable_sha256", "time_stream_sha256"}
        expected_steps = 1000 if stage == "short" else 5000
        if (metrics["steps"] != expected_steps or history_csv.step.iloc[-1] != expected_steps
            or states["steps"] != expected_steps or trace_hash != metrics["time_stream_sha256"]
            or len(trace) != expected_steps or metrics["parameter_count"] != 12932
            or actual_parameter_count != 12932 or set(states) != allowed_checkpoint_keys
            or config["freeze_manifest_sha256"] != freeze_digest):
            errors.append(f"budget, trace, parameter or freeze mismatch: {stage}/{model}/{mode}/{seed}")
        if stage != "short":
            for step in (1000, 2500, 5000):
                checkpoint = directory/"checkpoints"/f"step_{step}"
                if not all((checkpoint/p).exists() for p in ("model_state.pt", "training_state.pt")):
                    errors.append(f"missing fixed checkpoint {checkpoint}")
                if step < 5000 and not all((checkpoint/p).exists() for p in ("metrics.json", "dense_grid.npz", "time_series.csv")):
                    errors.append(f"missing checkpoint evaluation {checkpoint}")
        events = [json.loads(line) for line in (directory/"execution_events.jsonl").read_text(encoding="utf-8").splitlines()]
        records.append(dict(stage=stage, model=model, mode=mode, seed=seed, max_metric_difference=max(differences.values()),
                            masks_pass=masks_ok, physical_reconstruction_max_abs=fourier_error,
                            hard_initial_condition_max_abs=raw_ic_error, numerical_residual_estimate_violation=estimate_violation,
                            execution_event_types=[event["event"] for event in events], actual_parameter_count=actual_parameter_count))
    checks["all_runs_and_metrics"] = not errors and len(records) == len(expected)
    checks["paired_time_streams"] = all(len(v) == 1 for v in stage_seed_hashes.values())
    checks["paired_initial_weights"] = all(len(v) == 1 for v in stage_seed_weights.values())
    figures = figure_qa()
    checks["figure_artifacts"] = figures["pass_check"]
    report_path = ROOT/"R7_PHASE_RELEASE_EXPERIMENT_REPORT.md"
    # Materialize this newly created artifact before checking its report link.
    # Final audit results below replace only this experiment's own audit file.
    if not (RAW/"audit.json").exists():
        write_json(RAW/"audit.json", {"status": "RUNNING", "scientific_success_is_separate_from_engineering_audit": True})
    links = link_check(report_path)
    checks["report_links"] = links["pass_check"]
    checks["report_categorized_copy_identical"] = sha256(report_path) == sha256(DOCS/"report.md")
    prereg_links = link_check(DOCS/"preregistration.md")
    checks["preregistration_links"] = prereg_links["pass_check"]
    result = dict(status="PASS" if all(checks.values()) else "FAIL", checks=checks, errors=errors,
                  freeze_manifest_sha256=freeze_digest, history=history, expected_runs=len(expected),
                  actual_checked_runs=len(records), maximum_independent_metric_difference=maximum_difference,
                  report_sha256=sha256(report_path),
                  per_run=records, figure_qa=figures, report_links=links, preregistration_links=prereg_links,
                  scientific_success_is_separate_from_engineering_audit=True)
    write_json(RAW/"audit.json", result)
    print(json.dumps(dict(status=result["status"], checked_runs=len(records), max_metric_difference=maximum_difference,
                         changed_history=history["changed_or_missing"], errors=errors, checks=checks), ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(0 if audit()["status"] == "PASS" else 1)
