"""Read-only synthesis of the six completed experiments; writes new report outputs only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reports/dirac_ltpb_comprehensive"
TEMPLATE = ROOT / "reports/dirac_ltpb_comprehensive/template.md"
REPORT = ROOT / "DIRAC_LTPB_COMPREHENSIVE_EXPERIMENT_REPORT.md"
ROUNDS = [
    ("R1", "单相位迁移与守恒量诊断", "main", 20),
    ("R1S", "结构损失权重敏感性", "sensitivity", 2),
    ("R2", "可学习双能支相位 pilot", "learned_two_phase_pilot", 6),
    ("R3", "能支对齐机制消融", "ltpb_mechanism_ablation", 15),
    ("R4", "双波数 Multi-LTPB", "multimode_ltpb", 26),
    ("R5", "高斯波包共享谱 LTPB", "wavepacket_shared_spectral_ltpb", 26),
    ("R6", "弱周期外势耦合谱迁移", "periodic_potential_coupled_ltpb", 15),
]
LABELS = {
    "vanilla": "Vanilla", "sp_cpa": "SP-CPA", "sp_cpa_conservative": "SP-CPA-Conservative",
    "two_branch": "Two-Branch oracle", "two_branch_oracle": "Two-Branch oracle",
    "ltpb_cpa": "LTPB-CPA", "branch_complex": "Branch-Complex", "component_polar": "Component-Polar",
    "rotated_ltpb_15": "Rotated-LTPB 15°", "rotated_ltpb_30": "Rotated-LTPB 30°",
    "multi_branch_complex": "MultiBranchComplex", "multi_ltpb": "Multi-LTPB",
    "physical_vanilla": "Physical-Vanilla", "physical_sp_cpa": "Physical-SP-CPA",
    "physical_component_polar": "Physical-ComponentPolar", "shared_spectral_complex": "Shared-Spectral-Complex",
    "shared_spectral_ltpb": "Shared-Spectral-LTPB", "spectral_oracle": "Spectral oracle（未训练）",
    "four_carrier_oracle": "Four-carrier oracle（未训练）",
    "oracle": "Four-carrier oracle（未训练）",
    "coupled_shared_spectral_complex": "Coupled-Shared-Spectral-Complex",
    "coupled_shared_spectral_ltpb": "Coupled-Shared-Spectral-LTPB",
}
METRICS = {
    "spinor": ("spinor_rel_l2_gauge_aligned", "spinor_rel_l2"),
    "final_spinor": ("final_spinor_rel_l2_gauge_aligned", "final_spinor_rel_l2"),
    "density": ("density_rel_l2", "rho_rel_l2"),
    "current": ("current_rel_l2",), "fidelity": ("fidelity_mean",),
    "fidelity_min": ("fidelity_min",), "norm_drift": ("max_norm_drift",),
    "component_phase": ("relative_phase_rmse_rad", "spinor_component_relative_phase_rmse_rad"),
    "pde_loss": ("final_loss_pde", "final_pde_loss"),
    "pde_rms": ("dense_physical_pde_residual_rms", "dense_pde_residual_rms"),
    "seconds": ("training_seconds",), "parameters": ("parameter_count",),
    "peak_rss_mib": ("peak_rss_mb",), "rss_increment_mib": ("peak_rss_increment_mb",),
}
AUDITS = [
    "results/reproducibility_audit.json", "results/learned_two_phase_pilot_audit.json",
    "results/ltpb_mechanism_ablation_audit.json", "results/multimode_ltpb_audit.json",
    "results/wavepacket_shared_spectral_ltpb_audit.json", "results/raw/periodic_potential_coupled_ltpb/audit.json",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(name: str, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def link(label: str, path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return f"[{label}](<{path.resolve().as_posix()}>)"


def fmt(value, fidelity=False) -> str:
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "是" if value else "否"
    if fidelity:
        return f"{value:.6f}"
    return f"{value:.6g}"


def table(headers: list[str], rows) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |" for row in rows])


def mean_sd(values, fidelity=False):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return "—"
    if len(values) == 1:
        return fmt(values[0], fidelity) + "（n=1）"
    return f"{fmt(statistics.mean(values), fidelity)} ± {fmt(statistics.pstdev(values))}"


def original_paths() -> list[Path]:
    paths = set()
    for _, _, directory, _ in ROUNDS:
        paths.update(p for p in (ROOT / "results/raw" / directory).rglob("*") if p.is_file())
    paths.update(p for p in (ROOT / "results/figures").rglob("*") if p.is_file())
    paths.update((ROOT / "results").glob("*.json"))
    paths.update((ROOT / "results").glob("*.csv"))
    paths.update((ROOT / "results/raw").glob("*.json"))
    paths.update(p for p in (ROOT / "results/raw/figure_source_data").rglob("*") if p.is_file())
    paths.update(p for p in ROOT.glob("*.md") if p != REPORT)
    paths.update(p for p in (ROOT / "src").glob("*.py") if p != Path(__file__).resolve())
    paths.update((ROOT / "tests").glob("*.py"))
    proposal = ROOT.parent / "CPA_PINN推广至Dirac方程的局限与旋量双相位方法_开题报告.md"
    paths.add(proposal)
    return sorted(paths, key=str)


def inventory():
    records, native, artifacts = [], [], []
    for code, title, directory, expected in ROUNDS:
        raw = ROOT / "results/raw" / directory
        if code in {"R4", "R5", "R6"}:
            paths = sorted((raw / "learned").rglob("metrics.json"))
            if code in {"R4", "R5"}:
                paths += sorted((raw / "oracle").rglob("metrics.json"))
        else:
            paths = sorted(raw.rglob("metrics.json"))
        assert len(paths) == expected, (code, len(paths), expected)
        for path in paths:
            data = read_json(path)
            config = read_json(path.parent / "config.json")
            trained = (path.parent / "loss_history.csv").exists()
            oracle = "oracle" in data["model"] or data["model"] == "two_branch"
            category = "trained_oracle" if trained and oracle else "untrained_oracle" if oracle else "learned"
            assert trained or oracle, path
            record = {"round": code, "round_title": title, "model": data["model"], "case": data.get("case", ""),
                      "seed": data.get("seed", 0), "lambda_struct": data.get("lambda_struct") if code == "R1S" else None,
                      "category": category, "actual_optimizer_steps": data["steps"] if trained else 0,
                      "native_steps_field": data.get("steps"),
                      "spinor_convention": "one_global_constant_phase" if "spinor_rel_l2_gauge_aligned" in data else "raw",
                      "metrics_path": path.relative_to(ROOT).as_posix()}
            for key, options in METRICS.items():
                record[key] = next((data[o] for o in options if o in data), None)
            for key in ("spinor", "density", "current", "fidelity"):
                assert record[key] is not None and math.isfinite(record[key]), (path, key)
            artifact = {k: record[k] for k in ("round", "model", "case", "seed", "category")}
            required = ["config.json", "metrics.json", "time_series.csv", "dense_grid.npz"]
            if trained:
                required += ["loss_history.csv", "model_state.pt"]
                cfg = config.get("experiment_config", config.get("configuration", {}))
                assert record["actual_optimizer_steps"] == cfg["adam_steps"] == 5000
                assert cfg["hidden_layers"] == 4 and cfg["dtype"] == "float64" and cfg["learning_rate"] == 0.001
                with (path.parent / "loss_history.csv").open(encoding="utf-8-sig", newline="") as stream:
                    history = list(csv.DictReader(stream))
                assert int(history[-1]["step"]) == 5000, path
            for name in required:
                p = path.parent / name
                assert p.is_file() and p.stat().st_size > 0, p
                artifact[name] = p.relative_to(ROOT).as_posix()
            records.append(record)
            artifacts.append(artifact)
            native.append({"round": code, "metrics_path": record["metrics_path"], "sha256": sha(path), "metrics": data})
    orders = {
        "R1": ["vanilla", "sp_cpa", "sp_cpa_conservative", "two_branch", "two_branch_oracle"],
        "R3": ["ltpb_cpa", "branch_complex", "component_polar", "rotated_ltpb_15", "rotated_ltpb_30"],
        "R4": ["vanilla", "sp_cpa", "component_polar", "multi_branch_complex", "multi_ltpb", "oracle"],
        "R5": ["physical_vanilla", "physical_sp_cpa", "physical_component_polar", "shared_spectral_complex", "shared_spectral_ltpb", "spectral_oracle"],
        "R6": ["physical_vanilla", "coupled_shared_spectral_complex", "coupled_shared_spectral_ltpb"],
    }
    def ordering(row):
        order = orders.get(row["round"], [row["model"]])
        return ([r[0] for r in ROUNDS].index(row["round"]), {"P": 0, "M": 1, "": 2}[row["case"]],
                order.index(row["model"]) if row["model"] in order else 99, row["lambda_struct"] or 0, row["seed"])
    records.sort(key=ordering)
    return records, native, artifacts


def grouping(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["round"], r["case"], r["model"], r["lambda_struct"])].append(r)
    return groups


def primary_table(records):
    rows = []
    keys = ["spinor", "density", "current", "fidelity"]
    titles = ["spinor L2", "density L2", "current L2", "mean fidelity"]
    if any(r["component_phase"] is not None for r in records):
        keys.append("component_phase")
        titles.append("分量相对相位 RMSE / rad")
    for (_, case, model, lam), group in grouping(records).items():
        label = LABELS.get(model, model) + (f" / {case}" if case else "") + (f" / λ={lam:g}" if lam is not None else "")
        rows.append([label, len(group)] + [mean_sd([r[k] for r in group], k == "fidelity") for k in keys])
    return table(["模型 / Case", "n"] + titles, rows)


def native_summary(records, native_map, keys, titles):
    rows = []
    for (_, case, model, _), group in grouping(records).items():
        rows.append([LABELS.get(model, model) + (" / " + case if case else "")] +
                    [mean_sd([native_map[r["metrics_path"]].get(key) for r in group]) for key in keys])
    return table(["模型 / Case"] + titles, rows)


def figure(base: str, caption: str, sources: list[str]) -> str:
    png = (ROOT / (base + ".png")).as_posix()
    return f"![{caption}](<{png}>)\n\n{caption} " + " · ".join(
        [link("可编辑 SVG", base + ".svg"), link("PDF", base + ".pdf")]
        + [link("Source data" + (f" {i+1}" if len(sources) > 1 else ""), p) for i, p in enumerate(sources)])


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    protection_path = OUT / "original_source_sha256.json"
    if protection_path.exists():
        protection = read_json(protection_path)
        assert all(Path(p).is_file() and sha(Path(p)) == h for p, h in protection["files"].items()), "Original evidence changed"
        additions = {p.as_posix(): sha(p) for p in original_paths() if p.as_posix() not in protection["files"]}
        if additions:
            protection["files"].update(additions)
            protection.setdefault("scope_additions", []).append({"captured_utc": datetime.now(timezone.utc).isoformat(), "files": list(additions)})
            save_json("original_source_sha256.json", protection)
    else:
        protection = {"captured_utc": datetime.now(timezone.utc).isoformat(),
                      "scope": "Existing formal raw results, figures, root reports, result tables/audits, scientific Python and tests",
                      "files": {p.as_posix(): sha(p) for p in original_paths()}}
        save_json("original_source_sha256.json", protection)
    records, native, artifacts = inventory()
    native_map = {r["metrics_path"]: r["metrics"] for r in native}
    by_round = {code: [r for r in records if r["round"] == code] for code, *_ in ROUNDS}
    assert len(records) == 110
    assert sum(r["category"] == "learned" for r in records) == 106
    assert sum(r["category"] == "trained_oracle" for r in records) == 2
    assert sum(r["category"] == "untrained_oracle" for r in records) == 2
    save_csv("all_runs.csv", records)
    save_csv("run_artifact_index.csv", artifacts)
    save_json("native_metrics.json", native)
    aggregates = []
    for (code, case, model, lam), group in grouping(records).items():
        row = {"round": code, "case": case, "model": model, "lambda_struct": lam, "n": len(group),
               "category": group[0]["category"], "spinor_convention": group[0]["spinor_convention"]}
        for key in METRICS:
            values = [r[key] for r in group if r[key] is not None and math.isfinite(r[key])]
            row[key + "_mean"] = statistics.mean(values) if values else None
            row[key + "_population_sd"] = statistics.pstdev(values) if values else None
        aggregates.append(row)
    save_csv("aggregate_by_round.csv", aggregates)
    tokens = {"ROOT": ROOT.as_posix(), "PARENT": ROOT.parent.as_posix(), "OUT": OUT.as_posix(),
              "PROTECTED_COUNT": str(len(protection["files"]))}
    for code in by_round:
        tokens[code + "_PRIMARY"] = primary_table(by_round[code])

    # All individual runs, including the two sensitivity runs and oracle records.
    run_sections = []
    for code, title, _, _ in ROUNDS:
        rows = []
        for r in by_round[code]:
            name = LABELS.get(r["model"], r["model"]) + (f" / {r['case']}" if r["case"] else "")
            if r["lambda_struct"] is not None:
                name += f" / λ={r['lambda_struct']:g}"
            rows.append([link(name, r["metrics_path"]), r["seed"]] + [fmt(r[k], k == "fidelity") for k in
                         ("spinor", "final_spinor", "density", "current", "fidelity", "fidelity_min")])
        run_sections.append(f"### {code}：{title}\n\n" + table(
            ["模型（链接原始 metrics）", "seed", "spinor L2", "final L2", "density L2", "current L2", "mean F", "min F"], rows))
    tokens["ALL_RUNS"] = "\n\n".join(run_sections)

    resource_rows = []
    for a in aggregates:
        label = LABELS.get(a["model"], a["model"]) + (f" / {a['case']}" if a["case"] else "")
        if a["lambda_struct"] is not None:
            label += f" / λ={a['lambda_struct']:g}"
        def agg(key):
            if a[key + "_mean"] is None:
                return "—"
            return f"{fmt(a[key+'_mean'])} ± {fmt(a[key+'_population_sd'])}" if a["n"] > 1 else fmt(a[key+"_mean"])
        resource_rows.append([a["round"], label, fmt(a["parameters_mean"]), agg("seconds"), agg("pde_loss"),
                              agg("norm_drift"), agg("peak_rss_mib"), agg("rss_increment_mib")])
    tokens["RESOURCES"] = table(["轮次", "模型 / Case", "参数", "训练秒数", "最终 PDE loss", "max norm drift", "训练 peak RSS / MiB", "RSS 增量 / MiB"], resource_rows)

    sens = by_round["R1S"] + [r for r in by_round["R1"] if r["model"] == "sp_cpa_conservative" and r["case"] == "M" and r["seed"] == 0]
    sens.sort(key=lambda r: native_map[r["metrics_path"]]["lambda_struct"])
    tokens["R1_SENSITIVITY"] = table(["λ", "数据来源", "spinor", "current", "mean F", "连续性归一化 RMS", "max norm drift", "max energy drift"],
        [[fmt(native_map[r["metrics_path"]]["lambda_struct"]), "复用主实验" if r["round"] == "R1" else "独立训练", fmt(r["spinor"]), fmt(r["current"]), fmt(r["fidelity"], True),
          fmt(native_map[r["metrics_path"]]["continuity_normalized_rms"]), fmt(r["norm_drift"]), fmt(native_map[r["metrics_path"]]["max_abs_energy_drift"])] for r in sens])

    tokens["R2_DIAGNOSTICS"] = native_summary(by_round["R2"], native_map,
        ["phase_plus_slope", "phase_minus_slope", "phase_plus_rmse_rad", "phase_minus_rmse_rad", "amplitude_plus_drift_rel_rms", "amplitude_minus_drift_rel_rms", "branch_coefficient_rel_l2", "branch_occupation_rmse"],
        ["S+ 斜率", "S− 斜率¹", "S+ RMSE / rad", "S− RMSE / rad¹", "a+ 相对 RMS 漂移", "a− 相对 RMS 漂移¹", "分支系数 L2", "分支占据 RMSE"])
    # Replace meaningless absent-branch phase statistics for Case P only, without changing native source data.
    p_rows = tokens["R2_DIAGNOSTICS"].splitlines()
    for i, line in enumerate(p_rows):
        if "LTPB-CPA / P" in line:
            cells = line.split(" | ")
            for column in (2, 4, 6):
                cells[column] = "不适用（初始未占据）"
            p_rows[i] = " | ".join(cells)
    tokens["R2_DIAGNOSTICS"] = "\n".join(p_rows)

    tokens["R3_COMPLEXITY"] = native_summary(by_round["R3"], native_map,
        ["internal_amplitude_drift_rel_rms", "internal_exact_amplitude_drift_rel_rms",
         "internal_phase_velocity_fluctuation_weighted_rms", "internal_exact_phase_velocity_fluctuation_weighted_rms",
         "internal_phase_curvature_weighted_rms", "internal_exact_phase_curvature_weighted_rms"],
        ["振幅漂移：预测", "振幅漂移：精确", "相位速度波动：预测", "相位速度波动：精确", "加权曲率：预测", "加权曲率：精确"])

    r4target = [r for r in by_round["R4"] if r["model"] == "multi_ltpb"]
    tokens["R4_SLOPES"] = table(["seed", "k=2,+ 斜率", "k=2,− 斜率", "k=5,+ 斜率", "k=5,− 斜率", "四通道最大斜率相对误差"],
        [[r["seed"]] + [fmt(native_map[r["metrics_path"]][k+"_phase_slope"]) for k in ["k2_plus", "k2_minus", "k5_plus", "k5_minus"]] +
         [fmt(max(native_map[r["metrics_path"]][k+"_phase_slope_relative_error"] for k in ["k2_plus", "k2_minus", "k5_plus", "k5_minus"]))] for r in r4target])
    tokens["R4_MODES"] = native_summary(by_round["R4"], native_map,
        ["mode_coefficient_rel_l2_gauge_aligned", "mode_occupation_rmse_rel", "relative_phase_pairs_circular_rmse_rad", "phase_velocity_fluctuation_weighted_rms", "phase_curvature_weighted_rms"],
        ["四模态系数 L2", "模态占据相对 RMSE", "四对相对相位 RMSE / rad", "相位速度波动", "加权相位曲率"])
    r5target = [r for r in by_round["R5"] if r["model"] == "shared_spectral_ltpb"]
    tokens["R5_PHASES"] = table(["seed", "训练模态斜率加权相对误差", "中点斜率加权相对误差", "中点振幅相对 RMS 漂移", "中点谱残差 RMS", "中点最小有效时间比例"],
        [[r["seed"]] + [fmt(native_map[r["metrics_path"]][k]) for k in ["training_phase_slope_weighted_relative_error", "heldout_phase_slope_weighted_relative_error", "heldout_amplitude_drift_weighted_rel_rms", "heldout_spectral_pde_residual_rms", "heldout_phase_min_valid_fraction"]] for r in r5target])
    tokens["R5_BRANCHES"] = table(["seed", "v+", "v−", "中心 RMSE +", "中心 RMSE −", "宽度 L2 +", "宽度 L2 −"],
        [[r["seed"]] + [fmt(native_map[r["metrics_path"]][k]) for k in ["branch_plus_velocity", "branch_minus_velocity", "branch_plus_center_rmse", "branch_minus_center_rmse", "branch_plus_width_rel_l2", "branch_minus_width_rel_l2"]] for r in r5target])
    tokens["R5_MODES"] = native_summary(by_round["R5"], native_map,
        ["spectral_coefficient_rel_l2_gauge_aligned", "spectral_occupation_rel_l2", "dense_physical_pde_residual_rms", "max_norm_drift"],
        ["谱系数 L2", "谱占据 L2", "dense physical PDE RMS", "max norm drift"])

    with (ROOT / "results/raw/periodic_potential_coupled_ltpb/calibration/calibration_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        calibration = list(csv.DictReader(stream))
    tokens["R6_CALIBRATION"] = table(["T", "V₀", "中心外最大占据（J=6）", "J=6 外层最大总占据", "J=4/6 全轨迹复旋量相对差", "原判据"],
        [[fmt(float(r["t_max"])), fmt(float(r["potential_strength"])), fmt(float(r["large_ladder_observables__max_total_occupation_outside_center"])),
          fmt(float(r["large_ladder_observables__max_outer_layer_total_occupation"])), fmt(float(r["truncation_comparison__padded_full_trajectory_spinor_relative_l2"])), "未通过"] for r in calibration])
    tokens["R6_AUX"] = native_summary(by_round["R6"], native_map,
        ["modal_population_rel_l2", "branch_population_rel_l2", "max_norm_drift", "active_coherence_rel_l2", "active_branch_phase_circular_rmse", "active_branch_phase_coverage"],
        ["模态占据 L2", "分支占据 L2", "max norm drift", "活跃相干量 L2", "有效相位 RMSE / rad", "预测相位覆盖率"])
    r6decision = read_json(ROOT / "results/raw/periodic_potential_coupled_ltpb/decision.json")
    tokens["R6_PAIRS"] = table(["seed", "spinor：LTPB−Complex", "spinor：LTPB/Complex", "current：LTPB−Complex", "current：LTPB/Complex", "两项均胜出"],
        [[r["seed"], fmt(r["spinor_rel_l2_gauge_aligned_ltpb_minus_complex"]), fmt(r["spinor_rel_l2_gauge_aligned_ltpb_over_complex"]), fmt(r["current_rel_l2_ltpb_minus_complex"]), fmt(r["current_rel_l2_ltpb_over_complex"]), fmt(r["joint_win"])] for r in r6decision["paired_seed_comparisons"]])
    tokens["R6_TANGENT"] = native_summary([r for r in by_round["R6"] if r["model"] != "physical_vanilla"], native_map,
        ["initial_sideband_time_derivative_rel_l2", "initial_sideband_imaginary_tangent_parameter_sensitivity"],
        ["初始侧带导数 relative L2", "虚部切向参数敏感度"])

    figure_specs = {
        "FIG_R1": ("results/figures/01_main_metric_comparison", "图 1　原始单动量主实验：SP-CPA 的优势从纯正能支到混合能支明显退化。", ["results/summary.csv", "results/summary_aggregated.csv"]),
        "FIG_R1_CONS": ("results/figures/07_lambda_struct_sensitivity", "图 2　结构损失权重敏感性：守恒量改善不等于旋量和概率流恢复。", ["results/raw/figure_source_data/lambda_sensitivity_metrics.csv"]),
        "FIG_R2": ("results/figures/learned_two_phase_pilot/02_learned_branch_phases", "图 3　初步 LTPB 学到的独立能支相位；Case P 的空负能支相位不作为成功证据。", ["results/figures/learned_two_phase_pilot/source_data/02_learned_branch_phases.csv", "results/figures/learned_two_phase_pilot/source_data/02_learned_branch_phase_slopes.csv"]),
        "FIG_R3": ("results/figures/ltpb_mechanism_ablation/05_rotation_degradation", "图 4　基底旋转显著降低精度，但 15°→30° 的增幅未达到预注册 1.25 倍逐级门槛。", ["results/figures/ltpb_mechanism_ablation/source_data/05_rotation_degradation.csv"]),
        "FIG_R4": ("results/figures/multimode_ltpb/primary_error_comparison", "图 5　双波数实验的 learned 模型比较；精确四载频 oracle 不参加排名。", ["results/figures/multimode_ltpb/primary_error_comparison_source_data.csv"]),
        "FIG_R4_PHASE": ("results/figures/multimode_ltpb/learned_four_phases", "图 6　双波数四条物理投影相位，诊断载频是否真正恢复。", ["results/figures/multimode_ltpb/learned_four_phases_source_data.csv"]),
        "FIG_R5": ("results/figures/wavepacket_shared_spectral_ltpb/wavepacket_density_spacetime", "图 7　共享谱高斯波包的参考密度、预测及误差；代表图使用固定 seed 0。", ["results/figures/wavepacket_shared_spectral_ltpb/source_data/wavepacket_density_spacetime.csv"]),
        "FIG_R5_PHASE": ("results/figures/wavepacket_shared_spectral_ltpb/phase_slope_train_midpoint", "图 8　训练格点与未训练中点的相位斜率；支持已知谱区间内的插值，而非无限连续谱或未知谱发现。", ["results/figures/wavepacket_shared_spectral_ltpb/source_data/phase_slope_train_midpoint.csv"]),
        "FIG_R5_BRANCH": ("results/figures/wavepacket_shared_spectral_ltpb/branch_center_width", "图 9　正负能波包中心和宽度：两支向相反方向传播。", ["results/figures/wavepacket_shared_spectral_ltpb/source_data/branch_center_width.csv"]),
        "FIG_R6_CAL": ("results/figures/periodic_potential_coupled_ltpb/calibration_modal_transfer", "图 10　外势问题的纯参考解校准，包含原门槛失败与获授权的截断修订。", ["results/figures/periodic_potential_coupled_ltpb/source_data/calibration_modal_transfer_gates.csv", "results/figures/periodic_potential_coupled_ltpb/source_data/calibration_modal_transfer_approved_revision.csv"]),
        "FIG_R6": ("results/figures/periodic_potential_coupled_ltpb/all_seed_primary_metrics", "图 11　外势迁移全部五种子的主要误差；相对较好不等于达到绝对成功条件。", ["results/figures/periodic_potential_coupled_ltpb/source_data/all_seed_primary_metrics.csv"]),
        "FIG_R6_SIDEBAND": ("results/figures/periodic_potential_coupled_ltpb/center_sideband_trajectories_seed0", "图 12　固定 seed 0 的中心与一阶侧带占据：当前 LTPB 未恢复动态模态产生。", ["results/figures/periodic_potential_coupled_ltpb/source_data/center_sideband_trajectories_seed0.csv"]),
        "FIG_R6_DIAG": ("results/figures/periodic_potential_coupled_ltpb/active_channel_diagnostics_seed0", "图 13　有效相位区域和初始切向诊断：空通道的相位退化有局部数学证据，整体失败仍受共同优化失败混杂。", ["results/figures/periodic_potential_coupled_ltpb/source_data/active_channel_diagnostics_seed0_channels.csv", "results/figures/periodic_potential_coupled_ltpb/source_data/active_channel_diagnostics_seed0_tangent.csv"]),
    }
    for token, (base, caption, sources) in figure_specs.items():
        tokens[token] = figure(base, caption, sources)
    figure_index = []
    for directory in [ROOT / "results/figures"] + [ROOT / "results/figures" / d for _, _, d, _ in ROUNDS if d not in {"main", "sensitivity"}]:
        if directory.is_dir():
            for png in sorted(directory.glob("*.png")):
                figure_index.append([link(png.stem, png), link("SVG", png.with_suffix(".svg")), link("PDF", png.with_suffix(".pdf")), link("QA", directory / "FIGURE_QA.md")])
    tokens["FIGURE_INDEX"] = table(["完整图册（PNG）", "可编辑格式", "打印格式", "原图 QA"], figure_index)

    audit_rows = []
    for i, p in enumerate(AUDITS, 1):
        d = read_json(ROOT / p)
        assert d["status"] == "PASS", p
        count = f"{d['checks_passed']}/{d['checks_total']}" if "checks_passed" in d else "检查项见原 JSON"
        audit_rows.append([f"R{i}", link("原始审计", p), d["status"], count])
    tokens["AUDITS"] = table(["轮次", "证据", "工程审计", "计数"], audit_rows)
    manifests = ["results/multimode_ltpb_preregistration_sha256.json", "results/wavepacket_shared_spectral_ltpb_preregistration_sha256.json",
                 "results/raw/periodic_potential_coupled_ltpb/frozen_protocol_and_code.json"]
    tokens["FREEZES"] = table(["历史冻结证据", "该清单文件的当前 SHA-256（清单内另含各源文件摘要）"], [[link(Path(p).name, p), sha(ROOT / p)] for p in manifests])
    template = TEMPLATE.read_text(encoding="utf-8")
    for token, value in tokens.items():
        template = template.replace("@@" + token + "@@", value)
    assert not re.findall(r"@@[A-Z0-9_]+@@", template), "Unresolved template fields"
    REPORT.write_text(template, encoding="utf-8")

    # Reconcile new aggregate numbers to existing preregistered decision files.
    reconciled = []
    for code, model, decision_path in [("R4", "multi_ltpb", "results/multimode_ltpb_decision.json"),
                                      ("R5", "shared_spectral_ltpb", "results/wavepacket_shared_spectral_ltpb_decision.json")]:
        target = [r for r in by_round[code] if r["model"] == model]
        decision = read_json(ROOT / decision_path)
        for metric, field in [("spinor", "spinor_mean_le_0p05"), ("current", "current_mean_le_0p10"), ("fidelity", "fidelity_mean_ge_0p99")]:
            diff = abs(statistics.mean(r[metric] for r in target) - decision["absolute_thresholds"][field]["value"])
            assert diff < 1e-12, (code, metric, diff)
            reconciled.append({"round": code, "metric": metric, "absolute_difference": diff})
    for model, result in r6decision["absolute_results"].items():
        target = [r for r in by_round["R6"] if r["model"] == model]
        for metric, field in [("spinor", "spinor_rel_l2_gauge_aligned"), ("current", "current_rel_l2"), ("fidelity", "fidelity_mean")]:
            diff = abs(statistics.mean(r[metric] for r in target) - result["mean_checks"][field]["mean"])
            assert diff < 1e-12
            reconciled.append({"round": "R6", "model": model, "metric": metric, "absolute_difference": diff})

    # Audit all file links, including figure exports. Audit link itself is created below.
    missing, checked = [], []
    for match in re.finditer(r"!?\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)", template):
        target = match.group(1) or match.group(2)
        if target.startswith(("http:", "https:", "#")):
            continue
        path = Path(target)
        if not path.is_absolute():
            path = ROOT / path
        if path == OUT / "report_audit.json":
            continue
        checked.append(path.as_posix())
        if not path.exists():
            missing.append(path.as_posix())
    pytest_path = OUT / "pytest_report_only.xml"
    pytest_result = None
    if pytest_path.exists():
        suites = ET.parse(pytest_path).getroot().iter("testsuite")
        pytest_result = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        for suite in suites:
            for key in pytest_result:
                pytest_result[key] += int(suite.attrib.get(key, 0))
        assert pytest_result["tests"] == 137 and pytest_result["failures"] == pytest_result["errors"] == 0
    unchanged = all(Path(p).exists() and sha(Path(p)) == h for p, h in protection["files"].items())
    audit = {"status": "PASS" if not missing and unchanged else "FAIL", "generated_utc": datetime.now(timezone.utc).isoformat(),
             "scope": "Report-only synthesis: no new training, no re-evaluation or alteration of old metrics, no thresholds changed",
             "metric_records": len(records), "learned_training_runs": 106, "trained_oracles": 2, "untrained_oracles": 2,
             "optimizer_steps_total": sum(r["actual_optimizer_steps"] for r in records),
             "required_artifacts_present": sum(sum(k.endswith((".json", ".csv", ".npz", ".pt")) for k in row) for row in artifacts),
             "all_trained_runs_complete_5000_steps": True, "all_training_configurations_checked": True,
             "protected_source_count": len(protection["files"]), "all_protected_sources_unchanged": unchanged,
             "original_six_audits_pass": True, "decision_mean_reconciliations": reconciled,
             "local_link_occurrences_checked": len(checked), "unique_local_links": len(set(checked)), "missing_links": missing,
             "report_sha256": sha(REPORT), "template_sha256": sha(TEMPLATE), "builder_sha256": sha(Path(__file__)),
             "figure_count_in_full_index": len(figure_index), "embedded_existing_figures": len(figure_specs),
             "report_turn_full_pytest": pytest_result,
             "notes": ["Population SD uses ddof=0; n=1 oracle has no empirical uncertainty estimate.",
                       "Untrained oracle steps=5000 template metadata is preserved natively but counted as zero actual optimizer steps.",
                       "Round-one lambda=1 sensitivity reuses main seed 0 and is not counted as a new run.",
                       "Case P absent-branch phases retained in native JSON but suppressed as physical evidence in this report."]}
    save_json("report_audit.json", audit)
    assert audit["status"] == "PASS", audit
    print(json.dumps({"report": str(REPORT), "characters": len(template), "lines": len(template.splitlines()),
                      "records": len(records), "protected_sources": len(protection["files"]), "links_checked": len(checked), "audit": audit["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    build()
