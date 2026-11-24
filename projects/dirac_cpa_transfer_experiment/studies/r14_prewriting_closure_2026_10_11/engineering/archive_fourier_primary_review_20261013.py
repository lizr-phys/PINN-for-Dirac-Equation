"""Archive the available fixed m5T5 Fourier primary point; never train or accept R14."""
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RUN = BASE / "runtime/incremental_audit_runs/1791867510179330500"
CHECK = BASE / "runtime/cohort_manual_checks/1791868260596759400/verification.json"
RECEIPT = BASE / "analysis_outputs/cohort_receipts/m5T5_fourier_20000_first/receipt.json"
NOTE = BASE / "analysis_outputs/case_notes/m5T5_fourier_primary_20261013.md"
REVIEW = RUN / "m5T5_fourier_primary_manual_review.json"
SOURCES = RUN / "m5T5_fourier_primary_sources"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(relative):
    with (BASE / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual, expected):
    assert abs(float(actual) - float(expected)) < 1e-7, (actual, expected)


def select(table):
    result = [r for r in table if r.get("case") == "m5T5"
              and r.get("group") == "fourier" and r.get("step") == "20000"]
    assert len(result) == 5 and sorted(int(r["seed"]) for r in result) == list(range(810, 815))
    return {int(r["seed"]): r for r in result}


def main():
    assert not NOTE.exists() and not REVIEW.exists() and not SOURCES.exists()
    check, receipt = read(CHECK), read(RECEIPT)
    assert check["status"] == "PASS_SOURCE_AND_FROZEN_DECISION_CHECK"
    assert check["receipt_sha256"] == sha(RECEIPT)
    assert (check["case"], check["group"], check["step"]) == ("m5T5", "fourier", 20000)
    assert check["this_is_prescribed_training_terminal"] is False
    assert check["prescribed_training_budget"] == 26000
    seeds = read(BASE / "frozen/design.json")["formal_seeds"]
    assert seeds == receipt["all_five_seeds"] == list(range(810, 815))
    selection = read(BASE / "frozen/baseline_selection.json")
    assert selection["global_candidate"]["fourier"] == 0
    assert selection["strongest_family"] == "fourier"
    dev = [r for r in rows("analysis_outputs/development_costs_all_candidates.csv") if r["family"] == "fourier"]
    assert len(dev) == 12
    assert {(r["case"], int(r["candidate"]), int(r["seed"])) for r in dev} == {
        (case, candidate, seed) for case in ("T10", "m10") for candidate in range(3) for seed in (910, 911)}
    sources, development = [], []
    selected_dev = [r for r in selection["development_rows"] if r["family"] == "fourier"]
    assert len(selected_dev) == 12
    for row in dev:
        directory = f"development/{row['case']}/fourier/candidate_{row['candidate']}/seed_{row['seed']}"
        metric_path, complete_path = BASE / directory / "snapshots/04000/metrics.json", BASE / directory / "complete.json"
        completed = read(complete_path)
        assert completed["status"] == "COMPLETE" and completed["step"] == 4000
        assert completed["job"]["budget"] == 4000 and completed["fixed_endpoint_reached"]
        for key in ("training_wall_seconds", "training_cpu_seconds", "incurred_pipeline_seconds", "unit_spinor_residual_points", "peak_rss_bytes"):
            close(row[key], completed[key])
        close(completed["incurred_pipeline_seconds"], sum(completed[k] for k in (
            "training_wall_seconds", "evaluation_seconds", "checkpoint_io_seconds", "setup_seconds")))
        chosen = [r for r in selected_dev if r["case"] == row["case"] and r["candidate"] == int(row["candidate"]) and r["seed"] == int(row["seed"])]
        assert len(chosen) == 1 and chosen[0]["sha256"] == sha(metric_path)
        sources.extend((metric_path, complete_path))
        development.append(dict(row=row, unique_completed_cost=completed, metric_sha256=sha(metric_path), completed_sha256=sha(complete_path)))
    search = {key: sum(float(r[key]) for r in dev) for key in (
        "training_wall_seconds", "training_cpu_seconds", "incurred_pipeline_seconds", "unit_spinor_residual_points")}
    costs = select(rows("analysis_outputs/algorithm_costs_with_search_overhead.csv"))
    work = select(rows("analysis_outputs/independent_cost_workload_checks.csv"))
    inference = select(rows("isolated_inference/independent_costs.csv"))
    residual = select(rows("analysis_outputs/independent_residual_checks.csv"))
    parameters = {int(r["seed"]): r for r in rows("analysis_outputs/independent_parameter_counts.csv")
                  if r["case"] == "m5T5" and r["group"] == "fourier" and r["stage"] == "core"}
    assert sorted(parameters) == seeds
    native_metric_sources = {r["seed"]: r for r in receipt["per_seed_sources"]}
    per_seed = []
    for seed in seeds:
        directory = BASE / f"formal/coupled/m5T5/fourier/seed_{seed}"
        point = directory / "snapshots/20000"
        cost_path, config_path = point / "cost.json", directory / "config.json"
        native, config, cost, w, inf, res, param = read(cost_path), read(config_path), costs[seed], work[seed], inference[seed], residual[seed], parameters[seed]
        assert config["job"]["seed"] == seed and config["job"]["budget"] == 26000
        assert config["job"]["options"] == {"scale": 1.0, "lr": 0.001}
        assert config["bounds"] == [-64, 96] and config["configuration"]["mass"] == 5.0 and config["configuration"]["t_max"] == 5.0
        assert native["step"] == 20000 and native["free_steps"] == 0
        assert native["source_pretraining_wall_seconds"] == native["source_pretraining_cpu_seconds"] == 0
        for column, key in (("standalone_optimization_wall_seconds", "training_wall_seconds"),
                            ("standalone_optimization_cpu_seconds", "training_cpu_seconds"),
                            ("standalone_recorded_pipeline_seconds", "incurred_pipeline_seconds"),
                            ("standalone_unit_spinor_residual_points", "unit_spinor_residual_points"),
                            ("standalone_phasewise_training_peak_rss_bytes", "peak_rss_bytes")):
            close(cost[column], native[key])
        close(native["incurred_pipeline_seconds"], sum(native[k] for k in (
            "training_wall_seconds", "evaluation_seconds", "checkpoint_io_seconds", "setup_seconds")))
        close(cost["relevant_family_candidate_search_wall_seconds"], search["training_wall_seconds"])
        close(cost["relevant_family_candidate_search_cpu_seconds"], search["training_cpu_seconds"])
        close(cost["relevant_family_candidate_search_pipeline_seconds"], search["incurred_pipeline_seconds"])
        close(cost["relevant_family_candidate_search_unit_spinor_residual_points"], search["unit_spinor_residual_points"])
        close(cost["standalone_optimization_plus_relevant_search_seconds"], native["training_wall_seconds"] + search["training_wall_seconds"])
        close(cost["standalone_recorded_pipeline_plus_relevant_search_seconds"], native["incurred_pipeline_seconds"] + search["incurred_pipeline_seconds"])
        close(cost["standalone_unit_spinor_residual_points_plus_relevant_search"], native["unit_spinor_residual_points"] + search["unit_spinor_residual_points"])
        close(cost["five_seed_ensemble_mean_optimization_plus_once_search_seconds"], native["training_wall_seconds"] + search["training_wall_seconds"] / 5)
        close(cost["five_seed_ensemble_mean_pipeline_plus_once_search_seconds"], native["incurred_pipeline_seconds"] + search["incurred_pipeline_seconds"] / 5)
        assert w["pass_check"] == param["pass_check"] == inf["pass_check"] == res["pass_check"] == "True"
        assert not w["failed_checks"] and w["cost_record_sha256"] == sha(cost_path)
        assert int(w["independently_computed_own_residual_points"]) == 20000 * 32 * 161 == native["unit_spinor_residual_points"]
        assert int(w["independently_computed_auxiliary_points"]) == 0
        assert int(param["active_parameters"]) == int(param["total_parameters"]) == native["active_parameters"] == 16900
        assert int(param["frozen_parameters"]) == native["frozen_parameters"] == 0
        assert param["config_sha256"] == sha(config_path) and int(param["fixed_Fourier_feature_coefficients"]) == 64
        source = native_metric_sources[seed]
        for row in (inf, res):
            assert row["checkpoint_sha256"] == source["checkpoint_sha256"]
        assert res["prediction_sha256"] == inf["archived_prediction_sha256"] == source["prediction_sha256"]
        assert res["metric_sha256"] == inf["metric_record_sha256"] == source["metric_sha256"]
        assert res["no_reference_trajectory_loaded"] == res["no_optimizer_or_weight_update"] == "True"
        inf_path = BASE / inf["path"]
        inf_native = read(inf_path)
        assert sha(inf_path) == inf["record_sha256"] and inf_native["status"] == "COMPLETE_VERIFIED_INFERENCE"
        assert len(inf_native["inference_seconds_samples"]) == len(inf_native["inference_cpu_seconds_samples"]) == 7
        assert inf_native["evaluation_reference_not_loaded"] and inf_native["points_per_call"] == 201 * 161
        close(inf["inference_seconds_median"], statistics.median(inf_native["inference_seconds_samples"]))
        close(inf["inference_cpu_seconds_median"], statistics.median(inf_native["inference_cpu_seconds_samples"]))
        close(cost["isolated_inference_seconds_median"], inf["inference_seconds_median"])
        close(cost["isolated_inference_peak_rss_bytes"], inf["process_peak_rss_bytes"])
        sources.extend((cost_path, config_path, inf_path))
        per_seed.append(dict(seed=seed, cost=cost, native_cost=native, inference=inf, independent_workload=w,
                             independent_JVP=res, independent_parameters=param,
                             native_cost_sha256=sha(cost_path), native_config_sha256=sha(config_path)))
    unknown_path = BASE / "runtime/recovery_events/1791852771281141600/unknown_replay_costs_checkpoint_bound_corrected.csv"
    replay = [r for r in rows(unknown_path.relative_to(BASE)) if r["case"] == "m5T5" and r["group"] == "fourier"]
    assert len(replay) == 1 and replay[0]["seed"] == "811" and int(replay[0]["last_saved_step"]) == 21500
    assert not replay[0]["uncheckpointed_wall_seconds"] and not replay[0]["uncheckpointed_CPU_seconds"]
    sources.append(unknown_path)
    tables = ["analysis_outputs/algorithm_costs_with_search_overhead.csv", "analysis_outputs/development_costs_all_candidates.csv",
              "analysis_outputs/independent_cost_audit.json", "analysis_outputs/independent_cost_workload_checks.csv",
              "analysis_outputs/independent_statistical_audit.json", "isolated_inference/independent_costs.csv",
              "isolated_inference/independent_audit.json", "frozen/baseline_selection.json",
              "analysis_outputs/independent_residual_checks.csv", "analysis_outputs/residual_audit.json",
              "analysis_outputs/residual_provenance_audit.json", "analysis_outputs/independent_parameter_counts.csv",
              "analysis_outputs/independent_parameter_audit.json"]
    sources.extend(BASE / p for p in tables)
    columns = ("standalone_optimization_wall_seconds", "standalone_optimization_cpu_seconds", "standalone_recorded_pipeline_seconds",
               "standalone_optimization_plus_relevant_search_seconds", "standalone_recorded_pipeline_plus_relevant_search_seconds",
               "five_seed_ensemble_mean_optimization_plus_once_search_seconds", "five_seed_ensemble_mean_pipeline_plus_once_search_seconds",
               "standalone_unit_spinor_residual_points", "standalone_unit_spinor_residual_points_plus_relevant_search",
               "standalone_phasewise_training_peak_rss_bytes", "isolated_inference_seconds_median", "isolated_inference_peak_rss_bytes")
    means = {k: statistics.fmean(float(costs[s][k]) for s in seeds) for k in columns}
    SOURCES.mkdir(parents=True, exist_ok=False)
    preserved = []
    for source in sources:
        target = SOURCES / source.relative_to(BASE)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        with target.open("xb") as stream:
            stream.write(data)
        assert sha(target) == sha(source)
        preserved.append(dict(source=source.relative_to(BASE).as_posix(), preserved=target.relative_to(BASE).as_posix(), sha256=sha(target)))
    review = dict(reviewed_utc=time.time(), status="PASS_FIXED_PRIMARY_METRIC_COST_SELECTION_RESIDUAL_AND_SOURCE_CHECK",
                  scientific_status=check["scientific_status"], joint_successes=check["joint_successes"], all_five_seeds=seeds,
                  reviewer_sha256=sha(Path(__file__)), primary_only=True, terminal_full_five_independent_closure_not_asserted=True,
                  finite_metric_check=CHECK.relative_to(BASE).as_posix(), finite_metric_check_sha256=sha(CHECK),
                  receipt_sha256=sha(RECEIPT), means=means, search_cost=search, development_all12=development, per_seed=per_seed,
                  preserved_sources=preserved, interrupted_seed811_unknown_costs=replay,
                  unknown_replay_after_primary_step_not_added_to_earlier_primary_cost=True, whole_goal_complete=False)
    with REVIEW.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(review, stream, ensure_ascii=False, indent=2)
    m = check["means"]
    lines = ["# m5T5 预选Fourier特征适配基线：完整20,000主点材料", "",
             f"正式种子810–814的20,000步主点独立核验齐全，{check['joint_successes']}/5，{check['scientific_status']}。26,000步为原预定训练终点，其全五种子独立终点闭环尚未在本材料验收。继续原预算，不按正式结果重新调参或更换种子。", "",
             "| 主点 | 联合通过 | 旋量相对L2（时空单一常相位对齐） | 流相对L2 | 时间平均保真度 | 模态布居相对L2 | 最大范数漂移均值 |",
             "|---:|---:|---:|---:|---:|---:|---:|",
             f"| 20,000 | {check['joint_successes']}/5 | {m['spinor_rel_l2_gauge_aligned']:.12g} | {m['current_rel_l2']:.12g} | {m['fidelity_mean']:.12g} | {m['modal_population_rel_l2']:.12g} | {m['max_norm_drift']:.6g} |", "",
             "阈值依次为≤0.10、≤0.15、≥0.98、≤0.10、≤0.02，整组另需至少4/5联合通过。范数守恒属于硬约束的结构性质；动态准确性须依据其余独立指标。逐种子数据及各项失败均保留。", "",
             "| 种子 | 旋量相对L2 | 流相对L2 | 平均保真度 | 模态相对L2 | 联合通过 |", "|---:|---:|---:|---:|---:|:---:|"]
    for row in check["per_seed"]:
        v = row["values"]
        lines.append(f"| {row['seed']} | {v['spinor_rel_l2_gauge_aligned']:.10g} | {v['current_rel_l2']:.10g} | {v['fidelity_mean']:.10g} | {v['modal_population_rel_l2']:.10g} | {row['joint_pass']} |")
    lines.extend(["", "本组为Fourier特征网络在本项目谱表示、共享自由能支基底、硬初值/硬范数结构中的适配。开发阶段预选的最强神经族为Fourier，候选0（scale=1、lr=0.001）；两个开发条件×3候选×2开发种子共12次相关开发费用全部保留。每个正式种子重新训练。活跃参数16,900，固定Fourier映射64系数另计为buffer；此性能基准采用预注册调参，不承担同参数机制对照的功能。有限适配与预算结果不能外推所有Fourier特征方法。", "",
                  "| 已保存费用口径 | 五种子均值 |", "|---|---:|"])
    labels = [("优化训练墙钟/s", "standalone_optimization_wall_seconds"), ("优化训练CPU/s", "standalone_optimization_cpu_seconds"),
              ("含评价、I/O、设置的记录流水时间/s", "standalone_recorded_pipeline_seconds"),
              ("单模型承担全部相关开发费：优化墙钟/s", "standalone_optimization_plus_relevant_search_seconds"),
              ("单模型承担全部相关开发费：记录流水时间/s", "standalone_recorded_pipeline_plus_relevant_search_seconds"),
              ("五模型实验只发生一次开发费：平均优化墙钟/s", "five_seed_ensemble_mean_optimization_plus_once_search_seconds"),
              ("五模型实验只发生一次开发费：平均流水时间/s", "five_seed_ensemble_mean_pipeline_plus_once_search_seconds"),
              ("自身残差工作量/种子", "standalone_unit_spinor_residual_points"),
              ("单模型承担相关开发后的残差工作量", "standalone_unit_spinor_residual_points_plus_relevant_search"),
              ("训练进程RSS峰值均值/byte", "standalone_phasewise_training_peak_rss_bytes"),
              ("独立推理七次中位数的五种子均值/s", "isolated_inference_seconds_median"),
              ("独立推理RSS峰值均值/byte", "isolated_inference_peak_rss_bytes")]
    lines.extend(f"| {label} | {means[key]:.12g} |" for label, key in labels)
    lines.extend(["", f"12次相关开发优化墙钟合计{search['training_wall_seconds']:.12g}s、CPU {search['training_cpu_seconds']:.12g}s、记录流水{search['incurred_pipeline_seconds']:.12g}s、残差工作量{search['unit_spinor_residual_points']:.12g}。辅助载波预训练为0。开发费用按实际完成文件complete.json独立复核；主点费用按各20k快照cost.json核对，不混用较后的费用记录。项目唯一账本把各开发任务计一次，不能把五条单模型归属曲线直接相加。全部24次基线开发仍进入项目总账。", "",
                  "残差工作量单位为unit-spinor residual points，复标量分量是其两倍。5份JVP/来源、实际参数形状和工作量记录均再次核对；离散残差复核不是连续误差证书。RSS含解释器与缓存。独立推理为201时间×161模态谱输出，不含2048点空间重建或FFT指标评价；七次计时取中位数，机器同时训练，不能据此宣称普遍速度优势。", "",
                  "811曾在21,500步断点后发生工程中断并按原Adam、权重、RNG和采样恢复。未落盘0–500步的额外墙钟/CPU未知，须在其26k终点和项目总账补充披露，不填0；本20k主点的历史费用没有混入后续重放工作。", "",
                  f"[冻结设计](../../frozen/design.json)；[预选及开发来源](../../frozen/baseline_selection.json)；[指标成本口径](../../metric_and_cost_definitions.md)；[主点首次五种子判定](../cohort_receipts/m5T5_fourier_20000_first/receipt.json)；[实际SHA与冻结规则复查](../../{CHECK.relative_to(BASE).as_posix()})；[逐种子费用、开发、JVP及推理来源复查](../../{REVIEW.relative_to(BASE).as_posix()})；[未知重放费用补充](../../{unknown_path.relative_to(BASE).as_posix()})。", "",
                  "这是论文写作前的主点材料，整轮未验收。其余核心方法、扩展、45项零样本、正式图表与最终交付按冻结计划继续。", ""])
    with NOTE.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    print(json.dumps(dict(status=review["status"], note=NOTE.relative_to(BASE).as_posix(), note_sha256=sha(NOTE),
                         review=REVIEW.relative_to(BASE).as_posix(), review_sha256=sha(REVIEW), preserved_source_count=len(preserved),
                         means=means, search=search, whole_goal_complete=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
