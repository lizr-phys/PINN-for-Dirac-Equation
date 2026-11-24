"""Preserve both fixed Fourier endpoints and the recorded complete-job costs."""
import csv
import json
import statistics
import time
from pathlib import Path
from archive_fourier_primary_review_20261013 import BASE, sha, read, close

RUN = BASE / "runtime/incremental_audit_runs/1791869298597865900"
TABLES = RUN / "fourier_terminal_input_tables"
CHECK = BASE / "runtime/cohort_manual_checks/1791870759096012400/verification.json"
PRIMARY = BASE / "runtime/incremental_audit_runs/1791867510179330500/m5T5_fourier_primary_manual_review.json"
PRIMARY_NOTE = BASE / "analysis_outputs/case_notes/m5T5_fourier_primary_20261013.md"
NOTE = BASE / "analysis_outputs/case_notes/m5T5_fourier_fixed_endpoints_20261013.md"
OUTPUT = RUN / "m5T5_fourier_complete_manual_review.json"
SOURCES = RUN / "m5T5_fourier_terminal_sources"


def table(relative):
    with (TABLES / relative).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def select(relative, path_suffix=None):
    rows = [r for r in table(relative) if r.get("case") == "m5T5" and r.get("group") == "fourier" and r.get("step") == "26000"]
    if path_suffix is not None:
        rows = [r for r in rows if r["path"].endswith(path_suffix)]
    assert len(rows) == 5 and sorted(int(r["seed"]) for r in rows) == list(range(810, 815))
    return {int(r["seed"]): r for r in rows}


def main():
    assert not NOTE.exists() and not OUTPUT.exists() and not SOURCES.exists()
    primary, check = read(PRIMARY), read(CHECK)
    assert check["status"] == "PASS_SOURCE_AND_FROZEN_DECISION_CHECK"
    assert (check["case"], check["group"], check["step"]) == ("m5T5", "fourier", 26000)
    assert check["this_is_prescribed_training_terminal"] and len(check["terminal_attempt_rows"]) == 5
    assert sha(PRIMARY_NOTE) == "14d12373cd9fd26d0aba41835eda12efd9e03eff58cce20bee2d4bfad58885d2"
    assert primary["reviewer_sha256"] == sha(BASE / "engineering/archive_fourier_primary_review_20261013.py")
    for source in primary["preserved_sources"]:
        assert sha(BASE / source["preserved"]) == source["sha256"]
    staged = read(TABLES / "receipt.json")
    for source in staged["files"]:
        assert sha(BASE / source["preserved"]) == source["sha256"]
    costs = select("analysis_outputs/algorithm_costs_with_search_overhead.csv")
    work = select("analysis_outputs/independent_cost_workload_checks.csv", "/snapshots/26000/cost.json")
    complete_work = select("analysis_outputs/independent_cost_workload_checks.csv", "/complete.json")
    inference = select("isolated_inference/independent_costs.csv")
    residual = select("analysis_outputs/independent_residual_checks.csv")
    param = {int(r["seed"]): r for r in table("analysis_outputs/independent_parameter_counts.csv")
             if r["case"] == "m5T5" and r["group"] == "fourier" and r["stage"] == "core"}
    assert sorted(param) == list(range(810, 815))
    search = primary["search_cost"]
    for row in primary["development_all12"]:
        close(row["unique_completed_cost"]["incurred_pipeline_seconds"], float(row["row"]["incurred_pipeline_seconds"]))
    for key in search:
        close(search[key], sum(float(r["row"][key]) for r in primary["development_all12"]))
    receipt = BASE / check["receipt_path"]
    assert sha(receipt) == check["receipt_sha256"]
    source_rows = {r["seed"]: r for r in read(receipt)["per_seed_sources"]}
    sources, per_seed = [], []
    for seed in range(810, 815):
        directory = BASE / f"formal/coupled/m5T5/fourier/seed_{seed}"
        cp, donep, configp = directory / "snapshots/26000/cost.json", directory / "complete.json", directory / "config.json"
        native, done, config, cost, w, inf, res, par = read(cp), read(donep), read(configp), costs[seed], work[seed], inference[seed], residual[seed], param[seed]
        assert done["status"] == "COMPLETE" and done["fixed_endpoint_reached"] and done["step"] == native["step"] == 26000
        assert config["job"]["budget"] == 26000 and config["job"]["options"] == {"scale": 1.0, "lr": 0.001}
        for col, key in (("standalone_optimization_wall_seconds", "training_wall_seconds"),
                         ("standalone_optimization_cpu_seconds", "training_cpu_seconds"),
                         ("standalone_recorded_pipeline_seconds", "incurred_pipeline_seconds"),
                         ("standalone_unit_spinor_residual_points", "unit_spinor_residual_points"),
                         ("standalone_phasewise_training_peak_rss_bytes", "peak_rss_bytes")):
            close(cost[col], native[key])
        for value in (native, done):
            close(value["incurred_pipeline_seconds"], sum(value[k] for k in ("training_wall_seconds", "evaluation_seconds", "checkpoint_io_seconds", "setup_seconds")))
            assert value["source_pretraining_wall_seconds"] == value["source_pretraining_cpu_seconds"] == 0
            assert value["unit_spinor_residual_points"] == 26000 * 32 * 161
        close(native["training_wall_seconds"], done["training_wall_seconds"])
        close(native["training_cpu_seconds"], done["training_cpu_seconds"])
        close(cost["standalone_optimization_plus_relevant_search_seconds"], native["training_wall_seconds"] + search["training_wall_seconds"])
        close(cost["standalone_recorded_pipeline_plus_relevant_search_seconds"], native["incurred_pipeline_seconds"] + search["incurred_pipeline_seconds"])
        close(cost["standalone_unit_spinor_residual_points_plus_relevant_search"], native["unit_spinor_residual_points"] + search["unit_spinor_residual_points"])
        close(cost["five_seed_ensemble_mean_optimization_plus_once_search_seconds"], native["training_wall_seconds"] + search["training_wall_seconds"] / 5)
        close(cost["five_seed_ensemble_mean_pipeline_plus_once_search_seconds"], native["incurred_pipeline_seconds"] + search["incurred_pipeline_seconds"] / 5)
        assert w["pass_check"] == inf["pass_check"] == res["pass_check"] == par["pass_check"] == "True"
        assert w["cost_record_sha256"] == sha(cp) and not w["failed_checks"]
        cw = complete_work[seed]
        assert cw["pass_check"] == "True" and not cw["failed_checks"] and cw["cost_record_sha256"] == sha(donep)
        assert int(cw["independently_computed_own_residual_points"]) == 133952000
        assert int(w["independently_computed_own_residual_points"]) == 133952000 and int(w["independently_computed_auxiliary_points"]) == 0
        assert int(par["active_parameters"]) == native["active_parameters"] == 16900 and int(par["frozen_parameters"]) == 0
        assert par["config_sha256"] == sha(configp)
        expected = source_rows[seed]
        assert inf["checkpoint_sha256"] == res["checkpoint_sha256"] == expected["checkpoint_sha256"]
        assert inf["archived_prediction_sha256"] == res["prediction_sha256"] == expected["prediction_sha256"]
        assert inf["metric_record_sha256"] == res["metric_sha256"] == expected["metric_sha256"]
        ip = BASE / inf["path"]
        raw = read(ip)
        assert sha(ip) == inf["record_sha256"] and raw["status"] == "COMPLETE_VERIFIED_INFERENCE"
        assert len(raw["inference_seconds_samples"]) == 7 and raw["points_per_call"] == 201 * 161
        close(inf["inference_seconds_median"], statistics.median(raw["inference_seconds_samples"]))
        close(cost["isolated_inference_seconds_median"], inf["inference_seconds_median"])
        close(cost["isolated_inference_peak_rss_bytes"], inf["process_peak_rss_bytes"])
        sources.extend((cp, donep, configp, ip))
        per_seed.append(dict(seed=seed, cost_curve=cost, native_snapshot_cost=native, unique_completed_cost=done,
                             completion_minus_snapshot_pipeline_seconds=done["incurred_pipeline_seconds"] - native["incurred_pipeline_seconds"],
                             inference=inf, independent_workload=w, independent_completed_workload=cw,
                             independent_JVP=res, independent_parameters=par))
    sources.extend(TABLES / source["source"] for source in staged["files"])
    columns = list(primary["means"])
    means = {key: statistics.fmean(float(costs[s][key]) for s in range(810, 815)) for key in columns}
    completed_pipeline_mean = statistics.fmean(r["unique_completed_cost"]["incurred_pipeline_seconds"] for r in per_seed)
    SOURCES.mkdir(parents=True, exist_ok=False)
    preserved = []
    for source in sources:
        target = SOURCES / source.relative_to(BASE)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        assert sha(source) == sha(target)
        preserved.append(dict(source=source.relative_to(BASE).as_posix(), preserved=target.relative_to(BASE).as_posix(), sha256=sha(target)))
    output = dict(reviewed_utc=time.time(), status="PASS_FIXED_TERMINAL_METRIC_COST_AND_SOURCE_CHECK",
                  checker_sha256=sha(Path(__file__)), primary_manual_review=PRIMARY.relative_to(BASE).as_posix(),
                  primary_manual_review_sha256=sha(PRIMARY), primary_note_sha256=sha(PRIMARY_NOTE),
                  all_primary_preserved_sources_rehashed=53, finite_terminal_check=CHECK.relative_to(BASE).as_posix(),
                  finite_terminal_check_sha256=sha(CHECK), all_five_training_attempts_completed=True,
                  means=means, unique_completed_pipeline_mean_seconds=completed_pipeline_mean, per_seed=per_seed,
                  preserved_sources=preserved, relevant_development_search=search,
                  unknown_seed811_extra_replay_costs=primary["interrupted_seed811_unknown_costs"],
                  unknown_costs_not_zero=True, same_five_attempts_at_two_fixed_observations=True, whole_goal_complete=False)
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    pc = read(BASE / primary["finite_metric_check"])
    lines = ["# m5T5 预选Fourier特征适配基线：两个固定端点完整材料", "",
             "正式种子810–814的26,000步规定训练终点全部完成并独立核验。20,000主点与26,000终点属于同五次训练的两个预定观测，不算十个独立种子；原主点材料保留。", "",
             "| 步数 | 联合通过 | 冻结判定 | 旋量相对L2（时空单一常相位） | 流相对L2 | 平均保真度 | 模态相对L2 | 最大范数漂移均值 |",
             "|---:|---:|---|---:|---:|---:|---:|---:|"]
    for point in (pc, check):
        m = point["means"]
        lines.append(f"| {point['step']} | {point['joint_successes']}/5 | {point['scientific_status']} | {m['spinor_rel_l2_gauge_aligned']:.12g} | {m['current_rel_l2']:.12g} | {m['fidelity_mean']:.12g} | {m['modal_population_rel_l2']:.12g} | {m['max_norm_drift']:.6g} |")
    lines.extend(["", "阈值为≤0.10、≤0.15、≥0.98、≤0.10、≤0.02，另需至少4/5联合通过。范数是硬约束的结构性质，不能作为动力学准确证据。全部失败、逐种子模型和两个点均保留；不追加预算、换种子或按正式结果重选候选。", "",
                  "本组是共享自由能支基底、谱表示、硬初值/硬范数中的Fourier特征适配。全球开发预选候选0（scale=1、lr=0.001），活跃参数16,900；固定64映射系数计为buffer。其12次相关开发费用全部计入单模型归属曲线；项目唯一账本只实际计一次。有限调参、表示与预算结果不代表所有Fourier特征方法。", "",
                  "| 已保存费用口径/五种子均值 | 20,000主点 | 26,000终点 |", "|---|---:|---:|"])
    labels = [("优化墙钟/s", "standalone_optimization_wall_seconds"), ("优化CPU/s", "standalone_optimization_cpu_seconds"),
              ("快照记录流水时间/s", "standalone_recorded_pipeline_seconds"),
              ("单模型含完整相关开发：优化墙钟/s", "standalone_optimization_plus_relevant_search_seconds"),
              ("单模型含完整相关开发：记录流水/s", "standalone_recorded_pipeline_plus_relevant_search_seconds"),
              ("五模型仅发生一次开发：平均优化墙钟/s", "five_seed_ensemble_mean_optimization_plus_once_search_seconds"),
              ("自身unit-spinor残差工作量", "standalone_unit_spinor_residual_points"),
              ("单模型含相关开发的残差工作量", "standalone_unit_spinor_residual_points_plus_relevant_search"),
              ("训练RSS峰值/byte", "standalone_phasewise_training_peak_rss_bytes"),
              ("独立推理七次中位数/s", "isolated_inference_seconds_median"),
              ("独立推理RSS峰值/byte", "isolated_inference_peak_rss_bytes")]
    lines.extend(f"| {label} | {primary['means'][key]:.12g} | {means[key]:.12g} |" for label, key in labels)
    lines.extend(["", f"完整任务complete.json的记录流水均值另为{completed_pipeline_mean:.12g}s；原费用曲线沿用对应快照cost.json。两种记录的末次保存开销差异逐种子保留，不混用终点来源。相关12次开发优化墙钟{search['training_wall_seconds']:.12g}s、CPU {search['training_cpu_seconds']:.12g}s、完整记录流水{search['incurred_pipeline_seconds']:.12g}s、残差工作量{search['unit_spinor_residual_points']:.12g}，辅助载波预训练0。", "",
                  "811在21,500步断点后的中断可能丢失0–500步工作；实际额外墙钟/CPU未知，当前26k和项目费用为已保存记录，不能把未知补充填0。此前20k费用没有后续重放工作。RSS包含运行时/缓存；独立推理只输出201×161谱表示，未包含2048点空间重建与FFT指标评价。机器同时训练，不能宣称普遍速度优势。离散JVP/来源复核不是连续误差证书。", "",
                  f"[冻结设计](../../frozen/design.json)；[预选及开发来源](../../frozen/baseline_selection.json)；[原主点完整材料](m5T5_fourier_primary_20261013.md)；[26k首次五种子判定](../cohort_receipts/m5T5_fourier_26000_first/receipt.json)；[实际SHA与冻结终点规则复查](../../{CHECK.relative_to(BASE).as_posix()})；[费用与来源完整复查](../../{OUTPUT.relative_to(BASE).as_posix()})；[未知重放补充](../../runtime/recovery_events/1791852771281141600/unknown_replay_costs_checkpoint_bound_corrected.csv)。", "",
                  "本材料完成该方法在该条件的两个固定端点闭环。其余核心与扩展、45项零样本、正式图表、最终文档及18项整体验收仍须完成。", ""])
    with NOTE.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    print(json.dumps(dict(status=output["status"], note_sha256=sha(NOTE), output_sha256=sha(OUTPUT),
                         preserved_sources=len(preserved), means=means, unique_pipeline=completed_pipeline_mean, whole_goal_complete=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
