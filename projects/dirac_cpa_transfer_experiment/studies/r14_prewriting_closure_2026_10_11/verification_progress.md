# R14 独立验证进度

更新时间：2026-10-13T18:15:26+08:00。整体证据目标尚未验收。

正式固定终点 167/210；开发 24/24；数值/依赖失败 0。完成训练不代表准确度通过，完整五种子判定见独立汇总。

驱动器真实存活：True；活跃重复训练：False；冻结参考改动：0。科学源码SHA：0c2bf352f6bfd12ad106b75ae0814e09abf9af51ac75b59dbc3e6665d595f0c9。

| 验证项 | 当前记录 |
| --- | --- |
| theoretical_closure | [FROZEN_EQUATION_DERIVATIONS_REVIEWED_EXPERIMENTAL_CLOSURE_PENDING](engineering/theory_closure_audit.json) |
| attempts | [INCOMPLETE](analysis_outputs/attempt_completion_audit.json) |
| metrics | [PASS_AVAILABLE_SNAPSHOTS](analysis_outputs/independent_metric_audit.json) |
| residuals | [PASS_AVAILABLE_FIXED_ENDPOINTS](analysis_outputs/residual_audit.json) |
| residual_sources | [PASS_CURRENT_RESIDUAL_SOURCE_CHAIN](analysis_outputs/residual_provenance_audit.json) |
| postprocessing | WAITING_FOR_FROZEN_TRAINING（本地档案） |
| costs | [PASS_AVAILABLE_COST_RECORDS](analysis_outputs/independent_cost_audit.json) |
| cost_summary | [AVAILABLE_ATTEMPTS_ONLY](analysis_outputs/cost_summary.json) |
| statistics | [PASS_AVAILABLE_STATISTICAL_TABLES](analysis_outputs/independent_statistical_audit.json) |
| inference_cost_provenance | PASS_AVAILABLE_INFERENCE_RECORDS（本地档案） |
| parameter_counts | [PASS_AVAILABLE_PARAMETER_RECORDS](analysis_outputs/independent_parameter_audit.json) |
| spectral_benchmark | [PASS](analysis_outputs/independent_TSFP_audit.json) |
| pairing | [ALL_PRESCRIBED_BLOCKS_ACCOUNTED](analysis_outputs/pairing_audit.json) |
| phase | [PASS_AVAILABLE_PHASE_DIAGNOSTICS](analysis_outputs/carrier_phase_audit.json) |
| carrier_initial_controls | [ALL_INITIAL_CARRIER_CONTROLS_VERIFIED](analysis_outputs/carrier_initialization_audit.json) |
| learning_advantage | [ASSESSED_FIXED_PRIMARY_COMPARISONS](analysis_outputs/learning_advantage_assessment.json) |
| isolated_inference | WAITING_FOR_FIXED_ENDPOINTS（本地档案） |
| history_early | [PASS_AVAILABLE_LEGACY_ENDPOINTS](history/independent_R1_R8_audit.json) |
| history_late | [PASS](history/independent_R9_R13_audit.json) |
| history_snapshots | [PASS_CANONICAL_SNAPSHOTS](history/independent_canonical_snapshot_audit.json) |
| history_row_links | [PASS_CANONICAL_ROW_LINKS](history/canonical_row_traceability_audit.json) |
| zero_shot_audit | WAITING_FOR_PRESCRIBED_DIAGNOSTICS（本地档案） |
| links | REQUIRES_REVIEW（本地档案） |
| zero_shot | NOT_EXECUTED（尚无产物） |
| visual | [ACTUAL_VISUAL_CHECK_RECORDS](figures/actual_visual_QA.json) |

未完成：全部预定训练与终点验证、15个核心配对块、45项零样本、全部模型推理/内存测量、正式图组、最终主张证据表及投稿定位。数字以各项最近运行时间为准；它们不会随训练自动被标为完成。
