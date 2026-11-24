# R7 Figure QA

统计：全部种子；均值±population SD，ddof=0；固定动态seed0。无显著性推断。

程序检查源表与导出哈希、PNG尺寸、SVG可编辑文本和PDF可提取文字；核对最终PDF与渲染证据哈希。全部最终PDF已实际目视检查，详见 [目视记录](visual_review.md) 和 渲染记录（本地档案）；不能把文件完整性冒充目视通过。

| Figure | Checks | PNG | SVG text | Source tables | Seed scope |
| --- | --- | --- | ---: | ---: | --- |
| all_seed_primary_metrics | PASS | (2357, 1345) | 91 | 1 | all five seeds |
| checkpoint_accuracy | PASS | (2225, 726) | 37 | 1 | all five seeds |
| initial_tangent_and_short_dynamics | PASS | (2291, 1345) | 76 | 2 | all three short seeds; dynamics fixed seed 0 |
| main_initial_tangent | PASS | (2357, 1345) | 116 | 1 | all five main seeds |
| modal_occupation_heatmaps | PASS | (2330, 2665) | 293 | 1 | fixed seed 0 |
| sideband_trajectories | PASS | (2313, 1345) | 102 | 1 | fixed seed 0 |
| coherence_current | PASS | (2115, 1345) | 79 | 2 | all five seeds; current fixed seed 0 |
| density_current_fields | PASS | (2378, 2027) | 235 | 1 | fixed seed 0; release ordinary and causal |
| residual_weights | PASS | (2269, 1873) | 140 | 5 | all-seed dense segment source; trajectories fixed seed 0 |
| accuracy_cost | PASS | (2049, 1408) | 60 | 2 | all five seeds |
| factor_interactions | PASS | (1675, 729) | 32 | 2 | all five paired seeds |
