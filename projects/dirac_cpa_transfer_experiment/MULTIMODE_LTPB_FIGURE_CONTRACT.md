# Multi-LTPB 图表契约

本契约在查看正式结果前冻结。每幅主图必须同时保存 PNG（600 dpi）、可编辑 SVG（文字保留为文字）、PDF 和 CSV source data；绘图脚本只从正式 raw/汇总产物读取数据。颜色在所有图中固定：Vanilla `#4C78A8`、SP-CPA `#F58518`、ComponentPolar `#54A24B`、MultiBranchComplex `#B279A2`、Multi-LTPB `#E45756`、exact/oracle `#222222`。不使用红绿作为唯一语义编码。

| ID | 文件基名 | 科学问题 | 固定编码与坐标 | 不确定性 | Source data 最小字段 |
|---|---|---|---|---|---|
| F1 | `primary_error_comparison` | Multi-LTPB 是否同时改善 spinor/current/fidelity？ | 三面板；误差为 log-y；fidelity 显示 `1-F` 的 log-y 并在标签注明 | 五 seed mean±population SD，叠加 seed 点 | model, seed, metric, value |
| F2 | `per_seed_results` | 优势是否由多数 paired seeds 支持？ | seed 为 x；spinor/current 两面板 log-y；同 seed 连线仅连接同模型 | 原始五 seed，无平滑 | model, seed, metric, value |
| F3 | `representative_density_current` | 代表位置 `x=0` 的密度和概率流时间演化是否正确？ | exact 黑实线；各模型五 seed 淡线和均值粗线；rho/current 两面板 | 五 seed 全曲线与均值 | model, seed, t, quantity, predicted, exact |
| F4 | `learned_four_phases` | 四个物理投影相位是否恢复四个载频？ | Multi-LTPB；四通道分面；unwrap 相位；exact 黑虚线 | 五 seed 淡线、均值粗线 | seed, mode, t, phase_predicted, phase_exact, valid |
| F5 | `modal_coefficient_errors` | 各通道的系数和占据误差是否均匀？ | 通道为 x；coefficient rel L2 与 occupation RMSE 两面板，log-y | 五 seed mean±population SD，叠加点 | model, seed, mode, metric, value |
| F6 | `training_curves` | 优化是否收敛以及不同表示的速度如何？ | step-log x、PDE loss-log y；每模型固定颜色 | 五 seed几何中心曲线及范围，保留原始值 | model, seed, step, pde_loss |
| F7 | `accuracy_time_tradeoff` | 精度收益是否伴随明显训练成本变化？ | x=training seconds，y=spinor error(log)；点面积不得编码第三变量 | 每个 run 一个点；模型均值用大标记 | model, seed, training_seconds, spinor_error |

## 图表 QA 门槛

- 所有面板标题、坐标、图例和缩写无需报告正文也可理解；时间标单位为自然单位。
- log 轴不得含非正值；若出现机器零，只能在 source data 保留原值、绘图时以明确记录的正数 floor 显示。
- 误差条定义必须在 caption 和 `FIGURE_QA.md` 中注明为五 seed population SD；不得暗示置信区间。
- PNG 尺寸和 600 dpi、SVG 的 `<text>`、PDF 可提取文字、裁切、图例遮挡、颜色一致性与 source CSV 行数均由自动审计检查；之后进行逐图视觉检查。
- Oracle 不进入 learned ranking 图；若作为参考出现，必须明确标记为“exact-carrier diagnostic”。
- 每个 source CSV 必须由审计从 raw artifact 重算关键数值，防止图与统计表漂移。
