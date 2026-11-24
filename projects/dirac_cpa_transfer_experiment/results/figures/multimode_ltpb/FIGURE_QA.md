# Multi-LTPB Figure QA

图表依据预注册的 `MULTIMODE_LTPB_FIGURE_CONTRACT.md` 生成。误差条均为五 seeds 的 population SD (`ddof=0`)，不是置信区间。PDF 使用 Poppler 渲染检查；可提取文字检查不能替代视觉检查。

总体状态：**PASS**。

| Figure | Source rows | PNG 600 dpi | Editable SVG text | PDF text/render | Visual crop/labels/legend | Result |
|---|---:|---|---|---|---|---|
| `primary_error_comparison` | 75/75 | PASS | PASS | PASS | PASS | PASS |
| `per_seed_results` | 50/50 | PASS | PASS | PASS | PASS | PASS |
| `representative_density_current` | 10050/10050 | PASS | PASS | PASS | PASS | PASS |
| `learned_four_phases` | 4020/4020 | PASS | PASS | PASS | PASS | PASS |
| `modal_coefficient_errors` | 200/200 | PASS | PASS | PASS | PASS | PASS |
| `training_curves` | 12525/12525 | PASS | PASS | PASS | PASS | PASS |
| `accuracy_time_tradeoff` | 25/25 | PASS | PASS | PASS | PASS | PASS |

## 视觉检查项目

逐图检查裁切、重叠、字号、对数轴、单位、颜色一致性、图例遮挡和 exact/learned 语义。只有在最新 PNG 与 PDF 渲染均人工查看且无缺陷后，脚本才以 `--visual-pass` 记录 PASS。

机器可读详情见 [figure_qa.json](figure_qa.json)，PDF 临时渲染位于 `tmp/pdfs/multimode_ltpb/`。
