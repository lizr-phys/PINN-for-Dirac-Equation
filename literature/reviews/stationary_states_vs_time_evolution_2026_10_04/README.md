# 定态量子问题研究现状调研

- [中文综述](report.md)：以束缚态、定态散射、共振的研究现状为主，区分含时演化，实验/迁移联系为补充。
- [含时与定态的结合及研究方向](combined_directions.md)：已有结合工作、选题优先级与验证标准。
- [40 篇代表文献](references.md)；[BibTeX](references.bib)；[RIS](references.ris)。
- [逐条证据层级](annotations.json)、[检索范围](search_protocol.json)、[原始核查](verified_metadata.json)。

重核引文：

```powershell
Set-Location 'E:\PINN for Quantum Physics'
python literature\reviews\stationary_states_vs_time_evolution_2026_10_04\verify_metadata.py
python literature\reviews\stationary_states_vs_time_evolution_2026_10_04\generate_reference_tables.py
```

脚本默认复用 raw/ 的原始成功响应，避免重复请求；日期核查要以查询时的出版商/数据库为准。此次选择 40 篇，完整 DOI 核查 34 条，另 6 条 arXiv 版本含一篇硕士论文。不把 metadata_verified 当作性能或科学质量验证。

本次未重跑外部论文或本地实验，仅核实文献并形成问题分类。所有项目训练和结果保留。
