from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures" / "multimode_ltpb"
TEMP = ROOT / "tmp" / "pdfs" / "multimode_ltpb"

EXPECTED_ROWS = {
    "primary_error_comparison": 75,
    "per_seed_results": 50,
    "representative_density_current": 10_050,
    "learned_four_phases": 4_020,
    "modal_coefficient_errors": 200,
    "training_curves": 12_525,
    "accuracy_time_tradeoff": 25,
}


def run_qa(figure_dir: Path = FIGURES, *, visual_pass: bool = False) -> dict[str, object]:
    manifest_path = figure_dir / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["figures"]
    if TEMP.exists():
        shutil.rmtree(TEMP)
    TEMP.mkdir(parents=True, exist_ok=True)
    pdftoppm = shutil.which("pdftoppm")
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - environment audit records this
        PdfReader = None

    records: list[dict[str, object]] = []
    for item in manifest:
        name = str(item["id"])
        png = ROOT / str(item["outputs"]["png"])
        svg = ROOT / str(item["outputs"]["svg"])
        pdf = ROOT / str(item["outputs"]["pdf"])
        source = ROOT / str(item["source_data"])
        checks: dict[str, object] = {}
        for path in (png, svg, pdf, source):
            checks[f"exists_{path.suffix.lstrip('.')}"] = path.is_file() and path.stat().st_size > 0

        with Image.open(png) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (0.0, 0.0))
        checks["png_width_px"] = width
        checks["png_height_px"] = height
        checks["png_dpi"] = [float(dpi[0]), float(dpi[1])]
        checks["png_resolution_pass"] = width >= 2500 and height >= 1400
        checks["png_dpi_pass"] = min(float(dpi[0]), float(dpi[1])) >= 590.0

        svg_text = svg.read_text(encoding="utf-8")
        checks["svg_editable_text_pass"] = "<text" in svg_text
        checks["svg_no_embedded_raster_pass"] = "<image" not in svg_text

        frame = pd.read_csv(source)
        checks["source_rows"] = len(frame)
        checks["source_rows_expected"] = EXPECTED_ROWS[name]
        checks["source_rows_pass"] = len(frame) == EXPECTED_ROWS[name]
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        checks["source_numeric_finite_pass"] = bool(np.all(np.isfinite(numeric)))

        if PdfReader is not None:
            reader = PdfReader(str(pdf))
            extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
            checks["pdf_page_count"] = len(reader.pages)
            checks["pdf_text_extractable_pass"] = len(extracted.strip()) >= 20
        else:
            checks["pdf_page_count"] = None
            checks["pdf_text_extractable_pass"] = False
        render_path: str | None = None
        if pdftoppm:
            prefix = TEMP / name
            subprocess.run(
                [pdftoppm, "-png", "-singlefile", "-r", "180", str(pdf), str(prefix)],
                check=True,
                capture_output=True,
            )
            rendered = prefix.with_suffix(".png")
            render_path = rendered.relative_to(ROOT).as_posix()
            with Image.open(rendered) as image:
                checks["pdf_render_width_px"], checks["pdf_render_height_px"] = image.size
            checks["pdf_render_pass"] = rendered.is_file() and rendered.stat().st_size > 0
        else:
            checks["pdf_render_pass"] = False
        automated_keys = [key for key in checks if key.endswith("_pass")]
        automated_pass = all(bool(checks[key]) for key in automated_keys)
        records.append(
            {
                "id": name,
                "checks": checks,
                "automated_pass": automated_pass,
                "pdf_render": render_path,
                "visual_inspection_pass": bool(visual_pass),
                "overall_pass": bool(automated_pass and visual_pass),
            }
        )

    payload = {
        "figure_count": len(records),
        "pdftoppm_available": bool(pdftoppm),
        "pypdf_available": PdfReader is not None,
        "visual_inspection_recorded": bool(visual_pass),
        "figures": records,
        "status": "PASS" if records and all(bool(row["overall_pass"]) for row in records) else "FAIL",
    }
    (figure_dir / "figure_qa.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Multi-LTPB Figure QA",
        "",
        "图表依据预注册的 `MULTIMODE_LTPB_FIGURE_CONTRACT.md` 生成。误差条均为五 seeds 的 population SD (`ddof=0`)，不是置信区间。PDF 使用 Poppler 渲染检查；可提取文字检查不能替代视觉检查。",
        "",
        f"总体状态：**{payload['status']}**。",
        "",
        "| Figure | Source rows | PNG 600 dpi | Editable SVG text | PDF text/render | Visual crop/labels/legend | Result |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in records:
        checks = row["checks"]
        pdf_ok = bool(checks["pdf_text_extractable_pass"] and checks["pdf_render_pass"])
        lines.append(
            f"| `{row['id']}` | {checks['source_rows']}/{checks['source_rows_expected']} | {'PASS' if checks['png_dpi_pass'] and checks['png_resolution_pass'] else 'FAIL'} | {'PASS' if checks['svg_editable_text_pass'] and checks['svg_no_embedded_raster_pass'] else 'FAIL'} | {'PASS' if pdf_ok else 'FAIL'} | {'PASS' if row['visual_inspection_pass'] else 'PENDING'} | {'PASS' if row['overall_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 视觉检查项目",
            "",
            "逐图检查裁切、重叠、字号、对数轴、单位、颜色一致性、图例遮挡和 exact/learned 语义。只有在最新 PNG 与 PDF 渲染均人工查看且无缺陷后，脚本才以 `--visual-pass` 记录 PASS。",
            "",
            "机器可读详情见 [figure_qa.json](figure_qa.json)，PDF 临时渲染位于 `tmp/pdfs/multimode_ltpb/`。",
            "",
        ]
    )
    (figure_dir / "FIGURE_QA.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Multi-LTPB figure artifacts")
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--visual-pass", action="store_true")
    args = parser.parse_args()
    payload = run_qa(args.figure_dir, visual_pass=args.visual_pass)
    print(json.dumps({"status": payload["status"], "figure_count": payload["figure_count"]}))
    raise SystemExit(0 if payload["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
