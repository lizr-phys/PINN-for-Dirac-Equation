from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader

from .plot_periodic_potential_coupled_ltpb import FIGURES, FIGURE_IDS, sha
from .train_periodic_potential_coupled_ltpb import ROOT, write_json

RENDERS = FIGURES/"pdf_previews"


def qa(*, visual_review_passed=False):
    manifest = json.loads((FIGURES/"figure_manifest.json").read_text(encoding="utf-8"))
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise RuntimeError("Poppler pdftoppm is required for PDF visual QA")
    RENDERS.mkdir(exist_ok=True)
    records, problems = [], []
    for entry in manifest["figures"]:
        identifier = entry["id"]
        issues = []
        for extension in ("png", "svg", "pdf"):
            path = FIGURES/f"{identifier}.{extension}"
            if not path.is_file() or sha(path) != entry["output_sha256"][extension]:
                issues.append(f"{extension} missing or hash mismatch")
        with Image.open(FIGURES/f"{identifier}.png") as im:
            dimensions = list(im.size)
            if im.width < 2000 or im.height < 850:
                issues.append("PNG resolution too low")
            if im.mode == "RGBA" and im.getchannel("A").getextrema() != (255, 255):
                issues.append("PNG has unexpected transparency")
        svg = (FIGURES/f"{identifier}.svg").read_text(encoding="utf-8")
        if "<text" not in svg:
            issues.append("SVG text is not editable")
        pages = PdfReader(FIGURES/f"{identifier}.pdf").pages
        if len(pages) != 1 or len(pages[0].extract_text()) < 40:
            issues.append("PDF page count/text extraction failed")
        prefix = RENDERS/identifier
        subprocess.run([pdftoppm, "-f", "1", "-singlefile", "-r", "150", "-png", str(FIGURES/f"{identifier}.pdf"), str(prefix)], check=True, capture_output=True)
        for source in entry["source_data"]:
            path = FIGURES/source["file"]
            table = pd.read_csv(path)
            if len(table) != source["rows"] or list(table.columns) != source["columns"] or sha(path) != source["sha256"]:
                issues.append(f"Source-data mismatch: {path.name}")
            if "seed" in table and entry["seed_scope"] == "all five seeds" and set(table.seed) != set(range(5)):
                issues.append("Statistical figure omitted a seed")
            if "relative_phase_error_rad" in table:
                valid = table.phase_defined_mask.astype(bool)
                if not table.loc[~valid, "relative_phase_error_rad"].isna().all():
                    issues.append("Invalid phase entries are not masked")
                if not np.isfinite(table.loc[valid, "relative_phase_error_rad"]).all():
                    issues.append("Valid phase entries are nonfinite")
        records.append({"id": identifier, "status": "PASS" if not issues else "FAIL",
                        "png_dimensions": dimensions, "pdf_preview": f"pdf_previews/{identifier}.png",
                        "source_files": [s["file"] for s in entry["source_data"]], "problems": issues})
        problems.extend(f"{identifier}: {i}" for i in issues)
    if tuple(r["id"] for r in records) != FIGURE_IDS:
        problems.append("Figure inventory mismatch")
    for group in range(2):
        canvas = Image.new("RGB", (1800, 1420), "white")
        for index, record in enumerate(records[4*group:4*group+4]):
            with Image.open(FIGURES/record["pdf_preview"]) as im:
                thumb = ImageOps.contain(im.convert("RGB"), (890, 670))
            x, y = 900*(index%2), 710*(index//2)
            canvas.paste(thumb, (x+(900-thumb.width)//2, y+25))
            ImageDraw.Draw(canvas).text((x+10, y+5), record["id"], fill="black")
        canvas.save(RENDERS/f"contact_sheet_{group+1}.png")
    status = "FAIL" if problems else "PASS" if visual_review_passed else "PENDING_VISUAL_REVIEW"
    output = {"status": status, "automated_pass": not problems, "visual_review_passed": visual_review_passed,
              "figures": records, "problems": problems, "backend": "Python/matplotlib; Python-invoked Poppler rendering"}
    write_json(FIGURES/"figure_qa.json", output)
    lines = ["# Periodic-potential coupled-LTPB figure QA", "", f"Status: **{status}**", "",
             "全部图由 Python/matplotlib 生成；PNG/SVG/PDF、source CSV 哈希、可编辑文字、五 seed 完整性及相位 mask 已检查。PDF 全部经 Poppler 渲染。",
             "", "| Figure | Automated | PDF preview |", "|---|---|---|"]
    for r in records:
        lines.append(f"| {r['id']} | {r['status']} | [preview]({r['pdf_preview']}) |")
    lines.extend(["", "视觉检查："+("PASS。逐幅检查 PNG 与 PDF 渲染图：文字、图例、色条和面板无遮挡或裁切；参考/预测共用色限。统计图均包含五个种子；精度–时间图中接近的点可能重合，逐 seed 数值均保存在 source CSV。" if visual_review_passed else "待逐幅检查 PNG 与 PDF 渲染图。"),
                  "", "统计点为五个配对随机 seeds；中心为均值，误差棒为 population SD（ddof=0）。没有显著性检验或 p 值。动力学图固定 seed 0。",
                  "", "F2 的范数漂移面板使用 symlog（线性阈值 10⁻³），完整显示均值 ± SD 跨零的统计区间；负端不表示观测到负漂移。其余五个面板使用普通对数轴。",
                  "", "热图数据层以 400 dpi 嵌入 SVG/PDF，避免 PDF 查看器产生多边形接缝；文字、坐标轴和曲线保持可编辑矢量。未平滑、裁剪或修改原始场值，全部热图值均提供 source CSV。",
                  "", "F7 只在保存的有效 mask 内展示相对相位。无效相位保持缺失，不作为零误差。", ""])
    (FIGURES/"FIGURE_QA.md").write_text("\n".join(lines), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-review-passed", action="store_true")
    args = parser.parse_args()
    result = qa(visual_review_passed=args.visual_review_passed)
    print(json.dumps({"status": result["status"], "problems": result["problems"]}))
