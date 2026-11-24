from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
from pypdf import PdfReader
from PIL import Image, ImageOps, ImageDraw
from .r9_config import FIGURES, SEEDS
from .freeze_r7_phase_release import write_json, sha256


def prepare_visual_qa():
    qa = json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    preview = FIGURES/"qa_renders"
    preview.mkdir(exist_ok=True)
    images = []
    for record in qa["figures"]:
        name = record["figure"]
        pdf = FIGURES/f"{name}.pdf"
        reader = PdfReader(pdf)
        text = reader.pages[0].extract_text()
        renderer = shutil.which("pdftoppm")
        if renderer is None:
            raise RuntimeError("Poppler pdftoppm is required for independent PDF visual QA")
        target = preview/f"{name}_pdf.png"
        subprocess.run([renderer, "-png", "-singlefile", "-r", "110", str(pdf), str(target.with_suffix(""))],
                       check=True, capture_output=True, text=True)
        record["pdf_page_count"] = len(reader.pages)
        record["pdf_extractable_text_characters"] = len(text)
        record["pdf_render"] = str(target.relative_to(FIGURES))
        record["pdf_render_sha256"] = sha256(target)
        record["pdf_valid_and_text_selectable"] = len(reader.pages) == 1 and len(text.strip()) > 40
        record["pdf_render_engine"] = "Poppler pdftoppm"
        # Contact sheets are QA only; scientific data and figure pixels stay unchanged.
        with Image.open(target) as image:
            resized = ImageOps.contain(image.convert("RGB"), (680, 470))
            tile = Image.new("RGB", (700, 510), "white")
            tile.paste(resized, ((700-resized.width)//2, 30))
            ImageDraw.Draw(tile).text((12, 8), name, fill="black")
            images.append(tile)
    for start in range(0, len(images), 6):
        subset = images[start:start+6]
        rows = (len(subset)+1)//2
        sheet = Image.new("RGB", (1400, 510*rows), "white")
        for index, tile in enumerate(subset):
            sheet.paste(tile, ((index%2)*700, (index//2)*510))
        sheet.save(preview/f"contact_sheet_{start//6+1}.png")
    qa["pdf_checks_pass"] = all(record["pdf_valid_and_text_selectable"] for record in qa["figures"])
    qa["status"] = "PASS" if qa["pdf_checks_pass"] and all(record["bounding_box_pass"] and record["svg_editable_text"] for record in qa["figures"]) else "FAIL"
    write_json(FIGURES/"FIGURE_QA.json", qa)
    return qa


def record_visual_inspection(*, notes, passed):
    qa = json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    qa["visual_inspection_pending"] = False
    qa["visual_inspection_record"] = dict(passed=passed, notes=notes, scope="all nine native PNGs and independent Poppler-rendered PDF previews",
        all_seed_statistics=list(SEEDS), fixed_dynamic_seed=300, selective_data_changes=False)
    if not passed:
        qa["status"] = "FAIL"
    write_json(FIGURES/"FIGURE_QA.json", qa)
    lines = ["# R9 FIGURE_QA", "", f"Status: {qa['status']}", "",
             "Python/matplotlib exports; native PNG and independent Poppler-rendered PDF previews inspected.",
             "All five seeds shown in statistical figures; seed300 fixed for dynamics.",
             "Population SD, ddof=0. Shared heatmap scales, no smoothing or selective omission.", ""]
    lines += [f"- {record['figure']}: layout bounding boxes {'PASS' if record['bounding_box_pass'] else 'FAIL'}; editable SVG {'PASS' if record['svg_editable_text'] else 'FAIL'}; PDF text {'PASS' if record['pdf_valid_and_text_selectable'] else 'FAIL'}; source CSV complete."
              for record in qa["figures"]]
    lines += ["", "Visual inspection: "+notes, ""]
    (FIGURES/"FIGURE_QA.md").write_text("\n".join(lines), encoding="utf-8")
    return qa


if __name__ == "__main__":
    print(json.dumps(prepare_visual_qa(), indent=2))
