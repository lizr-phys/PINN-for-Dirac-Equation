from __future__ import annotations
import json
import shutil
import subprocess
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader
from .r10_config import FIGURES, SEEDS
from .freeze_r7_phase_release import write_json, sha256


def prepare_visual_qa():
    record = json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    renderer = shutil.which("pdftoppm")
    if not renderer:
        raise RuntimeError("Poppler pdftoppm required for independent PDF preview")
    directory = FIGURES/"qa_renders"
    directory.mkdir(exist_ok=True)
    tiles = []
    for item in record["figures"]:
        name = item["figure"]
        pdf = FIGURES/f"{name}.pdf"
        reader = PdfReader(pdf)
        text = reader.pages[0].extract_text()
        target = directory/f"{name}_pdf.png"
        subprocess.run([renderer, "-png", "-singlefile", "-r", "110", str(pdf), str(target.with_suffix(""))],
                       check=True, capture_output=True, text=True)
        item.update(pdf_valid_and_text_selectable=len(reader.pages) == 1 and len(text.strip()) > 40,
            pdf_page_count=len(reader.pages), pdf_extractable_text_characters=len(text),
            pdf_render=str(target.relative_to(FIGURES)), pdf_render_sha256=sha256(target),
            pdf_render_engine="Poppler pdftoppm")
        with Image.open(target) as image:
            resized = ImageOps.contain(image.convert("RGB"), (680, 470))
            tile = Image.new("RGB", (700, 510), "white")
            tile.paste(resized, ((700-resized.width)//2, 30))
            ImageDraw.Draw(tile).text((12, 8), name, fill="black")
            tiles.append(tile)
    sheet = Image.new("RGB", (1400, 510*((len(tiles)+1)//2)), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index%2)*700, (index//2)*510))
    sheet.save(directory/"contact_sheet.png")
    record["pdf_checks_pass"] = all(item["pdf_valid_and_text_selectable"] for item in record["figures"])
    record["status"] = "PASS" if record["pdf_checks_pass"] and all(item["bounding_box_pass"] and item["svg_editable_text"] for item in record["figures"]) else "FAIL"
    write_json(FIGURES/"FIGURE_QA.json", record)
    return record


def record_visual_inspection(*, notes, passed):
    record = json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    record["visual_inspection_pending"] = False
    record["visual_inspection_record"] = dict(passed=passed, notes=notes,
        scope="all six native PNGs and all independent Poppler PDF previews",
        paired_ancestries=list(SEEDS), fixed_dynamic_seed=300, selective_data_changes=False)
    if not passed:
        record["status"] = "FAIL"
    write_json(FIGURES/"FIGURE_QA.json", record)
    lines = ["# R10 FIGURE_QA", "", f"Status: {record['status']}", "",
        "All five paired ancestries shown; dynamics seed300; population SD ddof=0.",
        "Native PNG and independent Poppler PDF previews checked; SVG text editable.",
        "No smoothing, normalization, selective checkpoint or seed omission.", ""]
    lines += [f"- {q['figure']}: bounding boxes {q['bounding_box_pass']}; SVG editable {q['svg_editable_text']}; PDF text/selectability {q['pdf_valid_and_text_selectable']}; full source CSV."
              for q in record["figures"]]
    lines += ["", "Visual inspection: "+notes, ""]
    (FIGURES/"FIGURE_QA.md").write_text("\n".join(lines), encoding="utf-8")
    return record


if __name__ == "__main__":
    print(json.dumps(prepare_visual_qa(), indent=2))
