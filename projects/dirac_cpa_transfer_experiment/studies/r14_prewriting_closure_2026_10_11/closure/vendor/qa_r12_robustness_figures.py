from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader
from .r12_config import FIGURES, SEEDS
from .freeze_r7_phase_release import write_json, sha256

def prepare_visual_qa():
    record=json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    renderer=shutil.which("pdftoppm")
    bundled=Path("C:/Users/hp/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe")
    if not renderer and bundled.is_file():
        renderer=str(bundled)
    if not renderer:
        raise RuntimeError("Poppler required for independent PDF inspection")
    directory=FIGURES/"qa_renders"; directory.mkdir(exist_ok=True)
    tiles=[]
    for item in record["figures"]:
        name=item["figure"]; pdf=FIGURES/f"{name}.pdf"
        reader=PdfReader(pdf); text=reader.pages[0].extract_text()
        target=directory/f"{name}_pdf.png"
        subprocess.run([renderer,"-png","-singlefile","-r","110",str(pdf),str(target.with_suffix(""))],
                       check=True,capture_output=True,text=True)
        item.update(pdf_valid_and_text_selectable=len(reader.pages)==1 and len(text.strip())>40,
            pdf_page_count=len(reader.pages),pdf_extractable_text_characters=len(text),
            pdf_render=target.relative_to(FIGURES).as_posix(),pdf_render_sha256=sha256(target),
            pdf_render_engine="Poppler pdftoppm")
        with Image.open(target) as image:
            resized=ImageOps.contain(image.convert("RGB"),(680,470))
            tile=Image.new("RGB",(700,510),"white")
            tile.paste(resized,((700-resized.width)//2,30))
            ImageDraw.Draw(tile).text((12,8),name,fill="black"); tiles.append(tile)
    sheet=Image.new("RGB",(1500,510*((len(tiles)+1)//2)),"white")
    for index,tile in enumerate(tiles):
        sheet.paste(tile,((index%2)*700,(index//2)*510))
    sheet.save(directory/"contact_sheet.png")
    record["pdf_checks_pass"]=all(i["pdf_valid_and_text_selectable"] for i in record["figures"])
    record["status"]="PENDING_VISUAL_INSPECTION" if record["pdf_checks_pass"] and all(
        i["bounding_box_pass"] and i["svg_editable_text"] for i in record["figures"]) else "FAIL"
    write_json(FIGURES/"FIGURE_QA.json",record)
    return record

def record_visual_inspection(notes,passed):
    record=json.loads((FIGURES/"FIGURE_QA.json").read_text(encoding="utf-8"))
    record["visual_inspection_pending"]=False
    record["visual_inspection_record"]=dict(passed=passed,notes=notes,
        scope="all native PNGs and all independently rendered PDF pages",
        all_seeds=list(SEEDS),dynamics_seed=500,selective_data_changes=False)
    record["status"]="PASS" if passed and record["pdf_checks_pass"] and all(
        i["bounding_box_pass"] and i["svg_editable_text"] for i in record["figures"]) else "FAIL"
    write_json(FIGURES/"FIGURE_QA.json",record)
    lines=["# R12 FIGURE_QA","",f"Status: {record['status']}","",
        "Python/matplotlib; SVG editable text; PDF TrueType/selectable text; PNG 300dpi.",
        "All fresh paired seeds500–504; dynamics seed500; mean ± population SD(ddof=0).",
        "E* uses analytic prior and is diagnostic only. All fixed checkpoints retained.",
        "No smoothing, prediction renormalization, data modification or seed omission.",""]
    lines += [f"- {i['figure']}: bounding boxes={i['bounding_box_pass']}, SVG text={i['svg_editable_text']}, PDF text={i['pdf_valid_and_text_selectable']}; source CSV retained."
              for i in record["figures"]]
    lines += ["","Visual inspection: "+notes,""]
    (FIGURES/"FIGURE_QA.md").write_text("\n".join(lines),encoding="utf-8")
    return record

if __name__=="__main__":
    print(json.dumps(prepare_visual_qa(),indent=2))
