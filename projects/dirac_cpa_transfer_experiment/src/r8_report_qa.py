from __future__ import annotations

import json
import shutil
import subprocess
from .r8_config import ROOT, FIGURES
from .freeze_r7_phase_release import write_json, sha256


def render_figures():
    tool = shutil.which("pdftoppm")
    if not tool:
        raise RuntimeError("Available Poppler required for actual final PDF visual QA")
    manifest = json.loads((FIGURES/"figure_manifest.json").read_text(encoding="utf-8"))
    target = FIGURES/"qa/pdf_renders"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for figure in manifest["figures"]:
        name = figure["figure_id"]
        pdf, prefix = FIGURES/(name+".pdf"), target/name
        command = [tool, "-r", "140", "-singlefile", "-png", str(pdf), str(prefix)]
        result = subprocess.run(command, capture_output=True, text=True, check=True,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        rendered = prefix.with_suffix(".png")
        records.append(dict(figure_id=name, pdf_sha256=sha256(pdf),
            rendered_path=rendered.relative_to(ROOT).as_posix(), rendered_sha256=sha256(rendered),
            command=command, stderr=result.stderr))
    write_json(FIGURES/"qa/pdf_render_manifest.json", dict(renderer="Poppler via Python", figures=records,
        visual_inspection_must_be_performed_separately=True))
    print(f"Rendered {len(records)} final PDFs", flush=True)


if __name__ == "__main__":
    render_figures()
