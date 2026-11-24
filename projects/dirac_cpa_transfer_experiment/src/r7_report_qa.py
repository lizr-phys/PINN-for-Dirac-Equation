"""Render final PDF figures for human visual QA; does not train or change metrics."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .freeze_r7_phase_release import ROOT, sha256, write_json
from .r7_config import EXPERIMENT


def render_pdf_figures():
    folder = ROOT/"results/figures"/EXPERIMENT
    manifest = json.loads((folder/"figure_manifest.json").read_text(encoding="utf-8"))
    executable = shutil.which("pdftoppm")
    if not executable:
        raise RuntimeError("PDF render QA needs available Poppler; text extraction alone is insufficient")
    target = folder/"qa/pdf_renders"
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for figure in manifest["figures"]:
        name = figure["figure_id"]
        pdf, prefix = folder/(name+".pdf"), target/name
        command = [executable, "-r", "140", "-singlefile", "-png", str(pdf), str(prefix)]
        result = subprocess.run(command, capture_output=True, text=True, check=True,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        rendered = prefix.with_suffix(".png")
        if not rendered.exists(): raise AssertionError("PDF renderer did not create the expected evidence")
        records.append(dict(figure_id=name, pdf_sha256=sha256(pdf),
                            rendered_path=rendered.relative_to(ROOT).as_posix(), rendered_sha256=sha256(rendered),
                            command=command, stderr=result.stderr))
    write_json(folder/"qa/pdf_render_manifest.json", {"renderer": "Poppler pdftoppm via Python",
               "render_dpi": 140, "visual_inspection_separate": True, "figures": records})
    print(f"Rendered {len(records)} PDF figures for visual inspection", flush=True)


if __name__ == "__main__":
    render_pdf_figures()
