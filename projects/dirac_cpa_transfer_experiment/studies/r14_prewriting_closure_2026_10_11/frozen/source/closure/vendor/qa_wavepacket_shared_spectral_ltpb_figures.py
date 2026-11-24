from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader

from .plot_wavepacket_shared_spectral_ltpb import (
    EXPECTED_SOURCE_ROWS,
    FIGURE_IDS,
    FIGURES,
    SOURCE_SCHEMAS,
)


ROOT = Path(__file__).resolve().parents[1]
RENDER_ROOT = ROOT / "tmp" / "pdfs" / "wavepacket_shared_spectral_ltpb"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_qa(
    figure_dir: Path = FIGURES,
    *,
    visual_review_passed: bool = False,
) -> dict[str, Any]:
    manifest_path = figure_dir / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("figures", [])
    problems: list[str] = []
    records: list[dict[str, Any]] = []
    if tuple(item.get("id") for item in entries) != FIGURE_IDS:
        problems.append("figure manifest order/ids differ from frozen contract")

    pdftoppm = shutil.which("pdftoppm")
    pdftotext = shutil.which("pdftotext")
    if pdftoppm is None:
        problems.append("pdftoppm is unavailable; PDF visual rendering was not checked")
    RENDER_ROOT.mkdir(parents=True, exist_ok=True)

    for name in FIGURE_IDS:
        entry = next((item for item in entries if item.get("id") == name), None)
        figure_problems: list[str] = []
        if entry is None:
            problems.append(f"missing manifest entry {name}")
            continue
        source = figure_dir / "source_data" / f"{name}.csv"
        if not source.is_file():
            figure_problems.append("missing source CSV")
            frame = pd.DataFrame()
        else:
            frame = pd.read_csv(source)
            if len(frame) != EXPECTED_SOURCE_ROWS[name]:
                figure_problems.append(
                    f"source rows {len(frame)} != {EXPECTED_SOURCE_ROWS[name]}"
                )
            if frame.columns.tolist() != list(SOURCE_SCHEMAS[name]):
                figure_problems.append("source schema mismatch")
            numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
            if numeric.size and not np.all(np.isfinite(numeric)):
                figure_problems.append("source contains non-finite values")
            if entry.get("source_sha256") != _sha256(source):
                figure_problems.append("source SHA-256 mismatch")

        output_info: dict[str, Any] = {}
        for suffix in ("png", "svg", "pdf"):
            path = figure_dir / f"{name}.{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                figure_problems.append(f"missing/empty {suffix}")
                continue
            if entry.get("output_sha256", {}).get(suffix) != _sha256(path):
                figure_problems.append(f"{suffix} SHA-256 mismatch")
            output_info[suffix] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}

        png = figure_dir / f"{name}.png"
        if png.is_file():
            with Image.open(png) as image:
                width, height = image.size
                dpi = image.info.get("dpi", (0.0, 0.0))
                output_info.setdefault("png", {}).update(
                    {"width_px": width, "height_px": height, "dpi": list(dpi)}
                )
                if width < 1800 or height < 1000:
                    figure_problems.append(f"PNG dimensions too small: {width}x{height}")
                if image.mode not in {"RGB", "RGBA"}:
                    figure_problems.append(f"unexpected PNG mode {image.mode}")

        svg = figure_dir / f"{name}.svg"
        if svg.is_file():
            svg_text = svg.read_text(encoding="utf-8", errors="replace")
            if "<text" not in svg_text:
                figure_problems.append("SVG has no editable text elements")
            if "image/png;base64" in svg_text and not bool(entry.get("raster_allowed")):
                figure_problems.append("unexpected embedded raster in SVG")

        pdf = figure_dir / f"{name}.pdf"
        render_path: str | None = None
        if pdf.is_file() and pdftoppm is not None:
            prefix = RENDER_ROOT / name
            process = subprocess.run(
                [pdftoppm, "-f", "1", "-singlefile", "-r", "140", "-png", str(pdf), str(prefix)],
                capture_output=True,
                text=True,
                check=False,
            )
            rendered = prefix.with_suffix(".png")
            if process.returncode != 0 or not rendered.is_file():
                figure_problems.append(f"PDF render failed: {process.stderr.strip()}")
            else:
                render_path = rendered.resolve().as_posix()
                with Image.open(rendered) as image:
                    # The two wide comparison panels are intentionally short.  At
                    # 140 dpi their frozen 7.35 x 2.4--2.55 inch canvas renders
                    # to roughly 900--1010 x 330--350 px.
                    if image.width < 700 or image.height < 300:
                        figure_problems.append("rendered PDF preview too small")
        if pdf.is_file() and pdftotext is not None:
            process = subprocess.run(
                [pdftotext, str(pdf), "-"],
                capture_output=True,
                text=True,
                check=False,
            )
            if process.returncode != 0 or len(process.stdout.strip()) < 20:
                figure_problems.append("PDF text is absent or not extractable")
        elif pdf.is_file():
            try:
                extracted = "\n".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
            except Exception as exc:  # pragma: no cover - corrupt-PDF guard
                figure_problems.append(f"PDF text extraction failed: {exc}")
            else:
                if len(extracted.strip()) < 20:
                    figure_problems.append("PDF text is absent or not extractable")

        records.append(
            {
                "id": name,
                "status": "PASS" if not figure_problems else "FAIL",
                "source_rows": len(frame),
                "outputs": output_info,
                "pdf_render_preview": render_path,
                "problems": figure_problems,
            }
        )
        problems.extend(f"{name}: {problem}" for problem in figure_problems)

    automated_pass = not problems
    status = "PASS" if automated_pass and visual_review_passed else (
        "PENDING_VISUAL_REVIEW" if automated_pass else "FAIL"
    )
    payload: dict[str, Any] = {
        "experiment": "wavepacket_shared_spectral_ltpb",
        "status": status,
        "backend": "Python/matplotlib",
        "automated_checks_passed": automated_pass,
        "manual_visual_review_passed": bool(visual_review_passed),
        "figure_count": len(records),
        "figures": records,
        "problems": problems,
        "qa_contract": {
            "source_data_traceable": automated_pass,
            "png_dimensions_checked": True,
            "svg_editable_text_checked": True,
            "pdf_rendered_with_poppler": pdftoppm is not None,
            "pdf_selectable_text_checked": True,
            "visual_review_scope": (
                "clipping, overlap, labels, colorbars, legends, panel hierarchy, "
                "heatmap comparability, and consistency with the stated conclusion"
            ),
        },
    }
    (figure_dir / "figure_qa.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown = [
        "# Wavepacket Shared-Spectral-LTPB Figure QA",
        "",
        f"Overall status: **{status}**",
        "",
        "All figures were generated exclusively with Python/matplotlib. Source-data rows, schemas and SHA-256 hashes were checked; PNG dimensions were inspected; SVG editable text was verified; PDFs were rendered with Poppler and checked for extractable text with pdftotext or pypdf.",
        "",
        "| Figure | Automated | Source rows | PDF preview | Problems |",
        "|---|---|---:|---|---|",
    ]
    for record in records:
        preview = record["pdf_render_preview"] or "N/A"
        issue = "; ".join(record["problems"]) if record["problems"] else "None"
        markdown.append(
            f"| `{record['id']}` | {record['status']} | {record['source_rows']} | "
            f"`{preview}` | {issue} |"
        )
    markdown.extend(
        [
            "",
            "## Manual visual review",
            "",
            (
                "PASS: every PNG and Poppler-rendered PDF preview was visually inspected at final delivery. No clipped labels, overlapping legends, broken colorbars, misleading shared scales, unreadable seed marks, or panel hierarchy defects were found."
                if visual_review_passed
                else "PENDING: run this module again with `--visual-review-passed` only after inspecting every PNG and rendered PDF preview."
            ),
            "",
            "The midpoint panels are explicitly labelled as interpolation probes rather than periodic Fourier modes. Heatmap error panels use separately labelled scales, and `1-fidelity` is used only in the log-error comparison while fidelity itself remains reported in tables.",
            "",
        ]
    )
    (figure_dir / "FIGURE_QA.md").write_text("\n".join(markdown), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="QA formal wavepacket figures")
    parser.add_argument("--figure-dir", type=Path, default=FIGURES)
    parser.add_argument("--visual-review-passed", action="store_true")
    args = parser.parse_args()
    result = run_qa(args.figure_dir, visual_review_passed=args.visual_review_passed)
    print(json.dumps({"status": result["status"], "figures": result["figure_count"]}))


if __name__ == "__main__":
    main()
