"""Run the full repository test suite and final R11 audit after visual QA."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from .r11_config import ROOT, RAW, FIGURES, LOGS
from .freeze_r7_phase_release import write_json


def verify():
    from .r7_runtime import set_cpu_affinity
    set_cpu_affinity(128)
    plan = json.loads((RAW / "runtime_plan.json").read_text(encoding="utf-8"))
    qa = json.loads((FIGURES / "FIGURE_QA.json").read_text(encoding="utf-8"))
    if plan["status"] != "COMPLETE" or qa["status"] != "PASS" or qa["visual_inspection_pending"]:
        raise RuntimeError("All training and actual visual inspection must finish first")
    if not (ROOT / "R11_INDEPENDENT_REPLICATION_ABLATION_REPORT.md").is_file():
        raise RuntimeError("Build the complete report before the full repository test suite")
    result_path = RAW / "pytest_final_result.json"
    log_path = LOGS / "pytest_final.log"
    xml_path = RAW / "pytest_final.xml"
    if any(p.exists() for p in (result_path, log_path, xml_path)):
        raise RuntimeError("Existing full-test evidence is preserved; do not overwrite it")
    write_json(RAW / "postprocessing_status.json", dict(stage="full_pytest", status="RUNNING",
        utc=datetime.now(timezone.utc).isoformat()))
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    started = datetime.now(timezone.utc)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run([sys.executable, "-m", "pytest", "-q",
            "--junitxml=" + str(xml_path)], cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=False)
    suites = list(ET.parse(xml_path).getroot().iter("testsuite")) if xml_path.exists() else []
    totals = {key: sum(int(s.attrib.get(key, "0")) for s in suites)
        for key in ("tests", "errors", "failures", "skipped")}
    result = dict(status="PASS" if completed.returncode == 0 else "FAIL", **totals,
        passed=totals["tests"]-totals["errors"]-totals["failures"]-totals["skipped"],
        exit_code=completed.returncode, started_utc=started.isoformat(),
        completed_utc=datetime.now(timezone.utc).isoformat(),
        scope="complete repository pytest", cpu_affinity_mask=128,
        log=str(log_path), junit_xml=str(xml_path))
    write_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "PASS":
        write_json(RAW / "postprocessing_status.json", dict(stage="full_pytest", status="FAIL",
            utc=datetime.now(timezone.utc).isoformat(), evidence=str(result_path)))
        raise RuntimeError("Full pytest failed; retain and inspect the recorded evidence")
    # Add the completed full-test result to the report before checking its links.
    from .build_r11_ablation_report import build
    build()
    completed = subprocess.run([sys.executable, "-m", "src.audit_r11_ablation", "--final"],
        cwd=ROOT, env=environment, check=False)
    if completed.returncode:
        raise RuntimeError("Final audit failed; preserve its detailed audit record")
    print("R11 full pytest and final audit PASS", flush=True)


if __name__ == "__main__":
    verify()
