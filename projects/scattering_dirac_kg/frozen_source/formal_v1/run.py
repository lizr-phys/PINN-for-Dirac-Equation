from __future__ import annotations
import argparse
from .physics import ROOT, read_json


def run(stage="all"):
    if stage in ("reference", "all"):
        from .reference import reference_campaign
        from .precision_reference import high_precision_campaign
        # Reuse complete, independently verified reference outputs by default.
        paths = [ROOT/"results"/"reference"/"verification.json",
                 ROOT/"results"/"reference"/"high_precision"/"verification.json"]
        if not (paths[0].exists() and read_json(paths[0])["verified"]):
            reference_campaign()
        if not (paths[1].exists() and read_json(paths[1])["verified"]):
            high_precision_campaign()
    if stage in ("fixed", "all"):
        from .campaign import fixed_campaign
        fixed_campaign()
    if stage in ("baseline", "all"):
        from .campaign import baseline_campaign
        baseline_campaign()
    if stage in ("scan", "all"):
        from .reference import reference_scan
        from .campaign import conditional_campaign, retraining_scan, sensitivity_campaign
        from .benchmark import summarize_scan_costs
        from .postprocess import complete_reference_raw_outputs, refine_transmission_extrema, development_summary
        if not (ROOT/"results"/"reference"/"transmission_extrema.json").exists():
            reference_scan()
        complete_reference_raw_outputs()
        refine_transmission_extrema()
        conditional_campaign()
        retraining_scan()
        sensitivity_campaign()
        summarize_scan_costs()
        development_summary()
    if stage in ("figures", "all"):
        from .figures import make_figures
        make_figures()
    if stage in ("report", "all"):
        from .report import build_report
        build_report()
    if stage in ("audit", "all"):
        from .audit import audit
        audit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "reference", "fixed", "baseline", "scan", "figures", "report", "audit"), default="all")
    run(parser.parse_args().stage)
