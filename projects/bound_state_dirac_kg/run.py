"""Reproducible serial entry point; resumes complete, matching runs."""
from __future__ import annotations
import argparse
from .physics import config


def main(stage):
    if stage in ["all","reference"]:
        from .reference import main as reference
        reference()
    if stage in ["all","train"]:
        from .train import train
        for seed in config()["seeds"]:
            for kind in ["dirac","kg"]: train(kind,seed)
    if stage in ["all","evaluate"]:
        from .evaluate import main as evaluate
        evaluate()
    if stage in ["all","sensitivity"]:
        from .train import train
        from .evaluate import main as evaluate
        for tag,overrides in [("grid_refined",dict(collocation_points=768,quadrature_points=512)),
                              ("domain_extended",dict(half_length=24.0))]:
            for kind in ["dirac","kg"]: train(kind,0,tag,overrides)
            evaluate(tag)
    if stage in ["all","figures"]:
        from .figures import main as figures
        figures()
    if stage in ["all","report"]:
        from .report import main as report
        report()
    if stage in ["all","audit"]:
        from .audit import main as audit
        audit()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["all","reference","train","evaluate","sensitivity","figures","report","audit"],default="all")
    main(p.parse_args().stage)
