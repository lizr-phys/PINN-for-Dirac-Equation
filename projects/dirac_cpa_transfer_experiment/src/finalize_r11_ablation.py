from __future__ import annotations
import argparse
import json
import time
from datetime import datetime, timezone
from .r11_config import RAW
from .freeze_r7_phase_release import write_json

def finalize(wait):
    from .r7_runtime import set_cpu_affinity
    set_cpu_affinity(128)
    status_path=RAW/"postprocessing_status.json"
    while True:
        plan=json.loads((RAW/"runtime_plan.json").read_text(encoding="utf-8"))
        if plan["status"]=="COMPLETE": break
        if plan["status"]=="FAIL": raise RuntimeError("Formal failures retained; automatic finalization stopped")
        if not wait: raise RuntimeError("Formal training not complete")
        time.sleep(30)
    from .train_wavepacket_shared_spectral_ltpb import _set_runtime
    _set_runtime(0,1)
    from .aggregate_r11_ablation import aggregate
    from .audit_r11_ablation import audit
    from .plot_r11_ablation import make_figures
    from .qa_r11_ablation_figures import prepare_visual_qa
    for stage,action in (("aggregate",aggregate),("data_audit",audit),
                         ("plot",make_figures),("independent_pdf_preview",prepare_visual_qa)):
        write_json(status_path,dict(stage=stage,status="RUNNING",utc=datetime.now(timezone.utc).isoformat()))
        print("POSTPROCESS "+stage,flush=True)
        result=action()
        if stage=="data_audit" and result["status"]!="PASS":
            raise RuntimeError("Data audit failed: "+str(result["failed_checks"]))
    write_json(status_path,dict(stage="manual_visual_inspection",status="AWAITING_INSPECTION",
        utc=datetime.now(timezone.utc).isoformat(),next="inspect every PNG/PDF; record QA; build report; full pytest; final audit"))
    print("Plots and PDF previews ready for manual inspection",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--wait",action="store_true")
    args=parser.parse_args();finalize(args.wait)
