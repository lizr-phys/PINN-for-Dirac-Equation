"""Additional reference evidence and raw convergence outputs."""
from __future__ import annotations
import shutil
import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from .physics import ROOT, CASE, KINDS, ENERGIES, read_json, save_json, phase_error, energy_name
from .reference import jost, save_wave
from .precision_reference import taylor_jost, mp_amplitude


def complete_reference_raw_outputs():
    out = ROOT/"results"/"reference"
    csv = out/"convergence.csv"
    if not (out/"convergence_initial.csv").exists():
        shutil.copy2(csv, out/"convergence_initial.csv")
    frame = pd.read_csv(csv)
    if "phase_r_difference" in frame.columns:
        return
    frame["phase_r_difference"] = np.nan
    for kind in KINDS:
        for e in ENERGIES:
            name = f"{kind}_{energy_name(e)}"
            main = np.load(out/f"{name}.npz")
            main_r = complex(main["r"])
            for variant in frame[(frame.kind==kind)&(frame.energy==e)].variant.unique():
                rows = (frame.kind==kind)&(frame.energy==e)&(frame.variant==variant)
                if variant=="bvp":
                    data = np.load(out/f"{name}_bvp.npz")
                    rr = complex(data["r"])
                else:
                    if variant.startswith("domain_"):
                        length = float(variant.split("_")[1])
                        solution = jost(kind, e, length=length)
                    else:
                        tolerance = float(variant.split("_")[1])
                        solution = jost(kind, e, rtol=tolerance, atol=tolerance/10, max_step=.35)
                        length = 16.
                    rr = solution.r
                    save_wave(solution, np.linspace(-length,length,4001), out/f"{name}_{variant}.npz")
                    save_json(out/f"{name}_{variant}.json", solution.summary())
                frame.loc[rows,"phase_r_difference"] = phase_error(rr, main_r)
    frame.to_csv(csv, index=False)


def refine_transmission_extrema():
    """Refine confirmed real-energy maxima using symmetry, not complex poles.

    For the even real potential r/t is imaginary. Its sign changes identify
    real reflection zeros; this reference symmetry is not imposed on PINNs.
    """
    out = ROOT/"results"/"reference"
    path = out/"transmission_extrema.json"
    current = read_json(path)
    if current and current[0].get("refinement_version")==2:
        return
    if not (out/"transmission_extrema_initial.json").exists():
        shutil.copy2(path, out/"transmission_extrema_initial.json")
    elif current and current[0].get("high_precision_refined") and not (out/"transmission_extrema_hp_v1.json").exists():
        shutil.copy2(path, out/"transmission_extrema_hp_v1.json")
    initial = read_json(out/"transmission_extrema_initial.json")
    results = []
    for item in initial:
        started = time.perf_counter()
        kind = item["kind"]
        history = []
        def mismatch(e):
            sol = jost(kind,float(e))
            history.append(sol.summary())
            return float((sol.r/sol.t).imag)
        energy = brentq(mismatch,item["left_energy"],item["right_energy"],xtol=5e-15)
        # A high-precision secant slope removes the double-integrator error.
        first = taylor_jost(kind,energy)
        offset = 1e-6
        second = taylor_jost(kind,energy+offset)
        import mpmath as mp
        with mp.workdps(60):
            f0 = (mp_amplitude(first,"r")/mp_amplitude(first,"t")).imag
            f1 = (mp_amplitude(second,"r")/mp_amplitude(second,"t")).imag
            energy = float(mp.mpf(energy)-f0*mp.mpf(offset)/(f1-f0))
        precise = taylor_jost(kind,energy)
        check = taylor_jost(kind,energy,length=24.,step=.25,degree=40,digits=70)
        with mp.workdps(70):
            dr = float(abs(mp_amplitude(precise,"r")-mp_amplitude(check,"r")))
            dt = float(abs(mp_amplitude(precise,"t")-mp_amplitude(check,"t")))
        rr, tt = complex(mp_amplitude(precise,"r")), complex(mp_amplitude(precise,"t"))
        if max(dr,dt)>1e-20:
            raise RuntimeError("High precision extremum verification failed")
        results.append(dict(kind=kind, energy=energy,R=float(abs(rr)**2),T=float(abs(tt)**2),
            r_real=rr.real,r_imag=rr.imag,t_real=tt.real,t_imag=tt.imag,
            phase_r=None, phase_t=float(np.angle(tt)), unity_error=float(precise["unity_error"]),
            method="Real-energy root refinement; arbitrary-precision Taylor Jost amplitudes",
            length=precise["length"], degree=precise["degree"], digits=precise["digits"], step=precise["step"],
            seconds=time.perf_counter()-started, timing_scope="Jost root seed and all four Taylor solves",
            left_energy=item["left_energy"], right_energy=item["right_energy"],
            left_R=item["left_R"], right_R=item["right_R"],
            crosscheck_R_difference=float(abs(float(precise["R"])-float(check["R"]))),
            crosscheck_t_difference=dt,high_precision_refined=True,refinement_version=2,
            initial_reference=item, jost_root_history=history,
            high_precision_seed=first,high_precision_secant_sample=second,
            high_precision=precise,high_precision_check=check,
            interpretation="Real-energy transmission maximum; no complex-energy resonance claim.",
            root_statement="Real reflection zero of even potential; r/t imaginary condition used only in reference refinement, not imposed on PINN."))
    save_json(path, results)


def development_summary():
    rows = []
    for p in (ROOT/"results").rglob("*metrics.json"):
        if not any(part.startswith("pilot_") or part in ("conditional_pilot", "mlp_pilot") for part in p.parts):
            continue
        rows.append(read_json(p))
    if rows:
        pd.DataFrame(rows).to_csv(ROOT/"results"/"development_summary.csv", index=False)


if __name__=="__main__":
    complete_reference_raw_outputs()
    refine_transmission_extrema()
    development_summary()
