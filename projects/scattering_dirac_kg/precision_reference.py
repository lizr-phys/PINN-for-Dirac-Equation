"""Arbitrary-precision Taylor Jost amplitudes, independent of SciPy integration.

Gaussian Taylor coefficients are computed analytically, then both original
first-order systems are advanced using power-series recurrence. Step/degree/
precision/domain convergence is tested; amplitudes are stored as decimal
strings as well as complex128 for downstream error evaluation.
"""
from __future__ import annotations
import time
import mpmath as mp
import numpy as np
import pandas as pd
from .physics import ROOT, KINDS, ENERGIES, save_json, read_json


def taylor_jost(kind, energy, length=20., step=.4, degree=32, digits=60,
                depth=.5, width=2.):
    start = time.perf_counter()
    with mp.workdps(digits):
        # Exactly the represented input energy, not a rounded decimal change.
        e, L = mp.mpf(float(energy)), mp.mpf(float(length))
        mass, depth, width = mp.mpf(1), mp.mpf(float(depth)), mp.mpf(float(width))
        k = mp.sqrt(e*e-mass*mass)
        C = mp.sqrt((e+mass)/(2*k)) if kind=="dirac" else 1/mp.sqrt(2*k)
        ratio = k/(e+mass) if kind=="dirac" else mp.j*k
        f, g = C*mp.exp(mp.j*k*L), C*ratio*mp.exp(mp.j*k*L)
        blocks = int(mp.ceil(2*L/mp.mpf(step)))
        h = -2*L/blocks
        x = L
        for _ in range(blocks):
            v = [-depth*mp.exp(-x*x/(2*width*width))]
            b, two_c = -x/(width*width), -1/(width*width)
            for n in range(1, degree):
                v.append((b*v[n-1]+(two_c*v[n-2] if n>1 else 0))/n)
            if kind=="kg":
                q = [(e*e-mass*mass if n==0 else 0)-2*e*v[n]
                     +sum(v[j]*v[n-j] for j in range(n+1)) for n in range(degree)]
            fs, gs = [f], [g]
            for n in range(degree):
                if kind=="dirac":
                    fs.append(mp.j*((e+mass)*gs[n]-sum(v[j]*gs[n-j] for j in range(n+1)))/(n+1))
                    gs.append(mp.j*((e-mass)*fs[n]-sum(v[j]*fs[n-j] for j in range(n+1)))/(n+1))
                else:
                    fs.append(gs[n]/(n+1))
                    gs.append(-sum(q[j]*fs[n-j] for j in range(n+1))/(n+1))
            f, g = mp.polyval(list(reversed(fs)), h), mp.polyval(list(reversed(gs)), h)
            x += h
        incident = mp.exp(mp.j*k*L)*(f+g/ratio)/(2*C)
        reflected = mp.exp(-mp.j*k*L)*(f-g/ratio)/(2*C)
        r, t = reflected/incident, 1/incident
        record = dict(kind=kind, energy=energy, length=length, step=float(abs(h)),
                      degree=degree, digits=digits, seconds=time.perf_counter()-start,
                      r_real=mp.nstr(r.real, digits), r_imag=mp.nstr(r.imag, digits),
                      t_real=mp.nstr(t.real, digits), t_imag=mp.nstr(t.imag, digits),
                      R=mp.nstr(abs(r)**2, digits), T=mp.nstr(abs(t)**2, digits),
                      unity_error=mp.nstr(abs(abs(r)**2+abs(t)**2-1), digits))
        return record


def mp_amplitude(record, name):
    return mp.mpc(record[name+"_real"], record[name+"_imag"])


def high_precision_campaign():
    out = ROOT/"results"/"reference"/"high_precision"
    out.mkdir(exist_ok=True)
    main_records, convergence = [], []
    for kind in KINDS:
        for e in ENERGIES:
            main = taylor_jost(kind, e, length=20., step=.4, degree=32, digits=60)
            variants = [taylor_jost(kind, e, length=20., step=.25, degree=40, digits=70),
                        taylor_jost(kind, e, length=24., step=.25, degree=40, digits=70)]
            with mp.workdps(70):
                for alt in variants:
                    dr = abs(mp_amplitude(main, "r")-mp_amplitude(alt, "r"))
                    dt = abs(mp_amplitude(main, "t")-mp_amplitude(alt, "t"))
                    convergence.append(dict(kind=kind, energy=e, length=alt["length"],
                         step=alt["step"], degree=alt["degree"], digits=alt["digits"],
                         r_difference=float(dr), t_difference=float(dt), seconds=alt["seconds"]))
            main_records.append(main)
            save_json(out/f"{kind}_{e:g}.json", dict(main=main, variants=variants))
            print(f"HIGH PRECISION {kind} E={e:g} dt={convergence[-1]['t_difference']:.2e} ({main['seconds']:.1f}s)", flush=True)
    save_json(out/"amplitudes.json", main_records)
    pd.DataFrame(convergence).to_csv(out/"convergence.csv", index=False)
    verified = max(max(row["r_difference"], row["t_difference"]) for row in convergence)<1e-20
    save_json(out/"verification.json", dict(verified=verified,
        maximum_amplitude_difference=max(max(row["r_difference"], row["t_difference"]) for row in convergence),
        interpretation="Empirical arbitrary-precision Taylor convergence; domains 20 and 24, degrees 32/40, steps .4/.25, 60/70 digits."))
    if not verified:
        raise RuntimeError("High precision reference did not converge sufficiently")


def refined_amplitudes(kind, energy):
    path = ROOT/"results"/"reference"/"high_precision"/"amplitudes.json"
    if not path.exists():
        return None
    for record in read_json(path):
        if record["kind"]==kind and abs(record["energy"]-energy)<1e-14:
            return (complex(float(record["r_real"]), float(record["r_imag"])),
                    complex(float(record["t_real"]), float(record["t_imag"])))
    return None


if __name__ == "__main__":
    high_precision_campaign()
