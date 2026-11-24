"""Independent reference solvers. Never imported by model or training."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import minimize_scalar
from .physics import (ROOT, CASE, KINDS, ENERGIES, free_matrix, free_data,
                      ode_matrix, potential, current, relative_l2, phase_error,
                      save_json, energy_name)


@dataclass
class Solution:
    kind: str
    energy: float
    r: complex
    t: complex
    value: object
    seconds: float
    details: dict

    @property
    def R(self):
        return float(abs(self.r)**2)

    @property
    def T(self):
        return float(abs(self.t)**2)

    def summary(self):
        return dict(kind=self.kind, energy=self.energy, R=self.R, T=self.T,
                    r_real=float(self.r.real), r_imag=float(self.r.imag),
                    t_real=float(self.t.real), t_imag=float(self.t.imag),
                    phase_r=float(np.angle(self.r)) if abs(self.r)>1e-5 else None,
                    phase_t=float(np.angle(self.t)), unity_error=abs(self.R+self.T-1),
                    seconds=self.seconds, **self.details)


def jost(kind, energy, case=CASE, length=None, rtol=3e-13, atol=3e-14,
         max_step=0.08, potential_fn=None, segments=None):
    """Integrate the unit right-going Jost column backwards; normalize at left.

    A unit outgoing column at +L has coefficients (a,b) at -L. Left
    incidence with a=1 is obtained by dividing the full solution by a.
    Piecewise-constant square wells use exact segment endpoints, avoiding
    integration through a discontinuous RHS.
    """
    start = time.perf_counter()
    L = case["half_length"] if length is None else length
    mass = case["mass"]
    pot = (lambda x: potential(x, case)) if potential_fn is None else potential_fn
    y = free_matrix(kind, energy, L, mass)[:, 0]
    blocks = []
    intervals = [(L, -L, None)] if segments is None else segments
    evaluations = 0
    for right, left, constant in intervals:
        sol = solve_ivp(lambda x, z: ode_matrix(kind, energy, pot(x) if constant is None else constant, mass) @ z,
                        (right, left), y, method="DOP853", rtol=rtol, atol=atol,
                        max_step=max_step, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        blocks.append((left, right, sol.sol))
        y = sol.y[:, -1]
        evaluations += sol.nfev
    incident, reflected = np.linalg.solve(free_matrix(kind, energy, -L, mass), y)
    if abs(incident) < 1e-12:
        raise FloatingPointError("Cannot normalize the incident channel.")

    def values(x):
        x = np.atleast_1d(x)
        result = np.empty((len(x), 2), complex)
        for left, right, dense in blocks:
            mask = (x >= left-1e-12) & (x <= right+1e-12)
            result[mask] = dense(x[mask]).T/incident
        return result

    return Solution(kind, energy, complex(reflected/incident), complex(1/incident), values,
                    time.perf_counter()-start,
                    dict(method="DOP853 backward Jost", length=L, rtol=rtol, atol=atol,
                         max_step=max_step, nfev=evaluations))


def boundary_value(kind, energy, case=CASE, length=None, tol=2e-11):
    """Real four-state collocation BVP, independent of the Jost trajectory."""
    start = time.perf_counter()
    L = case["half_length"] if length is None else length
    left_inverse = np.linalg.inv(free_matrix(kind, energy, -L, case["mass"]))
    right_inverse = np.linalg.inv(free_matrix(kind, energy, L, case["mass"]))
    x = np.linspace(-L, L, 401)
    initial = np.array([free_matrix(kind, energy, xx, case["mass"])[:, 0] for xx in x]).T

    def fun(xx, yy):
        z = yy[:2]+1j*yy[2:]
        v = potential(xx, case)
        ev = energy-v
        if kind == "dirac":
            dz = np.array([1j*(ev+case["mass"])*z[1], 1j*(ev-case["mass"])*z[0]])
        else:
            dz = np.array([z[1], -(ev**2-case["mass"]**2)*z[0]])
        return np.concatenate([dz.real, dz.imag], axis=0)

    def bc(ya, yb):
        a = (left_inverse @ (ya[:2]+1j*ya[2:]))[0]-1
        b = (right_inverse @ (yb[:2]+1j*yb[2:]))[1]
        return np.array([a.real, b.real, a.imag, b.imag])

    sol = solve_bvp(fun, bc, x, np.concatenate([initial.real, initial.imag]),
                    tol=tol, bc_tol=tol, max_nodes=80000)
    if not sol.success:
        raise RuntimeError(f"{kind} E={energy}: {sol.message}")

    def values(xx):
        z = sol.sol(np.atleast_1d(xx))
        return (z[:2]+1j*z[2:]).T

    r = (left_inverse @ values([-L])[0])[1]
    t = (right_inverse @ values([L])[0])[0]
    return Solution(kind, energy, complex(r), complex(t), values, time.perf_counter()-start,
                    dict(method="real collocation BVP", length=L, tolerance=tol,
                         nodes=len(sol.x), iterations=sol.niter,
                         max_rms_residual=float(sol.rms_residuals.max())))


def square_analytic(kind, energy, depth=0.5, half_width=2., mass=1.):
    """Closed-form amplitude matching for V=-depth on [-b,b]."""
    k = np.sqrt(energy**2-mass**2)
    q = np.sqrt((energy+depth)**2-mass**2)
    eta = q/k if kind == "kg" else (q/(energy+depth+mass))/(k/(energy+mass))
    s, c = np.sin(2*q*half_width), np.cos(2*q*half_width)
    denominator = c-0.5j*(eta+1/eta)*s
    phase = np.exp(-2j*k*half_width)
    t = phase/denominator
    r = 0.5j*(eta-1/eta)*s*phase/denominator
    return complex(r), complex(t)


def square_jost(kind, energy, depth=0.5, half_width=2., length=8.):
    b = half_width
    return jost(kind, energy, length=length,
                segments=[(length, b, 0.), (b, -b, -depth), (-b, -length, 0.)])


def save_wave(solution, x, path):
    y = solution.value(x)
    np.savez_compressed(path, x=x, state=y, current=current(solution.kind, y),
                        r=solution.r, t=solution.t, energy=solution.energy)


def reference_campaign():
    import pandas as pd
    out = ROOT/"results"/"reference"
    out.mkdir(parents=True, exist_ok=True)
    convergence, tests, fixed = [], [], []
    for kind in KINDS:
        for energy in ENERGIES:
            free = jost(kind, energy, potential_fn=lambda x: 0.)
            square = square_jost(kind, energy)
            rr, tt = square_analytic(kind, energy)
            tests.append(dict(kind=kind, energy=energy,
                              free_r_error=abs(free.r), free_t_error=abs(free.t-1),
                              square_r_error=abs(square.r-rr), square_t_error=abs(square.t-tt)))
            main = jost(kind, energy)
            bvp = boundary_value(kind, energy)
            x = np.linspace(-CASE["half_length"], CASE["half_length"], 4001)
            save_wave(main, x, out/f"{kind}_{energy_name(energy)}.npz")
            save_wave(bvp, x, out/f"{kind}_{energy_name(energy)}_bvp.npz")
            fixed.append(main.summary())
            variants = [("bvp", bvp)]
            for length in (10., 12., 16., 20.):
                variants.append((f"domain_{length:g}", jost(kind, energy, length=length)))
            for tolerance in (1e-8, 1e-10, 3e-13):
                variants.append((f"tolerance_{tolerance:g}", jost(kind, energy, rtol=tolerance,
                                 atol=tolerance/10, max_step=0.35)))
            for label, alt in variants:
                common_L = min(CASE["half_length"], alt.details["length"])
                for n in (1001, 2001, 4001):
                    xx = np.linspace(-common_L, common_L, n)
                    y, other = main.value(xx), alt.value(xx)
                    delta = relative_l2(xx, other, y)
                    convergence.append(dict(kind=kind, energy=energy, variant=label,
                        grid_points=n, common_length=common_L, wave_l2_difference=delta,
                        r_difference=abs(alt.r-main.r), t_difference=abs(alt.t-main.t),
                        R_difference=abs(alt.R-main.R), T_difference=abs(alt.T-main.T),
                        phase_t_difference=phase_error(alt.t, main.t),
                        flux_difference=float(np.max(abs(current(kind, other)-current(kind, y)))),
                        unity_error=abs(alt.R+alt.T-1), **{k:v for k,v in alt.details.items() if k!='length'}))
                if label in ("bvp", "domain_20", "tolerance_3e-13"):
                    save_wave(alt, np.linspace(-common_L, common_L, 4001),
                              out/f"{kind}_{energy_name(energy)}_{label}.npz")
            print(f"REFERENCE {kind} E={energy} R={main.R:.10g} T={main.T:.10g} BVP-dt={abs(main.t-bvp.t):.2e}", flush=True)
    pd.DataFrame(tests).to_csv(out/"analytic_tests.csv", index=False)
    pd.DataFrame(convergence).to_csv(out/"convergence.csv", index=False)
    pd.DataFrame(fixed).to_csv(out/"fixed.csv", index=False)
    cv = pd.DataFrame(convergence)
    fine = cv[(cv.variant.isin(["bvp", "domain_20", "tolerance_3e-13"])) & (cv.grid_points==4001)]
    verified = bool(fine.wave_l2_difference.max()<2e-9 and fine.t_difference.max()<2e-9
                    and max(t["free_t_error"] for t in tests)<1e-10
                    and max(t["square_t_error"] for t in tests)<1e-10)
    save_json(out/"verification.json", dict(verified=verified, case=CASE,
              maximum_wave_difference=float(fine.wave_l2_difference.max()),
              maximum_amplitude_difference=float(max(fine.t_difference.max(), fine.r_difference.max())),
              statement="Empirical convergence evidence, not a rigorous bound; domain truncation and test-grid checks included."))
    if not verified:
        raise RuntimeError("Reference verification failed; do not proceed to PINN.")


def reference_scan(energies=None):
    import pandas as pd
    out = ROOT/"results"/"reference"
    energies = np.linspace(1.01, 2., 101) if energies is None else np.asarray(energies)
    records = []
    for kind in KINDS:
        ys, flux = [], []
        x = np.linspace(-16, 16, 1601)
        for energy in energies:
            sol = jost(kind, float(energy))
            summary = sol.summary()
            start_inference = time.perf_counter()
            state = sol.value(x)
            summary["inference_seconds"] = time.perf_counter()-start_inference
            records.append(summary)
            ys.append(state)
            flux.append(current(kind, state))
        np.savez_compressed(out/f"{kind}_scan.npz", energies=energies, x=x,
                            state=np.asarray(ys), current=np.asarray(flux))
    pd.DataFrame(records).to_csv(out/"scan.csv", index=False)
    # Minima of R are determined from a denser reference scan, then refined
    # continuously. This does not compute complex-energy poles or bound states.
    extrema, dense_records = [], []
    for kind in KINDS:
        dense = np.linspace(1.01, 2., 1001)
        R = []
        for e in dense:
            sol = jost(kind, float(e), max_step=0.15)
            dense_records.append(sol.summary())
            R.append(sol.R)
        R = np.array(R)
        for i in range(1, len(dense)-1):
            if R[i]<R[i-1] and R[i]<R[i+1]:
                result = minimize_scalar(lambda e: jost(kind, float(e)).R,
                         bounds=(dense[i-1], dense[i+1]), method="bounded",
                         options=dict(xatol=1e-13))
                sol = jost(kind, float(result.x))
                check = jost(kind, float(result.x), length=20., rtol=1e-13, atol=1e-14, max_step=.04)
                extrema.append(dict(**sol.summary(), left_energy=float(dense[i-1]),
                           right_energy=float(dense[i+1]), left_R=float(R[i-1]), right_R=float(R[i+1]),
                           crosscheck_t_difference=abs(sol.t-check.t), crosscheck_R_difference=abs(sol.R-check.R),
                           interpretation="Real-energy transmission maximum; no complex-energy resonance claim."))
    pd.DataFrame(dense_records).to_csv(out/"dense_scan.csv", index=False)
    save_json(out/"transmission_extrema.json", extrema)


if __name__ == "__main__":
    reference_campaign()
