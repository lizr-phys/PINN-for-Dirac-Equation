"""Post-training evaluation with fixed incident phase and independent grids."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from .physics import (ROOT, CASE, free_matrix, current, relative_l2, phase_error,
                      energy_name, save_json, read_json)
from .model import ScatteringNetwork
from .reference import jost
from .precision_reference import refined_amplitudes


def evaluate_arrays(kind, energy, x, state, residual, reference_state,
                    r, t, reference_r, reference_t, boundary_state, length=16.):
    left = np.linalg.solve(free_matrix(kind, energy, -length), boundary_state[0])
    right = np.linalg.solve(free_matrix(kind, energy, length), boundary_state[1])
    flow = current(kind, state)
    R, T = float(abs(r)**2), float(abs(t)**2)
    ref_R, ref_T = float(abs(reference_r)**2), float(abs(reference_t)**2)
    wave = state if kind=="dirac" else state[:, :1]
    ref_wave = reference_state if kind=="dirac" else reference_state[:, :1]
    residual_rms = float(np.sqrt(np.trapezoid(np.sum(abs(residual)**2, axis=1), x)/(x[-1]-x[0])))
    boundary = float(max(abs(left[0]-1), abs(right[1])))
    metrics = dict(kind=kind, energy=energy, R=R, T=T, reference_R=ref_R, reference_T=ref_T,
        R_absolute_error=abs(R-ref_R), T_absolute_error=abs(T-ref_T),
        r_real=float(r.real), r_imag=float(r.imag), t_real=float(t.real), t_imag=float(t.imag),
        r_complex_error=float(abs(r-reference_r)), t_complex_error=float(abs(t-reference_t)),
        r_phase_error=phase_error(r, reference_r), t_phase_error=phase_error(t, reference_t),
        phase_r=float(np.angle(r)) if abs(r)>1e-5 else None, phase_t=float(np.angle(t)),
        wave_relative_l2=relative_l2(x, wave, ref_wave),
        component_0_relative_l2=relative_l2(x, state[:, 0], reference_state[:, 0]),
        component_1_relative_l2=relative_l2(x, state[:, 1], reference_state[:, 1]),
        residual_rms=residual_rms, residual_max=float(abs(residual).max()),
        unity_error=abs(R+T-1), current_span=float(np.ptp(flow)),
        current_transmission_deviation=float(np.max(abs(flow-T))), boundary_error=boundary,
        incident_real=float(left[0].real), incident_imag=float(left[0].imag),
        right_incoming_real=float(right[1].real), right_incoming_imag=float(right[1].imag),
        amplitude_extraction_disagreement=float(max(abs(left[1]-r), abs(right[0]-t))),
        test_points=len(x), finite=bool(np.isfinite(state).all() and np.isfinite(residual).all()),
        phase_alignment="fixed unit incident coefficient; no post-hoc phase or normalization")
    metrics["passed"] = bool(metrics["finite"] and metrics["wave_relative_l2"]<5e-4
        and metrics["R_absolute_error"]<5e-4 and metrics["T_absolute_error"]<5e-4
        and metrics["r_complex_error"]<5e-4 and metrics["t_complex_error"]<5e-4
        and metrics["residual_rms"]<5e-4 and metrics["unity_error"]<5e-4
        and metrics["current_span"]<5e-4 and boundary<1e-9)
    return metrics


def evaluate_run(directory, energy=None, reference=None, points=4001, save=True):
    directory = Path(directory)
    cfg, training = read_json(directory/"config.json"), read_json(directory/"training.json")
    energy = cfg["energy"] if energy is None else energy
    net = ScatteringNetwork.load(directory/"model.npz")
    length = cfg["case"]["half_length"]
    x = np.linspace(-length, length, points)
    if reference is None:
        reference = jost(cfg["kind"], energy, length=max(length, 20.))
    precise = refined_amplitudes(cfg["kind"], energy)
    if precise is not None:
        reference.r, reference.t = precise
    yref = reference.value(x)
    start = time.perf_counter()
    state, residual = net.state_and_residual(x, energy)
    infer_seconds = time.perf_counter()-start
    r, t = net.amplitudes(energy)
    boundary_state = net.state_and_residual(np.array([-length, length]), energy)[0]
    metrics = evaluate_arrays(cfg["kind"], energy, x, state, residual, yref, r, t,
                    reference.r, reference.t, boundary_state, length)
    metrics.update(seed=cfg["seed"], group=cfg["group"], conditional=cfg["degree"]>0,
                    trainable_parameters=training["trainable_parameters"],
                    frozen_parameters=training["frozen_parameters"],
                    total_parameters=training["total_parameters"],
                    training_seconds=training["seconds"], inference_seconds=infer_seconds,
                    training_loss=training["physics_loss"], degree=cfg["degree"], features=cfg["features"])
    if save:
        prefix = energy_name(energy)+"_" if cfg["degree"]>0 else ""
        save_json(directory/f"{prefix}metrics.json", metrics)
        v = -.5*np.exp(-x*x/8)
        rho = np.sum(abs(state)**2, axis=1) if cfg["kind"]=="dirac" else 2*(energy-v)*abs(state[:, 0])**2
        np.savez_compressed(directory/f"{prefix}evaluation.npz", x=x, state=state, reference=yref,
                  residual=residual, current=current(cfg["kind"], state), density=rho,
                  error=state-yref, r=r, t=t, reference_r=reference.r, reference_t=reference.t)
    return metrics


def aggregate(records, path):
    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    numerical = ["R_absolute_error", "T_absolute_error", "r_complex_error", "t_complex_error",
                 "r_phase_error", "t_phase_error", "wave_relative_l2", "residual_rms",
                 "unity_error", "current_span", "boundary_error", "training_seconds"]
    result = df.groupby(["kind", "energy"])[numerical].agg(["mean", "std", "min", "max"])
    result.columns = ["_".join(c) for c in result.columns]
    result.to_csv(Path(path).with_name(Path(path).stem+"_aggregate.csv"))
    return df
