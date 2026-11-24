"""Physics-only linear output-layer training; no reference imports or labels."""
from __future__ import annotations
import argparse
import platform
import sys
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import lstsq
from .physics import ROOT, CASE, energy_name, save_json, read_json, source_hashes
from .model import ScatteringNetwork


def quadrature(points, length):
    z, w = np.polynomial.legendre.leggauss(points)
    return length*z, w/2  # weights integrate the spatial mean, not L2 normalization


def training_energies(nodes):
    kmin, kmax = np.sqrt(1.01**2-1), np.sqrt(2.**2-1)
    k = (kmax+kmin)/2+(kmax-kmin)/2*np.cos(np.arange(nodes)*np.pi/(nodes-1))
    return np.sqrt(1+k*k)[::-1]


def train(kind, energy=1.02, seed=0, group="pilot", features=64,
          collocation=512, degree=0, energy_nodes=1, rcond=1e-12,
          refinements=3, case=None, width_min=.8, width_max=2.8):
    import pandas as pd
    case = CASE.copy() if case is None else case.copy()
    conditional = degree > 0
    out = ROOT/"results"/group/kind/("conditional" if conditional else energy_name(energy))/f"seed_{seed}"
    cfg = dict(kind=kind, energy=None if conditional else energy, seed=seed, group=group,
               features=features, collocation=collocation, degree=degree, energy_nodes=energy_nodes,
               rcond=rcond, refinements=refinements, case=case, width_min=width_min, width_max=width_max,
               source_sha256=source_hashes(), flux_loss=False, reference_labels=False,
               optimizer="column-scaled complex SVD least squares with residual refinement",
               hidden_layer="fixed seeded Gaussian RBF; two known free-wave carriers",
               derivative="analytic neural derivatives checked by independent torch AD")
    if (out/"training.json").exists():
        if read_json(out/"config.json") != cfg:
            raise RuntimeError(f"Configuration/source changed; choose a new run group: {out}")
        if not (out/"model.npz").exists():
            raise RuntimeError(f"Incomplete completed run: {out}")
        return out
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Incomplete run retained; choose a new group: {out}")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/"config.json", cfg)
    save_json(out/"environment.json", dict(python=sys.version, executable=sys.executable,
                numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                threads=1, dtype="float64/complex128"))
    net = ScatteringNetwork(kind, seed, features, case, degree, width_min, width_max)
    history = []
    start = time.perf_counter()
    x, w = quadrature(collocation, case["half_length"])
    energies = training_energies(energy_nodes) if conditional else np.array([energy])
    np.savez_compressed(out/"training_points.npz", x=x, mean_weights=w, energies=energies)
    try:
        mats, offsets = [], []
        for e in energies:
            base, basis = net.residual_basis(x, float(e))
            ep = net.energy_features(float(e))[0]
            mat = (basis[:, :, None]*ep[None, None, :]).reshape(len(x), -1)
            weights = np.sqrt(w/len(energies))
            mats.append(weights[:, None]*mat)
            offsets.append(weights*base)
        matrix, offset = np.concatenate(mats), np.concatenate(offsets)
        scales = np.maximum(np.linalg.norm(matrix, axis=0), 1e-14)
        scaled = matrix/scales[None, :]
        cost0 = float(np.vdot(offset, offset).real)
        history.append(dict(step=0, phase="initial_unit_incident", physics_loss=cost0,
                            seconds=time.perf_counter()-start))
        solution, _, rank, singular = lstsq(scaled, -offset, cond=rcond, lapack_driver="gelsd")
        coefficients = solution/scales
        residual = matrix@coefficients+offset
        cost = float(np.vdot(residual, residual).real)
        history.append(dict(step=1, phase="SVD_output_layer", physics_loss=cost,
                            seconds=time.perf_counter()-start))
        # Re-solve the same scaled linear objective for iterative refinement.
        # Each trial must decrease the equation objective; no labels are used.
        for i in range(refinements):
            correction, _, _, _ = lstsq(scaled, -residual, cond=rcond, lapack_driver="gelsd")
            trial = coefficients+correction/scales
            next_residual = matrix@trial+offset
            next_cost = float(np.vdot(next_residual, next_residual).real)
            accepted = next_cost < cost
            if accepted:
                coefficients, residual, cost = trial, next_residual, next_cost
            history.append(dict(step=i+2, phase="linear_residual_refinement", physics_loss=cost,
                                accepted=accepted, seconds=time.perf_counter()-start))
            if not accepted:
                break
        if not np.isfinite(coefficients).all() or not np.isfinite(cost):
            raise FloatingPointError("Nonfinite linear training result.")
        net.coefficients = coefficients.reshape(net.coefficients.shape)
        seconds = time.perf_counter()-start
        net.save(out/"model.npz")
        np.savez_compressed(out/"linear_diagnostics.npz", singular_values=singular,
                            column_scales=scales, weighted_training_residual=residual)
        save_json(out/"training.json", dict(completed=True, seconds=seconds,
            trainable_parameters=net.trainable_parameters, frozen_parameters=net.frozen_parameters,
            total_parameters=net.trainable_parameters+net.frozen_parameters, physics_loss=cost,
            rank=int(rank), complex_unknowns=matrix.shape[1], rows=matrix.shape[0],
            condition_retained=float(singular[0]/singular[int(rank)-1]),
            coefficient_l2=float(np.linalg.norm(coefficients)), stop_reason="direct_solve_budget",
            training_energy_count=len(energies), objective_scale="spatial mean, upper incident amplitude factored out",
            boundary_constraint="hard a_left=1+0j, b_right=0j; no normalization or parity"))
        print(f"TRAIN {group} {kind} {'conditional' if conditional else f'E={energy:g}'} seed={seed} loss={cost:.3e} {seconds:.2f}s", flush=True)
    except Exception as exc:
        save_json(out/"failure.json", dict(exception=type(exc).__name__, message=str(exc),
                                         seconds=time.perf_counter()-start))
        raise
    finally:
        pd.DataFrame(history).to_csv(out/"history.csv", index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("dirac", "kg"), required=True)
    parser.add_argument("--energy", type=float, default=1.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group", default="pilot")
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--collocation", type=int, default=512)
    parser.add_argument("--degree", type=int, default=0)
    parser.add_argument("--energy-nodes", type=int, default=1)
    parser.add_argument("--refinements", type=int, default=3)
    args = parser.parse_args()
    train(args.kind, args.energy, args.seed, args.group, args.features, args.collocation,
          args.degree, args.energy_nodes, refinements=args.refinements)
