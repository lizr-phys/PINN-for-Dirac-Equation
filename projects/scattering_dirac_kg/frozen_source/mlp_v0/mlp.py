"""Conventional all-weight Adam/L-BFGS PINN baseline, physics-only training.

Total complex fields; soft channel boundary constraints; no parity, L2
normalization, reference wave, scattering coefficients, or phase labels.
"""
from __future__ import annotations
import hashlib
import time
import sys
import numpy as np
import torch
from torch import nn
from .physics import ROOT, CASE, free_matrix, energy_name, read_json, save_json


def grad(y, x):
    if torch.is_complex(y):
        return grad(y.real, x)+1j*grad(y.imag, x)
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


class MLP(nn.Module):
    def __init__(self, kind, energy, width=32):
        super().__init__()
        self.kind, self.energy = kind, energy
        self.k = np.sqrt(energy**2-1)
        outputs = 4 if kind=="dirac" else 2
        self.net = nn.Sequential(nn.Linear(4, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs))
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        z = torch.cat([x/16, torch.exp(-x*x/8), torch.sin(self.k*x), torch.cos(self.k*x)], dim=1)
        out = self.net(z)
        f = out[:, :1]+1j*out[:, 1:2]
        if self.kind=="dirac":
            return torch.cat([f, out[:, 2:3]+1j*out[:, 3:4]], dim=1)
        return f

    def state_residual(self, x):
        wave = self(x)
        v = -.5*torch.exp(-x*x/8)
        a, b = self.energy-v+1, self.energy-v-1
        df = grad(wave[:, :1], x)
        if self.kind=="dirac":
            dg = grad(wave[:, 1:2], x)
            return wave, torch.cat([-1j*dg-b*wave[:, :1], -1j*df-a*wave[:, 1:2]], dim=1)
        return torch.cat([wave, df], dim=1), grad(df, x)+a*b*wave

    def boundary(self):
        x = torch.tensor([[-16.], [16.]], requires_grad=True)
        state, _ = self.state_residual(x)
        left = torch.linalg.solve(torch.tensor(free_matrix(self.kind, self.energy, -16.)), state[0])
        right = torch.linalg.solve(torch.tensor(free_matrix(self.kind, self.energy, 16.)), state[1])
        return torch.stack([left[0]-1, right[1]]), left, right


def train_mlp(kind, energy=1.02, seed=0, group="mlp_baseline", adam_steps=500,
              lbfgs_steps=250, width=32, points=256, boundary_weight=10.):
    import pandas as pd
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    out = ROOT/"results"/group/kind/energy_name(energy)/f"seed_{seed}"
    cfg = dict(kind=kind, energy=energy, seed=seed, group=group, adam_steps=adam_steps,
              lbfgs_steps=lbfgs_steps, width=width, collocation=points, boundary_weight=boundary_weight,
              source_sha256=hashlib.sha256((ROOT/"mlp.py").read_bytes()).hexdigest(),
              reference_labels=False, flux_loss=False, representation="total complex field; soft boundary")
    if (out/"training.json").exists():
        if read_json(out/"config.json")!=cfg:
            raise RuntimeError("Baseline config changed: choose new group")
        return out
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Incomplete baseline retained: choose new group")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/"config.json", cfg)
    save_json(out/"environment.json", dict(python=sys.version, torch=torch.__version__, dtype="float64", threads=1))
    model = MLP(kind, energy, width)
    z, w = np.polynomial.legendre.leggauss(points)
    x = torch.tensor(16*z[:, None], requires_grad=True)
    weights = torch.tensor(w[:, None]/2)
    np.savez_compressed(out/"training_points.npz", x=x.detach().numpy(), mean_weights=weights.numpy())
    history = []
    calls, best, best_state = 0, float("inf"), None
    start = time.perf_counter()

    def objective():
        _, residual = model.state_residual(x)
        equation = torch.sum(weights*abs(residual)**2)
        bc, _, _ = model.boundary()
        boundary = torch.sum(abs(bc)**2)
        return equation+boundary_weight*boundary, equation, boundary

    def record(phase, force=False):
        nonlocal calls, best, best_state
        loss, equation, boundary = objective()
        calls += 1
        cost = float(loss.detach())
        if np.isfinite(cost) and cost<best:
            best = cost
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        if calls%10==0 or force:
            history.append(dict(step=calls, phase=phase, loss=cost,
                     physics_loss=float(equation.detach()), boundary_loss=float(boundary.detach()),
                     seconds=time.perf_counter()-start))
        return loss

    try:
        record("initial", True)
        adam = torch.optim.Adam(model.parameters(), lr=.002)
        for _ in range(adam_steps):
            adam.zero_grad()
            loss = record("Adam")
            loss.backward()
            adam.step()
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=lbfgs_steps,
                  max_eval=lbfgs_steps*2, history_size=60, tolerance_grad=1e-11,
                  tolerance_change=1e-13, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            loss = record("LBFGS")
            (1000*loss).backward()
            return 1000*loss
        optimizer.step(closure)
        record("final", True)
        seconds = time.perf_counter()-start
        if best_state is None:
            raise FloatingPointError("No finite baseline checkpoint")
        model.load_state_dict(best_state)
        loss, equation, boundary = objective()
        torch.save(model.state_dict(), out/"model.pt")
        save_json(out/"training.json", dict(completed=True, seconds=seconds, objective=float(loss.detach()),
                physics_loss=float(equation.detach()), boundary_loss=float(boundary.detach()),
                parameters=sum(p.numel() for p in model.parameters()), calls=calls,
                stop_reason="Adam/LBFGS iteration budget; minimum finite physics objective checkpoint"))
        print(f"MLP {kind} E={energy:g} seed={seed} objective={float(loss.detach()):.3e} {seconds:.2f}s", flush=True)
    except Exception as exc:
        save_json(out/"failure.json", dict(exception=type(exc).__name__, message=str(exc), seconds=time.perf_counter()-start))
        raise
    finally:
        pd.DataFrame(history).to_csv(out/"history.csv", index=False)
    return out


def evaluate_mlp(directory):
    from .reference import jost
    from .evaluate import evaluate_arrays
    from .physics import current
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    cfg = read_json(directory/"config.json")
    model = MLP(cfg["kind"], cfg["energy"], cfg["width"])
    model.load_state_dict(torch.load(directory/"model.pt", weights_only=True))
    xx = np.linspace(-16, 16, 4001)
    x = torch.tensor(xx[:, None], requires_grad=True)
    start = time.perf_counter()
    state, residual = model.state_residual(x)
    y, res = state.detach().numpy(), residual.detach().numpy()
    inference_seconds = time.perf_counter()-start
    _, left, right = model.boundary()
    r, t = complex(left[1].detach()), complex(right[0].detach())
    bs = model.state_residual(torch.tensor([[-16.], [16.]], requires_grad=True))[0].detach().numpy()
    ref = jost(cfg["kind"], cfg["energy"], length=20.)
    yref = ref.value(xx)
    metrics = evaluate_arrays(cfg["kind"], cfg["energy"], xx, y, res, yref, r, t, ref.r, ref.t, bs)
    training = read_json(directory/"training.json")
    metrics.update(seed=cfg["seed"], group=cfg["group"], training_seconds=training["seconds"],
                   inference_seconds=inference_seconds, total_parameters=training["parameters"],
                   trainable_parameters=training["parameters"], frozen_parameters=0,
                   training_loss=training["physics_loss"])
    save_json(directory/"metrics.json", metrics)
    np.savez_compressed(directory/"evaluation.npz", x=xx, state=y, residual=res, reference=yref,
                        current=current(cfg["kind"], y), error=y-yref, r=r, t=t, reference_r=ref.r, reference_t=ref.t)
    return metrics
