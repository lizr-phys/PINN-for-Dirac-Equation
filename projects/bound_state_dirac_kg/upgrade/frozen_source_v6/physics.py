from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def potential(x, case):
    return -case["depth"]*np.exp(-np.asarray(x)**2/(2*case["width"]**2))

def rhs(kind, x, y, energy, case):
    ev = energy-potential(x, case)
    m = case["mass"]
    if kind == "dirac":
        return np.array([-(ev+m)*y[1], (ev-m)*y[0]])
    return np.array([y[1], (m*m-ev*ev)*y[0]])

def full_values(kind, polynomial, x, state, derivative=0):
    x = np.asarray(x)
    values = polynomial(np.abs(x)) if derivative==0 else polynomial(np.abs(x), derivative)
    # solve_bvp returns axis=1 PPoly / OdeSolution; restored canonical PPoly uses axis=0.
    y = values.copy() if getattr(polynomial,"axis",1)==0 else values.T.copy()
    # u/phi parity (-1)^state; v or phi' has the opposite parity.
    for j in range(2):
        if (state+j+derivative) % 2:
            y[:, j] *= np.sign(x)
    return y

def reference_on(directory, kind, state, x):
    from scipy.interpolate import PPoly
    d = np.load(Path(directory)/f"{kind}_state_{state}.npz")
    poly = PPoly(d["coefficients"], d["knots"])
    y = full_values(kind, poly, x, state)/float(d["norm"])
    if kind == "kg":
        y = y[:, :1]
    return y, float(d["energy"])

def aligned_l2(x, y, ref):
    sign = 1.0 if np.trapezoid(np.sum(y*ref, axis=1), x) >= 0 else -1.0
    err = sign*y-ref
    return float(np.sqrt(np.trapezoid(np.sum(err*err, axis=1), x)/np.trapezoid(np.sum(ref*ref, axis=1), x))), sign

def count_nodes(y):
    z = np.asarray(y)
    z = z[np.abs(z)>1e-7*np.max(np.abs(z))]
    return int(np.sum(z[:-1]*z[1:] < 0))
