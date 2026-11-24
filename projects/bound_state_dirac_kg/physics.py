from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def potential(x, mass=1.0, depth=0.5, width=2.0):
    return -depth * np.exp(-x*x / (2*width*width))


def rhs(kind, x, y, energy, mass=1.0, depth=0.5, width=2.0):
    ev = energy - potential(x, mass, depth, width)
    if kind == "dirac":
        return np.array([-(ev+mass)*y[1], (ev-mass)*y[0]])
    return np.array([y[1], (mass*mass-ev*ev)*y[0]])


def count_nodes(y, threshold=1e-7):
    y = np.asarray(y)
    z = y[np.abs(y) > threshold*np.max(np.abs(y))]
    return int(np.sum(z[1:]*z[:-1] < 0))


def aligned_l2(x, actual, reference):
    dot = np.trapezoid(np.sum(actual*reference, axis=1), x)
    sign = 1.0 if dot >= 0 else -1.0
    error = sign*actual-reference
    return float(np.sqrt(np.trapezoid(np.sum(error**2, axis=1), x) /
                         np.trapezoid(np.sum(reference**2, axis=1), x))), sign
