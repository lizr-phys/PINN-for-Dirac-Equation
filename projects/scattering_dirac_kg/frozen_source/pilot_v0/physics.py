from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CASE = dict(mass=1.0, depth=0.5, width=2.0, half_length=16.0)
KINDS = ("dirac", "kg")
ENERGIES = (1.02, 1.2, 1.5, 2.0)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("physics.py", "model.py", "train.py")}


def potential(x, case=CASE):
    x = np.asarray(x)
    return -case["depth"] * np.exp(-x*x/(2*case["width"]**2))


def potential_prime(x, case=CASE):
    return -np.asarray(x)*potential(x, case)/case["width"]**2


def free_data(kind, energy, mass=1.0):
    if not energy > mass:
        raise ValueError("Scattering requires fixed E > m.")
    k = np.sqrt(energy**2-mass**2)
    amplitude = np.sqrt((energy+mass)/(2*k)) if kind == "dirac" else 1/np.sqrt(2*k)
    ratio = k/(energy+mass) if kind == "dirac" else 1j*k
    return k, amplitude, ratio


def free_matrix(kind, energy, x, mass=1.0):
    """Columns are right/left movers, each carrying flux +/-1.

    Dirac state=(psi1,psi2), KG state=(phi,phi'). The incident coefficient
    is exactly 1+0j in this global exp(ik*x) convention.
    """
    k, amplitude, ratio = free_data(kind, energy, mass)
    return amplitude*np.array([[1, 1], [ratio, -ratio]], complex) @ np.diag([np.exp(1j*k*x), np.exp(-1j*k*x)])


def ode_matrix(kind, energy, value, mass=1.0):
    a, b = energy-value+mass, energy-value-mass
    return np.array([[0, 1j*a], [1j*b, 0]], complex) if kind == "dirac" else np.array([[0, 1], [-a*b, 0]], complex)


def current(kind, state):
    state = np.asarray(state)
    product = state[..., 0].conj()*state[..., 1]
    return 2*product.real if kind == "dirac" else 2*product.imag


def relative_l2(x, value, reference):
    """No phase fitting, sign flipping or post-hoc amplitude normalization."""
    if value.ndim == 1:
        value, reference = value[:, None], reference[:, None]
    return float(np.sqrt(np.trapezoid(np.sum(abs(value-reference)**2, axis=1), x)
                         /np.trapezoid(np.sum(abs(reference)**2, axis=1), x)))


def phase_error(value, reference, amplitude_floor=1e-5):
    if min(abs(value), abs(reference)) < amplitude_floor:
        return None
    return float(abs(np.angle(value*reference.conjugate())))


def energy_name(energy):
    return f"E_{energy:.8f}".replace(".", "p")

