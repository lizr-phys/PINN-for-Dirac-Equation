"""Post-hoc explanatory diagnostics only; no optimization or decision changes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .aggregate_r7_phase_release import stage_runs
from .freeze_r7_phase_release import RAW, write_json


def rel(predicted, exact):
    return float(np.linalg.norm((predicted-exact).ravel())/np.linalg.norm(exact.ravel()))


def central_free_surrogate(t, cfg, initial):
    """Counterfactual V=0 trajectory, evaluated AFTER all training completes."""
    matrix = np.array([[cfg["mass"], cfg["k0"]], [cfg["k0"], -cfg["mass"]]], dtype=float)
    values, vectors = np.linalg.eigh(matrix)
    phase = np.exp(-1j*np.outer(t, values))
    center = np.einsum("sa,ta,a->ts", vectors, phase, vectors.T@initial)
    result = np.zeros((len(t), cfg["mode_count"], 2), dtype=complex)
    result[:, cfg["ladder_half_width"]] = center
    return result


def diagnose():
    main = stage_runs("main")
    if len(main) != 30:
        raise ValueError("Post-hoc diagnostics require all 30 runs; not used to adapt training")
    rows, surrogate_metrics = [], None
    for _, run in main.iterrows():
        directory = RAW/"main"/run.model/run.training_mode/f"seed_{run.seed}"
        cfg = json.loads((directory/"config.json").read_text(encoding="utf-8"))["configuration"]
        with np.load(directory/"dense_grid.npz") as data:
            exact, prediction = data["spectral_state_exact"], data["spectral_state_predicted_raw"]
            surrogate = central_free_surrogate(data["t"], cfg, exact[0, cfg["ladder_half_width"]])
            mode_power = np.mean(data["residual_by_mode_fine"], axis=0)
            sidebands = np.abs(data["mode_offsets"]) == 1
            jacobians = data["initial_output_tangent_jacobians"]
            row = dict(model=run.model, training_mode=run.training_mode, seed=int(run.seed),
                       predicted_raw_distance_to_uncoupled_center=rel(prediction, surrogate),
                       first_sideband_fraction_of_unweighted_residual=float(mode_power[sidebands].sum()/mode_power.sum()),
                       initial_phase_coordinate_tangent_norm_min=float(np.linalg.norm(jacobians[:, :, 1], axis=1).min()),
                       initial_phase_coordinate_tangent_norm_max=float(np.linalg.norm(jacobians[:, :, 1], axis=1).max()))
            if surrogate_metrics is None:
                overlap = np.vdot(exact.ravel(), surrogate.ravel())
                aligned = surrogate*np.exp(-1j*np.angle(overlap))
                fourier = np.exp(1j*np.outer(data["x"], data["wave_numbers"]))/np.sqrt(cfg["length"])
                physical = np.einsum("xk,tks->txs", fourier, surrogate)
                rho = np.sum(np.abs(physical)**2, axis=2)
                current = 2*np.real(np.conj(physical[:, :, 0])*physical[:, :, 1])
                norm = np.sum(np.abs(surrogate)**2, axis=(1, 2))
                ref_norm = np.sum(np.abs(exact)**2, axis=(1, 2))
                fidelity = np.abs(np.sum(exact.conj()*surrogate, axis=(1, 2)))**2/(norm*ref_norm)
                surrogate_metrics = dict(spinor_rel_l2_gauge_aligned=rel(aligned, exact),
                                         spinor_rel_l2_raw=rel(surrogate, exact),
                                         current_rel_l2=rel(current, data["current_exact"]),
                                         density_rel_l2=rel(rho, data["rho_exact"]),
                                         fidelity_mean=float(fidelity.mean()), max_norm_drift=float(np.max(abs(norm-1))),
                                         expected_unweighted_pde_loss=cfg["potential_strength"]**2/(4*cfg["length"]),
                                         outside_center_occupation=0.,
                                         uses_exact_free_evolution_only_for_posttraining_diagnostic=True,
                                         trained=False, participates_in_learned_ranking=False)
        history = pd.read_csv(directory/"loss_history.csv")
        last = history.iloc[-1]
        row.update(last_eighth_bin_weight=float(last.segment_8_weight),
                   last_raw_training_loss=float(last.unweighted_pde),
                   last_weighted_training_loss=float(last.objective),
                   last_amplitude_output_gradient_norm=float(last.output_r_gradient_norm),
                   last_phase_or_imaginary_output_gradient_norm=float(last.output_b_gradient_norm))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(RAW/"posthoc_failure_diagnostics.csv", index=False)
    write_json(RAW/"posthoc_failure_diagnostics.json", dict(posthoc_explanatory_only=True, changes_decision=False,
               no_new_trained_model=True, central_free_surrogate=surrogate_metrics,
               main_run_diagnostics=rows,
               limitation="Similarity to this counterfactual is evidence of an uncoupled basin, not proof of a unique failure cause"))
    print("Saved post-hoc uncoupled-basin and gradient/residual diagnostics for all 30 runs", flush=True)


if __name__ == "__main__":
    diagnose()
