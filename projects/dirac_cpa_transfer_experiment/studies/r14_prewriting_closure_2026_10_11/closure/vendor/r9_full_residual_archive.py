from __future__ import annotations

"""Post-freeze archival supplement only; no primary metric or learned state changes."""
import json
import time
import numpy as np
import pandas as pd
import torch
from .r9_config import RAW, MODELS, SEEDS, run_directory
from .r9_models import R9Model
from .r9_evaluate import spectral_grid
from .freeze_r9_wavepacket import selected_config, verify_freeze
from .freeze_r7_phase_release import sha256, write_json


def assemble_full_residual(retained, positive_coupling_source, cfg):
    if retained.shape[-2:] != (cfg.mode_count, 2) or positive_coupling_source.shape[-2:] != (16, 2):
        raise ValueError("Complete retained spinor residual and two outer eight-index blocks required")
    # Predicted exterior state is zero, so R_ext = -V0/2 * psi_boundary.
    full = np.concatenate((-positive_coupling_source[..., :8, :], retained,
                           -positive_coupling_source[..., 8:, :]), axis=-2)
    return np.arange(cfg.n_min-8, cfg.n_max+9), full


def supplement_run(directory):
    directory = directory.resolve()
    record_path = directory/"full_physical_residual_archival_record.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if sha256(directory/"full_physical_residual.npz") != record["supplement_sha256"]:
            raise RuntimeError("Changed completed full-residual supplement")
        return record
    cfg = selected_config()
    freeze_sha = verify_freeze()
    originals = {p.name: sha256(p) for p in directory.iterdir() if p.is_file()
                 and p.name not in ("full_physical_residual.npz", "full_physical_residual_archival_record.json")}
    tick = time.perf_counter()
    saved = torch.load(directory/"model_state.pt", weights_only=False, map_location="cpu")
    model = R9Model(cfg, saved["model"], saved["seed"])
    model.load_state_dict(saved["state_dict"])
    if model.carrier:
        model.carrier.freeze()
    model.eval()
    with np.load(directory/"dense_grid.npz") as dense:
        times = dense["residual_t"]
        retained = dense["residual_complex_unweighted"]
        source = dense["exterior_residual_complex"]
        raw = spectral_grid(model, cfg, times)
        expected_source = cfg.potential_strength/2*np.concatenate((raw[:, :8], raw[:, -8:]), 1)
        source_error = float(abs(expected_source-source).max())
        if source_error > 1e-12:
            raise RuntimeError("Archived exterior source differs from fixed final model")
        indices, full = assemble_full_residual(retained, source, cfg)
        full_norm = np.linalg.norm(full, axis=(1, 2))
        old_norm = np.sqrt(dense["residual_l2_norm"]**2+dense["exterior_residual_l2_norm"]**2)
        norm_error = float(abs(full_norm-old_norm).max())
        if norm_error > 1e-12:
            raise RuntimeError("Signed full residual changed the previously computed norm")
        np.savez_compressed(directory/"full_physical_residual.npz", t=times, n=indices, k=indices/8,
            residual_complex_unweighted=full, retained_n=np.asarray(cfg.indices),
            retained_residual_complex=retained, exterior_n=dense["exterior_residual_n"],
            exterior_residual_complex=-source, exterior_positive_coupling_source_complex=source,
            full_residual_l2_norm=full_norm, spectral_prediction_raw_fine=raw)
    if any(sha256(directory/name) != value for name, value in originals.items()):
        raise RuntimeError("Archival supplement modified original formal evidence")
    record = dict(status="PASS", model=saved["model"], seed=saved["seed"], original_files_unchanged=originals,
        freeze_manifest_sha256=freeze_sha, primary_metrics_changed=False, added_optimizer_steps=0,
        original_exterior_array_convention="positive potential coupling source; norm correct",
        physical_exterior_residual_convention="minus positive potential coupling source",
        archived_source_vs_final_prediction_max_abs_error=source_error,
        full_residual_norm_vs_original_max_abs_error=norm_error,
        postprocessing_forward_spinor_points=len(times)*cfg.mode_count,
        derived_exterior_residual_spinor_points=len(times)*16, additional_coupled_residual_calls=0,
        elapsed_postprocessing_seconds=time.perf_counter()-tick,
        supplement_sha256=sha256(directory/"full_physical_residual.npz"),
        supplement_code_sha256=sha256(__file__))
    write_json(record_path, record)
    return record


def supplement_all():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    records = [supplement_run(run_directory(model, seed)) for model in MODELS for seed in SEEDS]
    write_json(RAW/"full_physical_residual_archival_supplement.json", dict(status="PASS", records=records,
        all_original_files_unchanged=True, primary_metrics_changed=False, total_added_optimizer_steps=0))
    pd.DataFrame([{key: record[key] for key in ("model", "seed", "elapsed_postprocessing_seconds",
        "postprocessing_forward_spinor_points", "derived_exterior_residual_spinor_points",
        "additional_coupled_residual_calls", "added_optimizer_steps",
        "archived_source_vs_final_prediction_max_abs_error", "full_residual_norm_vs_original_max_abs_error")}
        for record in records]).to_csv(RAW/"postfreeze_archival_cost.csv", index=False)
    return records


if __name__ == "__main__":
    print(json.dumps(supplement_all(), indent=2))
