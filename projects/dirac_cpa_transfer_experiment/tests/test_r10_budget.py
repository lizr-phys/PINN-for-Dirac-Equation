import json
import numpy as np
import pytest
import torch
from src.r9_config import R9Config
from src.r9_loss import hamiltonian_apply
from src.r9_reference import initial_numpy
from src.r10_config import SEEDS, RAW, source_directory
from src.r10_norm_diagnostics import norm_balance, carrier_envelope_decomposition
from src.r10_state import restore_source

torch.set_num_threads(1)


def test_norm_balance_exposes_amplitude_drift_without_reference_or_normalization():
    cfg = R9Config(n_min=-48, n_max=80)
    times = np.linspace(0, 5, 4001)
    initial = initial_numpy(cfg)
    amplitude = 1+.01*times**2
    raw = amplitude[:, None, None]*initial
    dt = .02*times[:, None, None]*initial
    residual = 1j*dt-hamiltonian_apply(torch.tensor(raw), cfg).numpy()
    metrics, table = norm_balance(cfg, times, raw, residual)
    np.testing.assert_allclose(table.norm_rate_from_residual, 2*amplitude*.02*times, atol=2e-15)
    assert metrics["fine_grid_max_norm_drift"] == pytest.approx(.5625, abs=2e-15)
    assert metrics["norm_balance_integration_max_abs_discrepancy"] < 5e-8
    assert metrics["norm_balance_is_continuous_certificate"] is False


def test_arbitrary_unit_carrier_defect_has_no_direct_norm_source():
    cfg = R9Config(n_min=-48, n_max=80)
    times = np.linspace(0, 5, 17)
    k = np.asarray(cfg.wave_numbers)
    angle = np.arctan2(k, cfg.mass)/2
    u = np.stack((np.stack((np.cos(angle), np.sin(angle)), -1),
                  np.stack((-np.sin(angle), np.cos(angle)), -1)), 1)
    rates = np.stack((-.3-.1*k, .8+.2*k), -1)  # Deliberately incorrect, no exact rates.
    carrier = np.exp(1j*times[:, None, None]*rates[None])
    unit_state = carrier[..., None]*u[None]
    unit_dt = 1j*rates[None, ..., None]*unit_state
    free_apply = k[None, :, None, None]*unit_state[..., ::-1]
    free_apply += cfg.mass*np.stack((unit_state[..., 0], -unit_state[..., 1]), -1)
    unit_defect = 1j*unit_dt-free_apply
    rng = np.random.default_rng(712)
    envelope = (rng.normal(size=(129, 2))+1j*rng.normal(size=(129, 2)))/30
    raw = np.einsum("tkbc,kb->tkc", unit_state, envelope)
    dt = np.einsum("tkbc,kb->tkc", unit_dt, envelope)
    residual = 1j*dt-hamiltonian_apply(torch.tensor(raw), cfg).numpy()
    values, _ = carrier_envelope_decomposition(cfg, times, raw, residual, carrier, unit_defect)
    assert values["weighted_carrier_defect_spinor_rms"] > .1
    assert values["carrier_defect_direct_norm_projection_max_abs"] < 1e-13
    assert values["hermitian_coupling_global_norm_projection_max_abs"] < 1e-13
    assert values["envelope_vs_complete_norm_projection_max_difference"] < 1e-13


def test_source_optimizer_sampler_and_all_five_seed_ancestry_preserved():
    assert SEEDS == (300, 301, 302, 303, 304)
    cfg, model, optimizer, free, coupled, inventory = restore_source(300)
    assert inventory["inherited_optimizer_parameter_steps"] == [14000.]
    assert inventory["optimizer_reset"] is False and inventory["sampler_reset"] is False
    assert len(free.trace) == 6000 and len(coupled.trace) == 14000
    assert all(not p.requires_grad for p in model.carrier.parameters())
    control = torch.load(source_directory(300).parents[1]/"C_double_complex"/"seed_300"/"recovery_state.pt",
                         weights_only=False, map_location="cpu")
    _, time = coupled.sample()
    expected = control["coupled_sampler"]["trace"][14000].repeat_interleave(cfg.mode_count, 0)
    torch.testing.assert_close(time, expected, atol=0, rtol=0)
    assert all((source_directory(seed)/"metrics.json").is_file() for seed in SEEDS)


def test_short_exact_continuation_recovery_preserves_source_files(tmp_path):
    from src.train_r10_budget import train_continuation
    from src.freeze_r7_phase_release import sha256
    source = source_directory(300)
    before = {p.name: sha256(p) for p in source.iterdir() if p.is_file()}
    whole, resumed = tmp_path/"whole", tmp_path/"resumed"
    train_continuation(300, whole, additional_steps=2, smoke=True)
    train_continuation(300, resumed, additional_steps=2, smoke=True, stop_after=1)
    train_continuation(300, resumed, additional_steps=2, smoke=True)
    a, b = [torch.load(p/"model_state.pt", weights_only=False) for p in (whole, resumed)]
    assert a["steps"] == b["steps"] == 20002
    for key, value in a["state_dict"].items():
        torch.testing.assert_close(value, b["state_dict"][key], atol=0, rtol=0)
    assert a["coupled_time_stream_sha256"] == b["coupled_time_stream_sha256"]
    for name, digest in before.items():
        assert sha256(source/name) == digest


def test_formal_final_continuation_completeness_if_ready():
    from src.r10_config import run_directory, ARTIFACTS
    if not all((run_directory(seed)/"metrics.json").exists() for seed in SEEDS):
        pytest.skip("Fixed endpoint continuations not yet complete")
    for seed in SEEDS:
        directory = run_directory(seed)
        assert all((directory/name).is_file() for name in ARTIFACTS+("recovery_state.pt", "norm_balance.csv", "residual_decomposition.csv"))
        metrics = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
        assert metrics["fixed_final_step"] == 26000 and metrics["additional_optimizer_steps"] == 6000
        assert metrics["new_carrier_steps"] == 0 and metrics["coupled_steps"] == 20000
