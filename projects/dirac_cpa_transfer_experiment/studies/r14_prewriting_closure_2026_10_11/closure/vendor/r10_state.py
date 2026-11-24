from __future__ import annotations
import random
import numpy as np
import torch
from .r10_config import source_directory, BASE_MODEL
from .r9_models import R9Model, tensor_digest
from .r8_sampling import TimeSampler
from .freeze_r9_wavepacket import selected_config
from .freeze_r7_phase_release import sha256


def restore_source(seed):
    cfg = selected_config()
    directory = source_directory(seed)
    checkpoint = torch.load(directory/"model_state.pt", map_location="cpu", weights_only=False)
    saved = torch.load(directory/"recovery_state.pt", map_location="cpu", weights_only=False)
    if saved["step"] != 20000 or saved["phase"] != "coupled" or checkpoint["steps"] != 20000:
        raise RuntimeError("Only the registered R9 fixed final state may initialize continuation")
    if not all(torch.equal(value, saved["state_dict"][key]) for key, value in checkpoint["state_dict"].items()):
        raise RuntimeError("Source checkpoint and recovery differ")
    model = R9Model(cfg, BASE_MODEL, seed)
    model.load_state_dict(saved["state_dict"])
    model.carrier.freeze()
    optimizer = torch.optim.Adam(model.networks.parameters(), lr=cfg.learning_rate)
    optimizer.load_state_dict(saved["optimizer"])
    free, coupled = TimeSampler(cfg, seed, carrier=True), TimeSampler(cfg, seed)
    free.load_state_dict(saved["free_sampler"])
    coupled.load_state_dict(saved["coupled_sampler"])
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    inventory = dict(source_checkpoint_sha256=sha256(directory/"model_state.pt"),
        source_recovery_sha256=sha256(directory/"recovery_state.pt"),
        source_model_tensor_sha256=tensor_digest(model.state_dict().values()),
        frozen_carrier_tensor_sha256=tensor_digest(model.carrier.state_dict().values()),
        inherited_global_step=20000, inherited_coupled_updates=14000,
        inherited_carrier_updates=6000, optimizer_reset=False, sampler_reset=False,
        inherited_optimizer_parameter_steps=sorted(set(float(value["step"]) for value in optimizer.state.values())))
    return cfg, model, optimizer, free, coupled, inventory
