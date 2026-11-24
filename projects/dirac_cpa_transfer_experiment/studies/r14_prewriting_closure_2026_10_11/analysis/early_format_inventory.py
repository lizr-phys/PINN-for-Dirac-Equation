"""Read-only format inventory for legacy metric adapters."""
import json
import numpy as np
from plan import BASE,PROJECT,write

NAMES=('cpa_transfer','learned_two_phase_pilot','ltpb_mechanism_ablation','multimode_ltpb',
    'wavepacket_shared_spectral_ltpb','periodic_potential_coupled_ltpb','coupled_ltpb_phase_release_r7','r8_coupling_recovery')

def main():
    rows=[]
    for name in NAMES:
        paths=sorted((PROJECT/'results/raw'/name).rglob('dense_grid.npz'));p=paths[0]
        with np.load(p) as f:keys=f.files
        c=p.parent/'config.json';m=json.loads((p.parent/'metrics.json').read_text(encoding='utf-8'))
        conf=json.loads(c.read_text(encoding='utf-8')) if c.exists() else {}
        rows.append(dict(experiment=name,count_dense=len(paths),example=str(p.relative_to(PROJECT)),keys=keys,config=conf,
            metric_keys=[k for k in m if k in ('spinor_rel_l2','spinor_rel_l2_gauge_aligned','spinor_rel_l2_raw',
                'fidelity_mean','modal_population_rel_l2','mode_occupation_rmse_rel','spectral_occupation_rel_l2',
                'max_norm_drift','current_rel_l2','rho_rel_l2','density_rel_l2')]))
    write(BASE/'history/early_raw_format_inventory.json',rows)
    print(json.dumps([{k:v for k,v in r.items() if k not in ('config','keys')}|
        {'physical_keys':[k for k in r['keys'] if k.startswith('psi') or k in ('x','t','wave_numbers')]} for r in rows],indent=2))

if __name__=='__main__':main()
