"""Canonical history tables with native metrics and immutable source references."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from plan import BASE,PROJECT,write,sha

SOURCES=[(1,'cpa_transfer','results/summary/cpa_transfer/runs.csv'),
 (2,'learned_two_phase_pilot','results/summary/learned_two_phase_pilot/runs.csv'),
 (3,'ltpb_mechanism_ablation','results/summary/ltpb_mechanism_ablation/runs.csv'),
 (4,'multimode_ltpb','results/summary/multimode_ltpb/runs.csv'),
 (5,'wavepacket_shared_spectral_ltpb','results/summary/wavepacket_shared_spectral_ltpb/runs.csv'),
 (6,'periodic_potential_coupled_ltpb','results/summary/periodic_potential_coupled_ltpb/runs.csv'),
 (7,'coupled_ltpb_phase_release_r7','results/raw/coupled_ltpb_phase_release_r7/all_runs.csv'),
 (8,'r8_coupling_recovery','results/raw/r8_coupling_recovery/all_runs.csv'),
 (9,'r9_wavepacket_periodic_potential','results/raw/r9_wavepacket_periodic_potential/per_seed_metrics.csv'),
 (10,'r10_wavepacket_budget_diagnosis','results/raw/r10_wavepacket_budget_diagnosis/per_seed_metrics.csv'),
 (11,'r11_independent_replication_ablation','results/raw/r11_independent_replication_ablation/per_seed_metrics.csv'),
 (12,'r12_cross_condition_robustness','results/raw/r12_cross_condition_robustness/per_seed_metrics.csv'),
 (13,'r13_long_time_recovery','results/raw/r13_long_time_recovery/per_seed_metrics.csv')]

def main():
    frames=[];source=[];decision=[]
    for number,name,relative in SOURCES:
        p=PROJECT/relative
        if not p.exists():raise RuntimeError('Missing canonical historical table '+relative)
        d=pd.read_csv(p);d.insert(0,'historical_round',number);d.insert(1,'historical_experiment',name)
        d.insert(2,'canonical_source_csv',relative);d.insert(3,'canonical_source_sha256',sha(p));d.insert(4,'canonical_row_index',np.arange(len(d)))
        spinor_col='spinor_rel_l2_gauge_aligned' if 'spinor_rel_l2_gauge_aligned' in d else 'spinor_rel_l2'
        d['spinor_native_metric']=d.get(spinor_col,np.nan);d['spinor_native_definition']=spinor_col
        d['costs_require_native_definition']=True
        frames.append(d);source.append(dict(round=number,experiment=name,path=relative,sha256=sha(p),rows=len(d),columns=list(d.columns)))
        candidates=[PROJECT/'results/raw'/name/'decision.json',PROJECT/'results/summary'/name/'decision.json']
        for q in candidates:
            if q.exists():
                decision.append(dict(round=number,path=str(q.relative_to(PROJECT)),sha256=sha(q),original_decision=json.loads(q.read_text(encoding='utf-8'))));break
    out=BASE/'history';pd.concat(frames,ignore_index=True,sort=False).to_csv(out/'historical_results_all_native_fields.csv',index=False)
    write(out/'canonical_csv_index.json',source);write(out/'original_decisions.json',decision)
    # No row here is counted as an additional independent replicate.
    write(out/'unified_history_status.json',dict(status='ARCHIVED_CANONICAL_ROWS',rounds=13,rows=sum(len(d) for d in frames),
        original_scientific_decisions_preserved=True,mixed_physics_not_pooled=True,aliases_not_new_replicates=True,
        all_native_columns_preserved=True,unknown_costs_not_zero=True,new_training_not_triggered=True))
    print('HISTORY_TABLES',len(frames),sum(len(d) for d in frames))

if __name__=='__main__':main()
