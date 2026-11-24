"""Verify seven mechanism streams; retain blocks prevented by scientific failures."""
import json,time
from plan import BASE,CORE,SEEDS,write

def main():
    rows=[];missing=[];blocks=[];mismatches=[]
    groups=('original_pde','static_pde','original_hard','learned_hard','analytic_hard','joint_hard','no_carrier_hard')
    for case in CORE:
        for seed in SEEDS:
            configs={};states={};failed=[];unknown=[]
            for group in groups:
                directory=BASE/'formal/coupled'/case/group/f'seed_{seed}'
                config=directory/'config.json';checkpoint=directory/'snapshots/20000/model_state.pt'
                if not config.exists() or not checkpoint.exists():
                    failure=directory/'failure.json'
                    if failure.exists():
                        f=json.loads(failure.read_text(encoding='utf-8'))
                        if f['status']=='DEPENDENCY_FAILED' or (f['status']=='NUMERICAL_FAILURE' and f.get('occurred_step',0)<=20000):
                            failed.append(group);continue
                    unknown.append(group);missing.append(str(directory.relative_to(BASE)));continue
                configs[group]=json.loads(config.read_text(encoding='utf-8'))
                import torch
                states[group]=torch.load(checkpoint,weights_only=False)
            block=dict(case=case,seed=seed,groups_with_20000_provenance=len(states),
                scientific_failure_groups=failed,missing_or_unexplained_groups=unknown,
                status='INCOMPLETE' if unknown else 'SCIENTIFIC_FAILURE_PREVENTS_FULL_BITWISE_BLOCK')
            if len(configs)==7:
                envelopes={r['initial_hash']['envelope'] for r in configs.values()}
                streams={r['normalized_time_stream_sha256'] for r in states.values()}
                passed=len(envelopes)==1 and len(streams)==1
                block['status']='PASS_FULL_BITWISE_BLOCK' if passed else 'MISMATCH'
                if not passed:mismatches.append(block)
                else:rows.append(dict(case=case,seed=seed,all_seven_envelope_initializations_bitwise_paired=True,
                    all_seven_20000_time_streams_identical=True,envelope_sha256=next(iter(envelopes)),time_sha256=next(iter(streams))))
            blocks.append(block)
    accounted=sum(b['status'] in ('PASS_FULL_BITWISE_BLOCK','SCIENTIFIC_FAILURE_PREVENTS_FULL_BITWISE_BLOCK') for b in blocks)
    result=dict(updated_utc=time.time(),status='FAIL' if mismatches else ('ALL_PRESCRIBED_BLOCKS_ACCOUNTED' if accounted==15 else 'PASS_AVAILABLE_COMPLETE_SEED_BLOCKS'),
        seed_blocks=len(rows),blocks_accounted=accounted,checks=rows,prescribed_block_status=blocks,missing=missing,mismatches=mismatches,
        entire_study_pairing_complete=len(rows)==15,all_prescribed_blocks_accounted=accounted==15,
        scientific_failure_does_not_imply_bitwise_pairing_verified=True)
    write(BASE/'analysis_outputs/pairing_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','prescribed_block_status','missing')},ensure_ascii=False))
    if mismatches:raise RuntimeError('Mechanism pairing differs from frozen design')

if __name__=='__main__':main()
