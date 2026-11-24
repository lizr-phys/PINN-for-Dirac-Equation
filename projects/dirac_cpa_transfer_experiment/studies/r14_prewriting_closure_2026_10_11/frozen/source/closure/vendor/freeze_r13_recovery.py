from __future__ import annotations

import json
from datetime import datetime,timezone
from .r13_config import ROOT,RAW,DOCS
from .freeze_r7_phase_release import write_json,sha256

def preserve_sources():
    path=RAW/'protected_prior_results.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    from .freeze_r12_robustness import verify_freeze as old_freeze, verify_prior_results as old_history
    old_freeze()
    if old_history()['status']!='PASS':raise RuntimeError('Prior history integrity failed')
    prior=ROOT/'results/raw/r12_cross_condition_robustness'
    if json.loads((prior/'audit.json').read_text(encoding='utf-8'))['status']!='PASS':
        raise RuntimeError('R12 audit not PASS')
    files=dict(json.loads((prior/'protected_prior_results.json').read_text(encoding='utf-8'))['files'])
    for parent in ('results/raw','results/figures','results/logs','docs/experiments'):
        for p in (ROOT/parent/'r12_cross_condition_robustness').rglob('*'):
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for p in list((ROOT/'src').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+list(ROOT.glob('R*.md')):
        if 'r13' not in p.name.lower() and not p.name.startswith('R13'):
            files[p.relative_to(ROOT).as_posix()]=sha256(p)
    result=dict(captured_utc=datetime.now(timezone.utc).isoformat(),files=files,r1_through_r12_preserved=True)
    write_json(path,result);return result

def verify_prior_results():
    record=json.loads((RAW/'protected_prior_results.json').read_text(encoding='utf-8'))
    changed=[p for p,d in record['files'].items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=d]
    return dict(status='PASS' if not changed else 'FAIL',checked_files=len(record['files']),changed=changed)

def science_files():
    from .freeze_r12_robustness import science_files as inherited
    names=('r13_config.py','r13_models.py','r13_residual.py','r13_reference.py','r13_phase_evaluate.py',
        'r13_evaluate.py','r13_selection.py','r13_plan.py','train_r13_recovery.py','run_r13_recovery.py','freeze_r13_recovery.py')
    files=inherited()+[ROOT/'src'/n for n in names]+[ROOT/'tests/test_r13_recovery.py']
    files += [DOCS/n for n in ('preregistration.md','theory_check.md','figure_contract.md','batch_recovery_plan.md')]
    files += [RAW/n for n in ('protocol.json','reference_selection_policy.json','reference_validation.json',
        'history_review.json','cost_estimate.json','protected_prior_results.json','smoke_validation.json','pre_freeze_test_result.json')]
    files += [p for p in (RAW/'references').rglob('*') if p.is_file()]
    return sorted(set(files))

def freeze():
    path=RAW/'frozen_protocol_and_code.json'
    if path.exists():return verify_freeze()
    preserve_sources()
    if list((RAW/'formal').rglob('recovery_state.pt')):raise RuntimeError('Freeze required before formal training')
    for name in ('reference_validation.json','smoke_validation.json','pre_freeze_test_result.json'):
        if json.loads((RAW/name).read_text(encoding='utf-8'))['status']!='PASS':raise RuntimeError('Pre-freeze gate failed: '+name)
    write_json(path,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),files={p.relative_to(ROOT).as_posix():sha256(p) for p in science_files()}))
    return sha256(path)

def verify_freeze():
    path=RAW/'frozen_protocol_and_code.json'
    for p,d in json.loads(path.read_text(encoding='utf-8'))['files'].items():
        if not (ROOT/p).is_file() or sha256(ROOT/p)!=d:raise RuntimeError('R13 frozen file changed: '+p)
    return sha256(path)
