from __future__ import annotations
import json
from datetime import datetime,timezone
from .r12_config import ROOT,RAW,DOCS
from .freeze_r7_phase_release import write_json,sha256

def preserve_sources():
    path=RAW/'protected_prior_results.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    from .freeze_r11_ablation import verify_freeze,verify_prior_results
    verify_freeze()
    if verify_prior_results()['status'] != 'PASS':raise RuntimeError('Prior history integrity failed')
    previous=ROOT/'results/raw/r11_independent_replication_ablation'
    if json.loads((previous/'audit.json').read_text(encoding='utf-8'))['status'] != 'PASS':
        raise RuntimeError('R11 audit not PASS')
    files=json.loads((previous/'protected_prior_results.json').read_text(encoding='utf-8'))['files']
    for parent in ('results/raw','results/figures','results/logs','docs/experiments'):
        for p in (ROOT/parent/'r11_independent_replication_ablation').rglob('*'):
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha256(p)
    for p in [ROOT/'R11_INDEPENDENT_REPLICATION_ABLATION_REPORT.md']+list((ROOT/'src').glob('*r11*.py'))+list((ROOT/'tests').glob('*r11*.py')):
        files[p.relative_to(ROOT).as_posix()]=sha256(p)
    result=dict(captured_utc=datetime.now(timezone.utc).isoformat(),files=files,r1_through_r11_preserved=True)
    write_json(path,result);return result

def verify_prior_results():
    files=json.loads((RAW/'protected_prior_results.json').read_text(encoding='utf-8'))['files']
    changed=[p for p,d in files.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=d]
    return dict(status='PASS' if not changed else 'FAIL',checked_files=len(files),changed=changed)

def science_files():
    from .freeze_r11_ablation import science_files as inherited
    names=('r12_config.py','r12_models.py','r12_hamiltonian.py','r12_residual.py','r12_norm_diagnostics.py',
        'r12_reference.py','r12_evaluate.py','r12_decision.py','train_r12_robustness.py',
        'run_r12_robustness.py','freeze_r12_robustness.py')
    files=inherited()+[ROOT/'src'/n for n in names]+[ROOT/'tests/test_r12_robustness.py']
    files += [DOCS/n for n in ('preregistration.md','theory_check.md','figure_contract.md')]
    files += [RAW/n for n in ('protocol.json','reference_selection_policy.json','reference_validation.json',
        'history_review.json','cost_estimate.json','protected_prior_results.json','smoke_validation.json','pre_freeze_test_result.json')]
    files += [p for p in (RAW/'references').rglob('*') if p.is_file()]
    return sorted(set(files))

def freeze():
    path=RAW/'frozen_protocol_and_code.json'
    if path.exists():return verify_freeze()
    preserve_sources()
    if list((RAW/'formal').rglob('recovery_state.pt')):raise RuntimeError('Must freeze before formal training')
    for n in ('reference_validation.json','smoke_validation.json','pre_freeze_test_result.json'):
        if json.loads((RAW/n).read_text(encoding='utf-8'))['status']!='PASS':raise RuntimeError('Pre-freeze gate failed: '+n)
    write_json(path,dict(frozen_utc=datetime.now(timezone.utc).isoformat(),files={p.relative_to(ROOT).as_posix():sha256(p) for p in science_files()}))
    return sha256(path)

def verify_freeze():
    path=RAW/'frozen_protocol_and_code.json'
    for p,d in json.loads(path.read_text(encoding='utf-8'))['files'].items():
        if not (ROOT/p).is_file() or sha256(ROOT/p)!=d:raise RuntimeError('R12 frozen file changed: '+p)
    return sha256(path)
