from plan import BASE,write
from closure.train import run
from threadpoolctl import threadpool_limits
job=dict(type='coupled',stage='engineering',case='T10',group='joint_hard',seed=999004,budget=3,snapshots=[3],bounds=[-64,96],directory='engineering/full_pipeline')
with threadpool_limits(limits=1):result=run(job,engineering=True,do_evaluation=True)
import json,numpy as np
d=BASE/job['directory']/'snapshots/00003'
m=json.loads((d/'metrics.json').read_text(encoding='utf-8'))
with np.load(d/'prediction.npz') as f:
    shape=f['prediction'].shape
    finite=bool(np.isfinite(f['prediction']).all())
assert shape==(201,161,2) and finite
assert m['initial_condition_max_abs_error']<1e-13
assert m['evaluation_coupled_residual_points']==8001*161
write(BASE/'engineering/full_pipeline_check.json',dict(status='PASS',prediction_shape=shape,finite=finite,
    all_residual_points_executed=m['evaluation_coupled_residual_points'],external_evaluator=True,no_accuracy_selection=True))
print('FULL_PIPELINE_PASS')
