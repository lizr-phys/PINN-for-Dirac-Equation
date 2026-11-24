"""Damped Gauss-Newton refinement of the physics residual, with AD Jacobians."""
from __future__ import annotations
import time
import torch
from torch.nn.utils import parameters_to_vector,vector_to_parameters

def physics_vector(loss_fn,variant):
    model=loss_fn.model
    x=loss_fn.x.detach().requires_grad_(True)
    if variant!="legacy":
        scale=model.normalizer()
        r=model.residual(x,scale)
        if model.kind=="dirac" and variant=="derivative": r=r[:,1:2]
        terms=[(torch.sqrt(2*loss_fn.w)*r).reshape(-1)]
        if loss_fn.lower_state is not None:
            ly,le=loss_fn.lower_state
            yq=model.raw_wave(model.q.detach().requires_grad_(True))/scale
            if model.kind=="dirac":
                overlap=2*torch.sum(model.qw*yq*ly)
            else:
                v=model.potential(model.q)
                overlap=2*torch.sum(model.qw*(model.energy+le-2*v)*yq*ly)
                Q=2*torch.sum(model.qw*2*(model.energy-v)*yq*yq)
                Ql=2*torch.sum(model.qw*2*(le-v)*ly*ly)
                overlap=overlap/torch.sqrt(Q*Ql)
            terms.append(torch.sqrt(torch.tensor(loss_fn.ortho_weight))*overlap.reshape(1))
    else:
        r=model.residual(x)
        yq=model(loss_fn.q)
        norm=2*torch.sum(loss_fn.qw*yq*yq)
        terms=[(torch.sqrt(2*loss_fn.w)*r).reshape(-1),
               (torch.sqrt(torch.tensor(10.))*(norm-1)).reshape(1),
               .1*model.boundary_residual().reshape(-1)]
    return torch.cat(terms)

def residual_jacobian(vector,parameters,batch_size=16):
    rows=[]
    n=vector.numel()
    for first in range(0,n,batch_size):
        last=min(n,first+batch_size)
        directions=torch.zeros((last-first,n),dtype=vector.dtype)
        directions[torch.arange(last-first),torch.arange(first,last)]=1
        gs=torch.autograd.grad(vector,parameters,grad_outputs=directions,
                               is_grads_batched=True,retain_graph=last<n,allow_unused=True)
        rows.append(torch.cat([g.reshape(last-first,-1) if g is not None else
                               torch.zeros((last-first,p.numel()),dtype=p.dtype)
                               for g,p in zip(gs,parameters)],dim=1))
    return torch.cat(rows)

def polish(model,vector_fn,max_steps=40,target=1e-12,callback=None,batch_size=16):
    params=list(model.parameters())
    factor=1e-5
    trace=[]
    calls=0
    accepted=0
    start=time.perf_counter()
    reason="step_budget"
    for step in range(max_steps):
        vector=vector_fn();calls+=1
        r=vector.detach()
        cost=float(torch.dot(r,r))
        if cost<=target:
            reason="physics_tolerance"
            break
        J=residual_jacobian(vector,params,batch_size)
        if not bool(torch.isfinite(J).all()):
            reason="nonfinite_jacobian"
            break
        gram=J@J.T
        diagonal=max(float(torch.diagonal(gram).max()),1e-16)
        before=parameters_to_vector(params).detach().clone()
        identity=torch.eye(len(r),dtype=r.dtype)
        improvement=False
        for attempt in range(10):
            damping=max(factor*diagonal,1e-16)
            delta=-(J.T@torch.linalg.solve(gram+damping*identity,r))
            # Limit a single step before testing the actual nonlinear objective.
            dn=float(torch.linalg.norm(delta))
            if dn>1.: delta=delta/dn
            vector_to_parameters(before+delta,params)
            try:
                trial=vector_fn();calls+=1
            except FloatingPointError:
                calls+=1
                vector_to_parameters(before,params)
                factor*=10
                continue
            next_cost=float(torch.sum(trial.detach()**2))
            if torch.isfinite(trial).all() and next_cost<cost:
                improvement=True;accepted+=1;factor=max(factor/3,1e-14)
                break
            vector_to_parameters(before,params)
            factor*=10
        trace.append(dict(step=step+1,cost_before=cost,cost_after=next_cost if improvement else cost,
                          damping=damping,step_norm=dn,accepted=improvement,calls=calls,
                          seconds=time.perf_counter()-start))
        if callback is not None: callback(step+1,calls)
        if not improvement:
            reason="no_decreasing_step"
            break
    else:
        step=max_steps-1
    return dict(steps=len(trace),accepted_steps=accepted,evaluations=calls,stop_reason=reason,trace=trace)
