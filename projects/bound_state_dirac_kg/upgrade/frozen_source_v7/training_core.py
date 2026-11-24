"""Finite trial guards, physics-objective recovery and explicit mid-gap retries."""
from __future__ import annotations
import time
import torch
from .optimizer import physics_vector,polish

class NumericalTrialError(FloatingPointError):
    pass

def checked_energy(model):
    energy=float(model.energy.detach())
    if not 0<energy<model.mass:
        raise NumericalTrialError(f"Energy rounded to a gap boundary: E={energy}, m={model.mass}")

def checked_representation(model):
    checked_energy(model)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise NumericalTrialError("Nonfinite network parameter")
    if hasattr(model,"normalizer"):
        raw=model.raw_wave(model.q.detach().requires_grad_(True))
        square=2*torch.sum(model.qw*raw*raw)
        if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(square)) or float(square.detach())<=0:
            raise NumericalTrialError("Invalid unnormalized quadrature norm")
        actual=2*torch.sum(model.qw*(raw/torch.sqrt(square))**2)
        if not bool(torch.isfinite(actual)) or abs(float(actual.detach())-1)>1e-8:
            raise NumericalTrialError("Hard normalization failed in floating-point arithmetic")
        if not bool(torch.isfinite(model.boundary_residual()).all()):
            raise NumericalTrialError("Invalid relative boundary representation")

def optimize(model,loss_fn,cfg,factory,out,prefix):
    start=time.perf_counter()
    history=[];attempts=[];rejections=[]
    evaluations=0
    def checkpoint():
        path=out/"checkpoints"/f"eval_{evaluations}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(),path)
    def record(stage,step,total,parts,attempt,save=False):
        history.append(dict(stage=stage,attempt=attempt,step=step,evaluations=evaluations,
                            seconds=time.perf_counter()-start,energy=float(model.energy.detach()),
                            loss=float(total.detach()),**{k:float(v.detach()) for k,v in parts.items()}))
        if save:checkpoint()
    last_polish=None;stop="iteration_budget";calls=0;iterations=0
    for attempt in range(cfg.get("maximum_restarts",1)+1):
        if attempt:
            model,loss_fn=factory(cfg["restart_energy_fraction"],attempt)
        attempt_start=time.perf_counter()
        calls=0;iterations=0
        best_cost=float("inf");best=None
        numerical_failure=False
        def objective():
            nonlocal evaluations,best_cost,best
            checked_representation(model)
            total,parts=loss_fn();evaluations+=1
            if not bool(torch.isfinite(total)):
                raise NumericalTrialError("Nonfinite physics objective")
            if float(total.detach())<best_cost:
                best_cost=float(total.detach())
                best={k:v.detach().clone() for k,v in model.state_dict().items()}
            return total,parts
        def vector_fn():
            checked_representation(model)
            vector=physics_vector(loss_fn,model.variant if hasattr(model,"variant") else "legacy")
            if not bool(torch.isfinite(vector).all()):raise NumericalTrialError("Nonfinite residual vector")
            return vector
        initial=float(model.energy.detach())
        try:
            adam=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
            for step in range(cfg["adam_steps"]):
                adam.zero_grad();total,parts=objective()
                if step%100==0:record("adam",step,total,parts,attempt)
                total.backward()
                if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise NumericalTrialError("Nonfinite Adam gradient")
                adam.step()
            opt=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=cfg["lbfgs_chunk"],max_eval=cfg["lbfgs_chunk"]*2,
                                 history_size=80,tolerance_grad=1e-12,tolerance_change=1e-15,line_search_fn="strong_wolfe")
            calls=0
            def closure():
                nonlocal calls
                opt.zero_grad();total,parts=objective()
                (total*cfg["objective_scale"]).backward();calls+=1
                if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                    raise NumericalTrialError("Nonfinite L-BFGS gradient")
                if calls%20==0:record("lbfgs",calls,total,parts,attempt)
                return total*cfg["objective_scale"]
            for block in range((cfg["lbfgs_steps"]+cfg["lbfgs_chunk"]-1)//cfg["lbfgs_chunk"]):
                opt.step(closure)
                total,parts=objective()
                iterations=opt.state[model.raw_energy].get("n_iter",0)
                record("checkpoint",iterations,total,parts,attempt,True)
                print(f"{prefix} attempt={attempt} LBFGS {iterations}: R2={parts['pde'].item():.3e} E={model.energy.item():.10f}",flush=True)
                if float(total.detach())<=cfg["residual_stop_squared"]:
                    break
        except NumericalTrialError as exc:
            numerical_failure=True
            path=out/"rejected_trials"/f"attempt_{attempt}.pt"
            path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(),path)
            rejections.append(dict(attempt=attempt,reason=str(exc),seconds=time.perf_counter()-start))
            if best is None:raise
            model.load_state_dict(best)
            print(f"RECOVER {prefix}: {exc}; restored lowest finite physics objective",flush=True)
        total,parts=objective()
        threshold_drift=float(model.energy.detach())>.999*model.mass and float(total.detach())>cfg["retry_physics_squared"]
        last_polish=None
        if not threshold_drift and float(total.detach())>cfg["residual_stop_squared"]:
            previous=evaluations
            def callback(step,polish_calls):
                nonlocal evaluations
                evaluations=previous+polish_calls
                total,parts=loss_fn()
                record("gauss_newton",step,total,parts,attempt,True)
                if step%5==0:print(f"{prefix} attempt={attempt} GN {step}: loss={total.item():.3e}",flush=True)
            budget=cfg["recovery_gauss_newton_steps"] if numerical_failure else cfg["gauss_newton_steps"]
            last_polish=polish(model,vector_fn,budget,cfg["residual_stop_squared"],callback,cfg.get("jacobian_batch_size",16))
            evaluations=previous+last_polish["evaluations"]
        total,parts=objective()
        norm_error=abs(float(parts["norm"].detach())-1)
        if hasattr(model,"variant"):
            boundary=float(torch.linalg.norm(model.boundary_residual()).detach())
        else:boundary=float(torch.sqrt(parts["boundary"]).detach())
        stop="physics_tolerance" if float(total.detach())<=cfg["residual_stop_squared"] and norm_error<=cfg["normalization_stop"] and boundary<=cfg["boundary_stop"] else "iteration_budget"
        acceptable=float(total.detach())<=cfg["retry_physics_squared"] and 0<float(model.energy.detach())<model.mass
        attempts.append(dict(attempt=attempt,initial_energy=initial,energy=float(model.energy.detach()),
                             final_loss=float(total.detach()),final_pde=float(parts["pde"].detach()),
                             threshold_drift=threshold_drift,numerical_recovery=numerical_failure,
                             seconds=time.perf_counter()-attempt_start,physics_retry_accepted=acceptable,
                             lbfgs_closure_calls=calls,lbfgs_iterations=iterations,
                             stop_reason=stop,gauss_newton=last_polish))
        record("attempt_final",iterations,total,parts,attempt,True)
        if acceptable:break
        attempt_dir=out/"attempts"/f"attempt_{attempt}"
        attempt_dir.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),attempt_dir/"model.pt")
        if attempt<cfg.get("maximum_restarts",1):
            print(f"RESTART {prefix}: insufficient physics convergence; fixed mid-gap initial energy, no reference labels",flush=True)
    total,parts=objective();record("final",iterations,total,parts,attempt)
    return model,loss_fn,history,dict(energy=float(model.energy.detach()),seconds=time.perf_counter()-start,
                    loss=float(total.detach()),parts={k:float(v.detach()) for k,v in parts.items()},
                    evaluations=evaluations,lbfgs_closure_calls=sum(a["lbfgs_closure_calls"] for a in attempts),lbfgs_iterations=iterations,
                    stop_reason=stop,gauss_newton=last_polish,attempts=attempts,
                    restart_count=len(attempts)-1,rejected_trials=rejections,finite=True)
