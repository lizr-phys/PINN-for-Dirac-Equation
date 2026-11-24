"""Operational batches, fixed scientific jobs, exact-state recovery and progress."""
from __future__ import annotations
import argparse,concurrent.futures,json,os,subprocess,sys,time,traceback,shutil
from datetime import datetime,timezone,timedelta
from pathlib import Path
import psutil
from plan import BASE,CORE,EXTENSION,SEEDS,write
from closure.jobs import development_jobs,select_baselines,core_jobs,extension_jobs

ENV=os.environ.copy()
ENV.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
RUNNING={};STATE={'status':'PREPARING','driver_pid':os.getpid(),'stage':'preflight','started_utc':time.time(),'next_step':'preflight'}

def progress():
    jobs=list((BASE/'formal').rglob('complete.json')) if (BASE/'formal').exists() else []
    dev=list((BASE/'development').rglob('complete.json')) if (BASE/'development').exists() else []
    fails=list((BASE/'formal').rglob('failure.json')) if (BASE/'formal').exists() else []
    count={'formal_complete':len(jobs),'formal_planned':210,'development_complete':len(dev),'development_planned':24,'preserved_failures':len(fails)}
    STATE.update(count,active_children=list(RUNNING.values()),updated_utc=time.time(),free_disk_bytes=shutil.disk_usage(BASE).free)
    write(BASE/'progress.json',STATE)
    now=datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    text=f'# R14 当前进度\n\n北京时间：{now}\n\n状态：{STATE["status"]}；阶段：{STATE["stage"]}。\n\n正式终点 {len(jobs)}/210；开发终点 {len(dev)}/24；保留失败 {len(fails)}。未到固定终点的训练保持未完成。\n\n下一步：{STATE["next_step"]}。\n\n[冻结方案](preregistration.md) · [完整协议](frozen/design.json) · [资源估算](resource_estimate.json) · [实时状态](progress.json)\n'
    if STATE.get('error'):text+='\n工程中断：'+STATE['error']+'\n'
    (BASE/'progress.md').write_text(text,encoding='utf-8')

def task(job,slot):
    d=BASE/job['directory']
    if (d/'complete.json').exists() or (d/'failure.json').exists():return
    (BASE/'runtime/jobs').mkdir(parents=True,exist_ok=True)
    identifier=job['directory'].replace('/','__')
    spec=BASE/'runtime/jobs'/f'{identifier}.json';write(spec,job)
    log=BASE/'logs'/f'{identifier}.log';log.parent.mkdir(exist_ok=True)
    env=ENV.copy();env['R14_CPU_MASK']=str(1<<slot)
    with log.open('a',encoding='utf-8') as f:
        p=subprocess.Popen([sys.executable,'-X','utf8','-u','-m','closure.train',str(spec)],cwd=BASE,stdout=f,stderr=subprocess.STDOUT,env=env,creationflags=FLAGS)
        RUNNING[slot]={'slot':slot,'pid':p.pid,'mask':1<<slot,'job':job,'log':str(log.relative_to(BASE)),'started_utc':time.time()}
        rc=p.wait();RUNNING.pop(slot,None)
    if rc and not (d/'failure.json').exists():raise RuntimeError('Engineering interruption in '+job['directory']+'; see '+str(log))

def sequence(jobs,slot):
    for job in jobs:task(job,slot)

def batch(queues):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(sequence,j,slot) for slot,j in enumerate(queues)]
        while not all(f.done() for f in futures):
            progress();concurrent.futures.wait(futures,timeout=30,return_when=concurrent.futures.ALL_COMPLETED)
            for f in futures:
                if f.done() and f.exception():
                    for child in list(RUNNING.values()):
                        try:psutil.Process(child['pid']).terminate()
                        except psutil.Error:pass
                    raise f.exception()
        for f in futures:f.result()
    progress()

def main():
    runtime=BASE/'runtime';runtime.mkdir(exist_ok=True)
    lock=runtime/'driver_lock.json'
    if lock.exists():
        old=json.loads(lock.read_text(encoding='utf-8'))
        try:
            p=psutil.Process(old['pid'])
            if old['pid']!=os.getpid() and abs(p.create_time()-old['created'])<.01:raise RuntimeError('Another driver is active')
        except psutil.NoSuchProcess:pass
    process=psutil.Process();process.cpu_affinity([7]);write(lock,dict(pid=os.getpid(),created=process.create_time()))
    from closure.train import verify_freeze
    verify_freeze()
    try:
        STATE.update(status='RUNNING',stage='development',next_step='完成24个固定开发候选，按冻结规则选择神经基线');progress()
        jobs=development_jobs();batch([jobs[i::5] for i in range(5)])
        selection=select_baselines()
        write(runtime/'selected_execution_plan.json',dict(selection=selection,
            core={c:{str(s):core_jobs(c,s,selection) for s in SEEDS} for c in CORE},
            extension={c:{str(s):extension_jobs(c,s,selection) for s in SEEDS} for c in EXTENSION}))
        for case in CORE:
            STATE.update(stage='core:'+case,next_step='完成核心七组消融和两类神经基线；每种子保留固定终点');progress()
            batch([core_jobs(case,s,selection) for s in SEEDS])
        for case in EXTENSION:
            STATE.update(stage='extension:'+case,next_step='复用同种子m5T5载波，重训包络，与解析载波和最强神经基线比较');progress()
            batch([extension_jobs(case,s,selection) for s in SEEDS])
        STATE.update(stage='postprocessing',next_step='独立重算、零样本诊断、图表视觉检查、文献和总报告');progress()
        # Postprocessing is an explicit agent gate; training completion cannot mark the goal complete.
        STATE.update(status='TRAINING_ATTEMPTS_FINISHED_REQUIRES_VERIFICATION');progress()
    except Exception as error:
        STATE.update(status='ENGINEERING_INTERRUPTED',error=repr(error),traceback=traceback.format_exc(),
            next_step='读取日志排查工程错误，保持冻结科学方案，按原状态恢复');progress();raise
    finally:
        write(runtime/f'driver_exit_{time.time_ns()}.json',STATE)

if __name__=='__main__':main()
