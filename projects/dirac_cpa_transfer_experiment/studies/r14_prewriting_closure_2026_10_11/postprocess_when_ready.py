"""Operational queue after all frozen training attempts; acceptance remains an agent gate."""
from __future__ import annotations
import json,os,subprocess,sys,time,traceback
import psutil
from plan import BASE,write,sha

def live_training():
    found=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            if (process.info['name'] or '').lower() not in ('python.exe','pythonw.exe'):continue
            command=process.info['cmdline'] or []
            if 'closure.train' in command and (any(str(BASE).lower() in s.lower() for s in command) or process.cwd()==str(BASE)):
                found.append(process.info['pid'])
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return found

def report(**fields):
    write(BASE/'runtime/postprocessing_progress.json',dict(pid=os.getpid(),updated_utc=time.time(),
        entire_goal_accepted_complete=False,**fields))

def main():
    process=psutil.Process();process.cpu_affinity([6]);runtime=BASE/'runtime';lock=runtime/'postprocessing_lock.json'
    if lock.exists():
        old=json.loads(lock.read_text(encoding='utf-8'))
        try:
            other=psutil.Process(old['pid'])
            if abs(other.create_time()-old['created'])<.01 and other.pid!=os.getpid():raise RuntimeError('A postprocessing queue is already active')
        except psutil.NoSuchProcess:pass
    write(lock,dict(pid=os.getpid(),created=process.create_time()))
    try:
        while True:
            progress=json.loads((BASE/'progress.json').read_text(encoding='utf-8'));active=live_training()
            if progress['status']=='TRAINING_ATTEMPTS_FINISHED_REQUIRES_VERIFICATION' and not active:break
            report(status='WAITING_FOR_FROZEN_TRAINING',training_status=progress['status'],
                completed_formal=progress['formal_complete'],preserved_scientific_failures=progress['preserved_failures'],
                active_training_pids=active,next_step='预定训练完成并无活跃训练进程后启动独立复核')
            time.sleep(30)
        from closure.train import verify_freeze
        verify_freeze()
        queue_path=runtime/'postprocessing_queue.json';queue=json.loads(queue_path.read_text(encoding='utf-8'))
        if queue['scientific_source_sha256']!=verify_freeze():raise RuntimeError('Operational queue scientific source differs')
        run=runtime/'postprocessing_runs'/str(time.time_ns());run.mkdir(parents=True);completed=[]
        environment=os.environ.copy();environment.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        for index,phase in enumerate(queue['phases']):
            if live_training():raise RuntimeError('Training resumed during postprocessing; stop for process reconciliation')
            identifier=phase['id'];argv=[sys.executable,'-X','utf8','-u',*phase['python_arguments']]
            log=run/f'{index:02d}_{identifier}.log';start=time.time()
            with log.open('w',encoding='utf-8') as stream:
                child=subprocess.Popen(argv,cwd=BASE,env=environment,stdout=stream,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                while child.poll() is None:
                    report(status='RUNNING_PRESCRIBED_POSTPROCESSING',phase=identifier,phase_index=index+1,
                        phase_count=len(queue['phases']),child_pid=child.pid,completed_phases=completed,
                        phase_log=log.relative_to(BASE).as_posix(),next_step=identifier)
                    time.sleep(30)
                if child.returncode:raise RuntimeError(f'Postprocessing engineering failure: {identifier}; log={log}')
            for target,key,expected in phase.get('required_output_checks',[]):
                data=json.loads((BASE/target).read_text(encoding='utf-8'))
                if data.get(key)!=expected:raise RuntimeError(f'Postprocessing validation gate failed: {target}:{key}')
            completed.append(identifier);write(run/f'{index:02d}_{identifier}_completed.json',dict(
                phase=phase,started_utc=start,ended_utc=time.time(),queue_sha256=sha(queue_path),
                scientific_source_sha256=verify_freeze(),child_exit_code=child.returncode))
        report(status='AUTOMATED_VERIFICATION_FINISHED_REQUIRES_AGENT_REVIEW',completed_phases=completed,
            run_directory=run.relative_to(BASE).as_posix(),
            next_step='逐图实际视觉检查、最终总报告/主张证据表/代码数据清单/投稿定位、链接与整体验收')
    except Exception as error:
        report(status='ENGINEERING_REPAIR_REQUIRED',error=repr(error),traceback=traceback.format_exc(),
            next_step='读取独立后处理日志修复工程问题，科学协议和原始证据保持冻结');raise

if __name__=='__main__':main()
