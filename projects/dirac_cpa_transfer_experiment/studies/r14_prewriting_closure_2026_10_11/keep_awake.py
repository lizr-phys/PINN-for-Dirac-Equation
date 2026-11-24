"""Temporary AC-only Windows sleep request while this authorized study is active."""
import ctypes,json,os,time
import psutil
from plan import BASE,write

def main():
    lock=BASE/'runtime/awake_guard_lock.json'
    if lock.exists():
        old=json.loads(lock.read_text(encoding='utf-8'))
        try:
            p=psutil.Process(old['pid'])
            if abs(p.create_time()-old['created'])<.01 and p.pid!=os.getpid():raise RuntimeError('A study awake guard is already active')
        except psutil.NoSuchProcess:pass
    process=psutil.Process();process.cpu_affinity([7]);write(lock,dict(pid=os.getpid(),created=process.create_time()))
    ES_CONTINUOUS=0x80000000;ES_SYSTEM_REQUIRED=0x1
    kernel=ctypes.windll.kernel32;kernel.SetThreadExecutionState.argtypes=[ctypes.c_uint];kernel.SetThreadExecutionState.restype=ctypes.c_uint
    try:
        while True:
            acceptance=BASE/'acceptance.json'
            if acceptance.exists() and json.loads(acceptance.read_text(encoding='utf-8')).get('status')=='COMPLETE':break
            progress=json.loads((BASE/'progress.json').read_text(encoding='utf-8'))
            alive=psutil.pid_exists(progress['driver_pid'])
            verification=BASE/'verification_progress.json'
            recent=verification.exists() and time.time()-verification.stat().st_mtime<7200
            if not alive and not recent:break
            battery=psutil.sensors_battery();ac=battery is None or battery.power_plugged
            state=ES_CONTINUOUS|(ES_SYSTEM_REQUIRED if ac else 0)
            result=kernel.SetThreadExecutionState(state)
            write(BASE/'runtime/awake_guard_status.json',dict(pid=os.getpid(),updated_utc=time.time(),active=True,AC_power=ac,
                system_idle_sleep_request=bool(ac and result),display_sleep_allowed=True,no_permanent_power_settings_changed=True,
                stops_when_accepted_or_work_inactive=True))
            if not result:raise RuntimeError('Temporary Windows power request failed')
            time.sleep(30)
    finally:
        kernel.SetThreadExecutionState(ES_CONTINUOUS)
        write(BASE/'runtime/awake_guard_status.json',dict(pid=os.getpid(),updated_utc=time.time(),active=False,system_idle_sleep_request=False,
            no_permanent_power_settings_changed=True))

if __name__=='__main__':main()
