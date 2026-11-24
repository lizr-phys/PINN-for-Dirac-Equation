import psutil
from threadpoolctl import threadpool_limits
from closure.spectral import run
psutil.Process().cpu_affinity([6])
with threadpool_limits(limits=1):run()
