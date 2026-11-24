from __future__ import annotations
import ctypes
import os


def actual_affinity_mask():
    if os.name != "nt":
        return sum(1 << cpu for cpu in os.sched_getaffinity(0))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return process.value
