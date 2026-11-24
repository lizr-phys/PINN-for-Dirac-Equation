from __future__ import annotations

import ctypes
import os


def physical_core_masks():
    """One available logical processor from each distinct Windows physical core."""
    if os.name != "nt":
        available = sorted(os.sched_getaffinity(0))
        return [1 << cpu for cpu in available]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_ulong(0)
    function = kernel.GetLogicalProcessorInformationEx
    function.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    function.restype = ctypes.c_int
    function(0, None, ctypes.byref(size))  # RelationProcessorCore
    data = ctypes.create_string_buffer(size.value)
    if not function(0, data, ctypes.byref(size)):
        raise ctypes.WinError(ctypes.get_last_error())
    masks, offset = [], 0
    while offset < size.value:
        record_size = ctypes.c_ulong.from_buffer(data, offset+4).value
        group_count = ctypes.c_ushort.from_buffer(data, offset+8+22).value
        if group_count != 1:
            raise RuntimeError("R7 CPU protocol requires a single processor group")
        group_mask = ctypes.c_size_t.from_buffer(data, offset+8+24).value
        group = ctypes.c_ushort.from_buffer(data, offset+8+24+ctypes.sizeof(ctypes.c_size_t)).value
        if group != 0:
            raise RuntimeError("Multiple processor groups need an explicit execution amendment")
        masks.append(group_mask & -group_mask)
        offset += record_size
    return masks


def set_cpu_affinity(mask):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), mask):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.sched_setaffinity(0, {n for n in range(mask.bit_length()) if mask & (1 << n)})
    os.environ["R7_CPU_AFFINITY_MASK"] = str(mask)
