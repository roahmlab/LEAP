"""CPU affinity for fence solves; one hardware thread per physical core."""
import os
from pathlib import Path


def physical_cores(count):
    seen, cpus = set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        top = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((top / "physical_package_id").read_text(),
               (top / "core_id").read_text())
        if key not in seen:
            seen.add(key)
            cpus.append(cpu)
        if len(cpus) == count:
            break
    if len(cpus) != count:
        raise RuntimeError(f"Requested {count} physical cores; found {len(cpus)}")
    os.sched_setaffinity(0, cpus)
    return cpus

