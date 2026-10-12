"""This module, imported by `leaptrack` before anything that computes, does two
things.

1. **One thread per math library.** OpenMP, OpenBLAS and MKL are pinned to one
   thread through their environment variables, which take effect only if they are
   set before numpy is first imported; `THREAD_ENV_OK` says whether that was in time.

2. **The plain (non-AVX) build of ProxQP.** The proxsuite wheel ships an AVX-512,
   an AVX2 and a plain build and picks one by CPU (`proxsuite/__init__.py`). This
   module makes it load the plain build, and raises ImportError if another build
   is loaded (`LEAPTRACK_QP_BUILD=native` lets proxsuite choose, for diagnosis).

`QP_BUILD` names the build in use; `leaptrack.rollout` records `describe()` with
every rollout.
"""
from __future__ import annotations

import os
import sys
import types

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
THREAD_ENV_OK = "numpy" not in sys.modules
if THREAD_ENV_OK:
    for _k in THREAD_VARS:
        os.environ[_k] = "1"

# DIAGNOSTIC ONLY: LEAPTRACK_QP_BUILD=native lets proxsuite pick its build by CPU.
_NATIVE = os.environ.get("LEAPTRACK_QP_BUILD", "plain").lower() == "native"

_fake_used = "proxsuite" not in sys.modules and not _NATIVE
if _fake_used:
    # proxsuite/__init__.py asks `proxsuite.instructionset` which build the CPU
    # supports; answering "neither AVX-512 nor AVX2" makes it load the plain one.
    _fake = types.ModuleType("proxsuite.instructionset")
    _fake.has_AVX2 = lambda: False
    _fake.has_AVX512F = lambda: False
    sys.modules["proxsuite.instructionset"] = _fake
    try:
        import proxsuite  # noqa: E402
    finally:
        del sys.modules["proxsuite.instructionset"]
else:
    import proxsuite  # noqa: E402

QP_BUILD = getattr(getattr(proxsuite, "_submodule", None), "__name__", "unknown")
QP_BUILD_IS_PLAIN = QP_BUILD.endswith("proxsuite_pywrap")
if not QP_BUILD_IS_PLAIN and not _NATIVE:
    # the selection above relies on proxsuite/__init__.py of the pinned version
    if "proxsuite" in sys.modules and _fake_used is False:
        why = "proxsuite was imported before leaptrack, so its build could not be chosen"
    else:
        why = f"proxsuite {getattr(proxsuite, '__version__', '?')} did not load its plain build"
    raise ImportError(f"leaptrack.determinism: {why} (loaded: {QP_BUILD}). Import leaptrack before "
                      "proxsuite and install the versions of tracking/requirements.txt; set "
                      "LEAPTRACK_QP_BUILD=native to run with the build proxsuite chooses.")


def describe() -> dict:
    """Recorded in metrics.json / evaluation.json."""
    return dict(qp_build=QP_BUILD, qp_build_plain=QP_BUILD_IS_PLAIN, thread_env_set_before_numpy=THREAD_ENV_OK,
                threads={k: os.environ.get(k) for k in THREAD_VARS},
                pythonhashseed=os.environ.get("PYTHONHASHSEED"))
