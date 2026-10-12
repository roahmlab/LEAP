#!/usr/bin/env bash
# Isolated stock OpenMP build; do not replace other experiments' libraries.
set -euo pipefail
comparison_root="$(cd "$(dirname "$0")/.." && pwd)"
native_build="${CROCODDYL_BUILD_ROOT:-$comparison_root/../../build}/crocoddyl_unified_release_mt"
test -f "$native_build/libcrocoddyl.so"
export LD_PRELOAD="$native_build/libcrocoddyl.so${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="$native_build:${CROCODDYL_ENV_PREFIX:-$comparison_root/../../env}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=FALSE
exec "${CROCODDYL_ENV_PREFIX:-$comparison_root/../../env}/bin/python" "$@"
