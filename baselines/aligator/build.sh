#!/usr/bin/env bash
# One-time setup, after creating the conda environment:
#   1. restore the seed archives from their .h5.gz copies;
#   2. compile the native Python extensions in place under src/ (logs in build_logs/).
#
#   usage:  ALIGATOR_PYTHON=<prefix>/bin/python ./build.sh        (default: `python` on PATH)
#
# Compilers: $CC / $CXX, default <prefix>/bin/cc and <prefix>/bin/c++ (the environment's GCC).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/src
PY=${ALIGATOR_PYTHON:-python}
PREFIX=$("$PY" -c 'import sys; print(sys.prefix)')
export CC=${CC:-$PREFIX/bin/cc}
export CXX=${CXX:-$PREFIX/bin/c++}
for tool in "$CC" "$CXX"; do
  command -v "$tool" >/dev/null || { echo "compiler not found: $tool (install gxx in the environment or set CC/CXX)" >&2; exit 1; }
done
"$PY" -c 'import aligator, pinocchio, casadi, coal, h5py, yaml; print("aligator", aligator.__version__, "| pinocchio", pinocchio.__version__, "| casadi", casadi.__version__)'

echo "== 1. seed archives"
while IFS= read -r gz; do
  h5=${gz%.gz}
  if [ ! -f "$h5" ]; then gzip -dc "$gz" > "$h5.tmp" && mv "$h5.tmp" "$h5"; fi
  echo "  ok ${h5#$HERE/}"
done < <(find "$SRC" -name 'seed_*.h5.gz' | sort)

mkdir -p "$HERE/build_logs"
echo "== 2. native extensions (CC=$CC CXX=$CXX)"
build() {  # build <dir> <setup script>
  local dir=$1 setup=$2
  echo "  $dir/$setup"
  local log=$HERE/build_logs/$(echo "$dir" | tr / _)_${setup%.py}.log
  (cd "$SRC/$dir" && "$PY" "$setup" build_ext --inplace > "$log" 2>&1) \
    || { echo "build failed, see $log" >&2; exit 1; }
}
# shared residual adapters (Python package aligator_bench)
build comparisons/aligator_bench setup_sparse.py
build comparisons/aligator_bench setup_hybrid.py
# fence formulation models and the capped solver entry point (the terrain driver uses them too)
build comparisons/aligator_unified_fence/scripts setup_public_models.py
build comparisons/aligator_unified_fence/scripts setup_public_run.py
# iterate-capturing wrappers around SolverFDDP.run (fence and terrain experiments)
build comparisons/aligator_unified_fence/fence_benchmark_along_the_way setup_trace.py
build comparisons/aligator_unified_terrain/along_the_way setup_trace.py
# terrain formulation model (quadratic torque / force holds)
build comparisons/aligator_unified_terrain/scripts setup_terrain_models.py
echo "build complete"
