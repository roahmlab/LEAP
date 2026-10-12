#!/usr/bin/env bash
# One-time setup of the Crocoddyl baseline.
#
#   ./setup.sh [--create-env] [--skip-stock]
#
#   --create-env   create the conda environment at $CROCODDYL_ENV_PREFIX from conda-linux-64.lock
#                  (needs conda or mamba on PATH, or CONDA_EXE)
#   --skip-stock   do not build the stock Crocoddyl library (it must already exist at
#                  $CROCODDYL_BUILD_ROOT/crocoddyl_unified_release_mt/libcrocoddyl.so)
#
# Steps: 1. unpack the seed archives (*.h5.xz -> *.h5);
#        2. build stock Crocoddyl 3.2.1 with OpenMP (comparisons/crocoddyl_unified/setup_stock.sh;
#           needs git, cmake, make and network access to github.com);
#        3. build the native model modules of fence/ and terrain/ with the environment's compilers;
#        4. import every module the solvers load.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/env.sh"
create_env=0
skip_stock=0
for arg in "$@"; do
  case "$arg" in
    --create-env) create_env=1 ;;
    --skip-stock) skip_stock=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done
prefix="$CROCODDYL_ENV_PREFIX"
py="$prefix/bin/python"

# ---------------------------------------------------------------- 0. environment
if [ "$create_env" = 1 ]; then
  conda_exe="${CONDA_EXE:-$(command -v mamba || command -v conda || true)}"
  if [ -z "$conda_exe" ]; then echo "conda/mamba not found (set CONDA_EXE)" >&2; exit 1; fi
  "$conda_exe" create -y -p "$prefix" --file "$here/conda-linux-64.lock"
fi
if [ ! -x "$py" ]; then
  echo "no python at $py: set CROCODDYL_ENV_PREFIX or run ./setup.sh --create-env" >&2
  exit 1
fi
echo "environment: $prefix"

# ---------------------------------------------------------------- 1. seed archives
echo "[1/4] unpacking seed archives"
(cd "$here" && "$py" - <<'PYEOF'
import lzma
from pathlib import Path
for archive in sorted(Path('.').glob('*/**/seed_*.h5.xz')):
    target = archive.with_suffix('')
    if not target.exists():
        target.write_bytes(lzma.decompress(archive.read_bytes()))
    print(f'  ok {target}')
PYEOF
)

# ---------------------------------------------------------------- 2. stock Crocoddyl
stock="$CROCODDYL_BUILD_ROOT/crocoddyl_unified_release_mt/libcrocoddyl.so"
if [ -f "$stock" ]; then
  echo "[2/4] stock Crocoddyl present: $stock"
elif [ "$skip_stock" = 1 ]; then
  echo "[2/4] --skip-stock given but $stock is missing" >&2
  exit 1
else
  echo "[2/4] building stock Crocoddyl 3.2.1 into $CROCODDYL_BUILD_ROOT"
  mkdir -p "$CROCODDYL_BUILD_ROOT"
  bash "$here/fence/comparisons/crocoddyl_unified/setup_stock.sh"
fi

# ---------------------------------------------------------------- 3. native modules
echo "[3/4] building native modules"
export CC="$prefix/bin/cc" CXX="$prefix/bin/c++"
for tree in fence terrain; do
  c="$here/$tree/comparisons"
  echo "  $tree: _impact_native"
  bash "$c/crocoddyl_impact_cone/build.sh" > "$c/crocoddyl_impact_cone/build.log" 2>&1
  echo "  $tree: _public_models"
  "$py" "$c/aligator_unified_fence/scripts/setup_public_models.py" build_ext --inplace > "$c/aligator_unified_fence/scripts/build.log" 2>&1
  echo "  $tree: aligator_bench._generated_sparse"
  "$py" "$c/aligator_bench/setup_sparse.py" build_ext --inplace > "$c/aligator_bench/build.log" 2>&1
  if [ "$tree" = terrain ]; then
    echo "  $tree: _terrain_models, _terrain_acceleration"
    "$py" "$c/aligator_unified_terrain/scripts/setup_terrain_models.py" build_ext --inplace > "$c/aligator_unified_terrain/scripts/build_models.log" 2>&1
    "$py" "$c/aligator_unified_terrain/scripts/setup_terrain_acceleration.py" build_ext --inplace > "$c/aligator_unified_terrain/scripts/build_acceleration.log" 2>&1
  fi
done

# ---------------------------------------------------------------- 4. import check
echo "[4/4] import check"
"$here/fence/comparisons/crocoddyl_impact_cone/run_python.sh" -B -c "
import sys; sys.path.insert(0, '$here/fence/comparisons/crocoddyl_impact_cone')
import physics, trace, recheck, crocoddyl
from _impact_native import run
print('  fence ok: crocoddyl', crocoddyl.__version__, [l.split()[-1] for l in open('/proc/self/maps') if 'libcrocoddyl.so' in l][0])
"
"$here/terrain/comparisons/crocoddyl_uneven_terrain_along_the_way/run_python.sh" -B -c "
import sys; sys.path.insert(0, '$here/terrain/comparisons/crocoddyl_uneven_terrain_along_the_way')
import physics, trace, tw_verify, crocoddyl
from _impact_native import run
print('  terrain ok: crocoddyl', crocoddyl.__version__, [l.split()[-1] for l in open('/proc/self/maps') if 'libcrocoddyl.so' in l][0])
"
echo "setup complete"
