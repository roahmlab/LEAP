#!/usr/bin/env bash
# Build modeling adapters only. This script never runs an optimizer.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
prefix="${CROCODDYL_ENV_PREFIX:-$here/../../../env}"
export CC="$prefix/bin/cc" CXX="$prefix/bin/c++"
"$prefix/bin/python" "$here/setup_native.py" build_ext --inplace
