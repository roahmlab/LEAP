#!/usr/bin/env bash
set -euo pipefail
comparison_root="$(cd "$(dirname "$0")/.." && pwd)"
source_dir="${CROCODDYL_BUILD_ROOT:-$comparison_root/../../build}/crocoddyl_unified_stock_3_2_1"
if ! test -d "$source_dir"; then
 git clone --depth 1 --branch v3.2.1 --recurse-submodules --shallow-submodules https://github.com/loco-3d/crocoddyl.git "$source_dir"
fi
test "$(git -C "$source_dir" rev-parse HEAD)" = 174c6082f0aef4dca80fa767f47fe025d6c4fef1
# The same two Eigen-5 compatibility expressions used by conda-forge. No
# optimization code is changed. Applying twice is harmless after verification.
if git -C "$source_dir" apply --check "$comparison_root/crocoddyl_unified/eigen5-packaging.patch" 2>/dev/null; then
 git -C "$source_dir" apply "$comparison_root/crocoddyl_unified/eigen5-packaging.patch"
else
 git -C "$source_dir" apply --reverse --check "$comparison_root/crocoddyl_unified/eigen5-packaging.patch"
fi
cmake -S "$source_dir" -B "${CROCODDYL_BUILD_ROOT:-$comparison_root/../../build}/crocoddyl_unified_release_mt" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${CROCODDYL_ENV_PREFIX:-$comparison_root/../../env}" \
 -DCMAKE_CXX_COMPILER="${CROCODDYL_ENV_PREFIX:-$comparison_root/../../env}/bin/c++" \
 -DCMAKE_C_COMPILER="${CROCODDYL_ENV_PREFIX:-$comparison_root/../../env}/bin/cc" \
 -DCMAKE_INSTALL_PREFIX="${CROCODDYL_BUILD_ROOT:-$comparison_root/../../build}/crocoddyl_unified_install" \
 -DPYTHON_EXECUTABLE="${CROCODDYL_ENV_PREFIX:-$comparison_root/../../env}/bin/python" \
 -DBUILD_TESTING=OFF -DBUILD_BENCHMARK=OFF -DBUILD_EXAMPLES=OFF \
 -DBUILD_WITH_ODYN=OFF -DBUILD_WITH_IPOPT=OFF -DBUILD_WITH_CODEGEN_SUPPORT=OFF \
 -DBUILD_PYTHON_INTERFACE=OFF -DBUILD_WITH_MULTITHREADS=ON -DBUILD_WITH_NTHREADS=12
cmake --build "${CROCODDYL_BUILD_ROOT:-$comparison_root/../../build}/crocoddyl_unified_release_mt" --target crocoddyl -j12
