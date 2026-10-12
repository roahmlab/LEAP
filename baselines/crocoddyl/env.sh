# Sourced by setup.sh, run.sh and run_all.sh. Set any of these before calling them to override.
#   CROCODDYL_ENV_PREFIX  conda environment created from conda-linux-64.lock (or environment.yml)
#   CROCODDYL_BUILD_ROOT  holds crocoddyl_unified_release_mt/libcrocoddyl.so (built by setup.sh)
#   CROCODDYL_STAGE_ROOT  terrain only: directory for the per-iteration archive during a solve
#                         (default in run.sh: the parent directory of the run's output folder)
_croco_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CROCODDYL_ENV_PREFIX="${CROCODDYL_ENV_PREFIX:-$_croco_here/env}"
export CROCODDYL_BUILD_ROOT="${CROCODDYL_BUILD_ROOT:-$_croco_here/build}"
unset _croco_here
