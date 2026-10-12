#!/usr/bin/env bash
# The isolated stock Crocoddyl 3.2.1 OpenMP build (comparisons/crocoddyl_unified/setup_stock.sh).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/../crocoddyl_unified/run_python.sh" "$@"
