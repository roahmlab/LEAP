#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
exec "$here/../crocoddyl_unified/run_python.sh" "$@"
