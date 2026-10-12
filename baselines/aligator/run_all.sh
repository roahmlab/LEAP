#!/usr/bin/env bash
# Run the ten instances one after the other with run.sh: <root>/<task>/<robot>/, log <root>/<task>/<robot>.log
#
#   usage: run_all.sh <root> [--cpus LIST] [--python PATH]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[ $# -ge 1 ] || { sed -n '4p' "$0" >&2; exit 2; }
ROOT=$1; shift
status=0
for case in terrain:hopper2d_foot terrain:go2 terrain:h1 terrain:g1_29dof terrain:talos \
            fence:hopper2d fence:go2 fence:h1 fence:g1_29dof fence:talos; do
  task=${case%%:*}; robot=${case#*:}
  mkdir -p "$ROOT/$task"
  if "$HERE/run.sh" "$task" "$robot" "$ROOT/$task/$robot" "$@" > "$ROOT/$task/$robot.log" 2>&1; then
    echo "$(date -Is) $task $robot: done"
  else
    echo "$(date -Is) $task $robot: FAILED (see $ROOT/$task/$robot.log)"; status=1
  fi
done
exit $status
