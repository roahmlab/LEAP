#!/usr/bin/env bash
# Run the ten instances one after the other.
#
#   ./run_all.sh <out_root> [--cpus LIST] [solve.py options]
#
# Writes <out_root>/<task>/<robot>/ (+ .log, .postprocess.log) for task = terrain, fence and
# <out_root>/run_all.tsv (task, robot, exit status, start and end time). An instance whose
# output folder already exists is skipped. Options are passed to every ./run.sh call.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
if [ $# -lt 1 ]; then sed -n '2,8p' "$0"; exit 2; fi
root="$1"; shift
mkdir -p "$root"
tsv="$root/run_all.tsv"
[ -f "$tsv" ] || printf 'task\trobot\texit\tstart_utc\tend_utc\n' > "$tsv"
for item in terrain:hopper2d_foot terrain:go2 terrain:h1 terrain:g1_29dof terrain:talos \
            fence:hopper2d fence:go2 fence:h1 fence:g1_29dof fence:talos; do
  task="${item%%:*}"; robot="${item#*:}"
  out="$root/$task/$robot"
  if [ -e "$out" ]; then echo "skip $task $robot: $out exists"; continue; fi
  start="$(date -u +%FT%TZ)"
  echo "start $task $robot $start"
  "$here/run.sh" "$task" "$robot" "$out" "$@"
  code=$?
  end="$(date -u +%FT%TZ)"
  printf '%s\t%s\t%s\t%s\t%s\n' "$task" "$robot" "$code" "$start" "$end" >> "$tsv"
  echo "end $task $robot exit=$code $end"
done
column -t -s $'\t' "$tsv"
