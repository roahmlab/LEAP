#!/usr/bin/env bash
# Run Crocoddyl on one benchmark instance.
#
#   ./run.sh <fence|terrain> <robot> <out_dir> [--cpus LIST] [solve.py options]
#
#   robot     fence:   hopper2d go2 h1 g1_29dof talos
#             terrain: hopper2d_foot go2 h1 g1_29dof talos
#   out_dir   new directory for the run (must not exist); the logs go to <out_dir>.log
#             (and <out_dir>.postprocess.log for the terrain)
#   --cpus    1 to 12 distinct physical cores, e.g. 24-35 or 0,2,4 (default: the first 12
#             physical cores of the current CPU affinity)
#   any other option is passed to solve.py after the defaults, e.g. --solver-seconds 600
#
# Defaults: --solver-seconds 14400 --wall-seconds 86400. The terrain run is followed by
# postprocess.py (offline audit of every stored candidate) on the same cores.
# Exit status: the solver's (0 = physical acceptance gate met, 2 = stopped without meeting it); 3 = postprocess.py failed.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
source "$here/env.sh"
if [ $# -lt 3 ]; then sed -n '2,16p' "$0"; exit 2; fi
task="$1"; robot="$2"; out="$3"; shift 3
cpus=""
extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --cpus) cpus="$2"; shift 2 ;;
    --cpus=*) cpus="${1#*=}"; shift ;;
    *) extra+=("$1"); shift ;;
  esac
done
case "$task" in
  fence)   impl="$here/fence/comparisons/crocoddyl_impact_cone"; robots="hopper2d go2 h1 g1_29dof talos"
           budget=(--solver-seconds 14400 --wall-seconds 86400) ;;
  terrain) impl="$here/terrain/comparisons/crocoddyl_uneven_terrain_along_the_way"; robots="hopper2d_foot go2 h1 g1_29dof talos"
           budget=(--solver-seconds 14400.0 --wall-seconds 86400.0) ;;
  *) echo "task must be fence or terrain" >&2; exit 2 ;;
esac
case " $robots " in *" $robot "*) ;; *) echo "$task robots: $robots" >&2; exit 2 ;; esac
py="$CROCODDYL_ENV_PREFIX/bin/python"
# CPU list: expand ranges; default = first 12 physical cores of the affinity mask
cpus="$("$py" - "$cpus" <<'PYEOF'
import os, sys
from pathlib import Path
spec = sys.argv[1]
if spec:
    out = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
else:
    seen, out = set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        top = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((top / 'physical_package_id').read_text(), (top / 'core_id').read_text())
        if key not in seen:
            seen.add(key); out.append(cpu)
        if len(out) == 12:
            break
print(','.join(map(str, out)))
PYEOF
)"
ncpu=$(echo "$cpus" | tr ',' '\n' | wc -l)
if [ -e "$out" ]; then echo "$out exists" >&2; exit 2; fi
mkdir -p "$(dirname "$out")"
out="$(cd "$(dirname "$out")" && pwd)/$(basename "$out")"
cmd=("$impl/run_python.sh" "$impl/solve.py" --robot "$robot" --output "$out" --cpus "$cpus" "${budget[@]}")
if [ "$task" = terrain ]; then cmd+=(--stage-root "${CROCODDYL_STAGE_ROOT:-$(dirname "$out")}"); fi
cmd+=(${extra[@]+"${extra[@]}"})
echo "${cmd[*]}" | tee "$out.log"
set +e
"${cmd[@]}" >> "$out.log" 2>&1
code=$?
set -e
echo "solve.py exit $code ($out.log)"
if [ "$task" = terrain ] && { [ "$code" = 0 ] || [ "$code" = 2 ]; }; then
  pp=(taskset -c "$cpus" "$impl/run_python.sh" "$impl/postprocess.py" "$out" --workers "$ncpu")
  echo "${pp[*]}" | tee "$out.postprocess.log"
  "${pp[@]}" >> "$out.postprocess.log" 2>&1 || { echo "postprocess.py failed ($out.postprocess.log)" >&2; exit 3; }
  echo "postprocess.py done ($out.postprocess.log)"
fi
exit "$code"
