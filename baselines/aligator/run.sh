#!/usr/bin/env bash
# Run one Aligator instance and its offline audit into a new run folder.
#
#   usage: run.sh <fence|terrain> <robot> <out_dir> [--cpus LIST] [--python PATH]
#     fence robots:   hopper2d go2 h1 g1_29dof talos
#     terrain robots: hopper2d_foot go2 h1 g1_29dof talos
#     --cpus    12 logical CPUs on 12 distinct physical cores, e.g. 36-47
#               (default: the first 12 physical cores of the current CPU affinity)
#     --python  the environment's interpreter (default: $ALIGATOR_PYTHON, else `python` on PATH)
#
# <out_dir> must not exist; it becomes the run folder that leaptrack reads (track.py --run-dir).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/src
usage() { sed -n '4,10p' "$0" >&2; exit 2; }
[ $# -ge 3 ] || usage
TASK=$1; ROBOT=$2; OUT=$3; shift 3
CPUS=""; PY=${ALIGATOR_PYTHON:-python}
while [ $# -gt 0 ]; do
  case $1 in
    --cpus) CPUS=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    *) usage ;;
  esac
done
[ -e "$OUT" ] && { echo "output folder exists: $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"
OUT=$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")
if [ -z "$CPUS" ]; then
  CPUS=$("$PY" - <<'EOF'
import os
from pathlib import Path
seen, cpus = set(), []
for cpu in sorted(os.sched_getaffinity(0)):
    top = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    key = ((top / 'physical_package_id').read_text(), (top / 'core_id').read_text())
    if key not in seen:
        seen.add(key); cpus.append(cpu)
print(','.join(map(str, cpus[:12])))
EOF
)
fi
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=FALSE
echo "$(date -Is) $TASK $ROBOT -> $OUT (cpus $CPUS, python $PY)"

if [ "$TASK" = terrain ]; then
  case $ROBOT in
    hopper2d_foot|go2|h1|talos) DATASET=$ROBOT ;;
    g1_29dof) DATASET=g1_29dof_repaired ;;
    *) echo "unknown terrain robot: $ROBOT" >&2; exit 2 ;;
  esac
  T=$SRC/comparisons/aligator_unified_terrain/terrain_benchmark_along_the_way
  # all phases, then the offline audit (postprocess.py), on the 12 given CPUs
  "$PY" "$T/run.py" --robot "$DATASET" --cpus "$CPUS" --out "$OUT"
  echo "$(date -Is) done: $OUT"
  exit 0
fi

[ "$TASK" = fence ] || usage
case $ROBOT in hopper2d|go2|h1|g1_29dof|talos) ;; *) echo "unknown fence robot: $ROBOT" >&2; exit 2 ;; esac
F=$SRC/comparisons/aligator_unified_fence/fence_benchmark_along_the_way
on_cpus() { taskset -c "$CPUS" "$PY" "$@"; }
field() {  # field <python expression on the summary dict s>
  "$PY" -c "import json,sys; s=json.load(open(sys.argv[1])); print($1)" "$OUT/summary.json"
}
keep_going() {  # the returned candidate does not pass and solve budget remains
  [ "$(field "bool(s['returned_audit'].get('passed')) or s['solve_compute_s']>=14400")" = False ]
}
case $ROBOT in h1) CALLS1=5; CALLS2=10 ;; *) CALLS1=4; CALLS2=8 ;; esac
CALLS3=(); [ "$ROBOT" = talos ] && CALLS3=(--max-calls 12)

# phase 00: cold start from the seed archive, at most 4 calls of 300 s within 1200 s
set +e
on_cpus "$F/run.py" --robot "$ROBOT" --out "$OUT" --core-offset 0 --budget 1200 --call-seconds 300 \
  --rounds 4 --iterations 10000 --dt 0.005 --no-postprocess
code=$?
set -e
[ -f "$OUT/summary.json" ] || { echo "phase 00 failed with exit $code" >&2; exit 1; }
# a non-finite returned primal ends phase 00 (exit 1); its best finite sample is selected by the audit
if [ "$(field "bool(s['returned_audit'].get('finite', False))")" = False ]; then
  on_cpus "$F/postprocess.py" "$OUT" --workers 12
fi
# phase 01: total budget 14400 s; outer augmented Lagrangian, dual step 0.25, geometry weight 1
# (a robot whose phase 00 passed only gets its budget raised to 14400 s)
on_cpus "$F/extend_run.py" "$OUT" --budget 14400 --max-calls "$CALLS1" --no-postprocess
# phase 02: geometry weight 100
if keep_going; then
  on_cpus "$F/extend_run.py" "$OUT" --budget 14400 --geometry-weight 100 --max-calls "$CALLS2" --no-postprocess
fi
# phase 03: geometry weight 100, dual step 1, continuing the phase-02 multipliers
if keep_going; then
  on_cpus "$F/extend_run.py" "$OUT" --budget 14400 --geometry-weight 100 --dual-step 1 --continue-duals \
    ${CALLS3[@]+"${CALLS3[@]}"} --no-postprocess
fi
# offline audit of every stored iterate
on_cpus "$F/postprocess.py" "$OUT" --workers 12
echo "$(date -Is) done: $OUT"
