#!/bin/bash
# One cold solve of one benchmark instance (Table II), with the per-family transcription
# settings below.
#
#   usage: scripts/run_benchmark.sh <fence|terrain> <robot> <cpulist> [tag] [dotted overrides...]
#   e.g.:  scripts/run_benchmark.sh fence   go2 0-11            # -> results/fence/sol_fence_go2.h5 + fence_go2.log
#          scripts/run_benchmark.sh terrain h1  12-23 retry     # -> results/terrain/sol_terrain_h1_retry.h5
#
# Problem file:  problems/<robot>_jump_fence.yaml | problems/<robot>_terrain_walk.yaml
#                (the yamls carry the same per-family settings themselves).
# Method knobs:  the `unified:` defaults compiled into the solver (solver/recipe.hpp);
#                `unified.key=value` overrides them.
# Threads:       the size of <cpulist>, pinned with taskset + OMP_PLACES=cores.
# Output:        results/<family>/<family>_<robot>[_<tag>].log and the archive
#                results/<family>/sol_<family>_<robot>[_<tag>].h5 (LEAP_RESULTS_DIR overrides
#                the directory). The summary line of the log starts with "[unified]".
set -u
FAM=$1; ROBOT=$2; CPUS=$3; TAG=${4:-}; shift 3; [ $# -gt 0 ] && shift
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
BIN=${LEAP_SOLVE_BIN:-$ROOT/build/leap_solve}
[ -x "$BIN" ] || { echo "leap_solve binary not found at $BIN (build it: cmake --build build)" >&2; exit 2; }
NT=$(python3 -c "
s='$CPUS'; n=0
for p in s.split(','):
    a,b=(p.split('-')+[p])[:2]; n+=int(b)-int(a)+1
print(n)")
case "$FAM" in
  fence)
    YAML=$ROOT/problems/${ROBOT}_jump_fence.yaml
    # 8 graded (0.6) degree-4 elements per mode, LS dynamics-defect cost on 8 points per
    # element at rho 3e6, exact rounded capsule-box distance, judged on the dt = 0.01 s
    # grid (dt = 0.001 s reported only).
    BASE="solver.split=8 solver.hgrade=0.6 solver.lsqnqs=8,8,8 solver.rho=3000000 obstacles.smooth=0 obstacles.rounded=true output.finegrid_dt=0.01 output.finegrid_dt2=0.001";;
  terrain)
    YAML=$ROOT/problems/${ROBOT}_terrain_walk.yaml
    # the yaml mesh (one degree-8 element per mode) + one movable attitude and joint-box
    # point, two torque-box points, stance acceleration at the no-slip points,
    # Bernstein-hull (degree 16) torque box for explicit-torque robots.
    BASE="output.finegrid_dt=0.01 output.internode_trace=0 problem.attnq=1 problem.qboxnq=1 problem.uboxnq=2 problem.noslipacc=true problem.uhull=16";;
  *) echo "family must be fence or terrain" >&2; exit 2;;
esac
[ -f "$YAML" ] || { echo "no problem file $YAML" >&2; exit 2; }
OUTDIR=${LEAP_RESULTS_DIR:-$ROOT/results/$FAM}
mkdir -p "$OUTDIR"
STEM=sol_${FAM}_${ROBOT}${TAG:+_$TAG}
LOG=$OUTDIR/${FAM}_${ROBOT}${TAG:+_$TAG}.log
export OMP_NUM_THREADS=$NT OMP_PROC_BIND=close OMP_PLACES=cores OPENBLAS_NUM_THREADS=1
echo "# leap_solve fam=$FAM robot=$ROBOT tag=${TAG:-base} cpus=$CPUS threads=$NT start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" > "$LOG"
echo "# cmd: $BIN $YAML output.stem=$OUTDIR/$STEM $BASE $*" >> "$LOG"
/usr/bin/time -f "RUSAGE user=%U sys=%S wall=%e maxrss_kb=%M" taskset -c "$CPUS" stdbuf -oL "$BIN" "$YAML" output.stem="$OUTDIR/$STEM" $BASE "$@" >> "$LOG" 2>&1
RC=$?
echo "rc=$RC end $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$LOG"
grep -m1 '^\[unified\]' "$LOG" || echo "no [unified] summary line in $LOG (rc=$RC)"
exit $RC
