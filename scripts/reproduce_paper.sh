#!/bin/bash
# Solve the ten benchmark instances (Table II), then print their table
# (scripts/make_results_table.py).
#
#   usage: [CORE_BASE=<first core>] [SNAPSHOTS=1] scripts/reproduce_paper.sh <results_root> [lanes] [tag]
#   e.g.:  scripts/reproduce_paper.sh /scratch/leap_repro 4 repro
#          CORE_BASE=64 scripts/reproduce_paper.sh results/repro 5 repro   # cores 64-123
#          SNAPSHOTS=1 scripts/reproduce_paper.sh results/repro 4 repro    # + the Fig. 3 candidates
#
# The instances run in `lanes` parallel lanes of 12 physical cores each (lane i uses
# cores CORE_BASE + 12i .. CORE_BASE + 12i + 11), longest first. Choose CORE_BASE so that
# the cores are idle physical cores.
#
# SNAPSHOTS=1: also write the budget-study candidates (Fig. 3) to
# <results_root>/<family>/snapshots_<robot>/ (`output.snapshots`).
#
# Outputs: <results_root>/<family>/<family>_<robot>_<tag>.log and
# <results_root>/<family>/sol_<family>_<robot>_<tag>.h5.
set -u
ROOT_OUT=${1:?results root}
LANES=${2:-4}
TAG=${3:-repro}
CORE_BASE=${CORE_BASE:-0}
SNAPSHOTS=${SNAPSHOTS:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
NEW_SRC=$(cd "$HERE/.." && pwd)
TABLE=$NEW_SRC/scripts/make_results_table.py
[ -x "$NEW_SRC/build/leap_solve" ] || { echo "build first: cmake --build $NEW_SRC/build" >&2; exit 2; }
mkdir -p "$ROOT_OUT/fence" "$ROOT_OUT/terrain"
echo "# reproduce_paper.sh root=$ROOT_OUT lanes=$LANES tag=$TAG core_base=$CORE_BASE snapshots=$SNAPSHOTS start $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"

# instance list, longest first: family robot threads
INSTANCES=(
  "terrain h1 12"
  "terrain talos 12"
  "terrain g1_29dof 12"
  "fence talos 12"
  "fence g1_29dof 12"
  "fence h1 12"
  "terrain go2 12"
  "fence go2 12"
  "terrain hopper2d_foot 12"
  "fence hopper2d 12"
)

run_lane() {  # lane index, instances (as "fam robot nt" strings)
  local lane=$1; shift
  local base=$((CORE_BASE + 12 * lane))
  for inst in "$@"; do
    set -- $inst
    local fam=$1 robot=$2 nt=$3
    local cpus="$base-$((base + nt - 1))"
    local extra=()
    [ "$SNAPSHOTS" = 1 ] && extra=("output.snapshots=$ROOT_OUT/$fam/snapshots_$robot")
    echo "[lane $lane] $fam $robot on cores $cpus start $(date -Is)"
    LEAP_RESULTS_DIR=$ROOT_OUT/$fam "$NEW_SRC/scripts/run_benchmark.sh" "$fam" "$robot" "$cpus" "$TAG" "${extra[@]}" > /dev/null
    echo "[lane $lane] $fam $robot done $(date -Is): $(grep -m1 '^\[unified\]' "$ROOT_OUT/$fam/${fam}_${robot}_${TAG}.log" | grep -o 'met=[01] gate=[0-9.e+-]* fine=[0-9.e+-]*.*total=[0-9]* solvewall=[0-9.]*')"
  done
}

# round-robin assignment of the instances to the lanes
declare -a LANE_LISTS
for ((i = 0; i < LANES; i++)); do LANE_LISTS[$i]=""; done
for ((k = 0; k < ${#INSTANCES[@]}; k++)); do
  lane=$((k % LANES))
  LANE_LISTS[$lane]="${LANE_LISTS[$lane]}|${INSTANCES[$k]}"
done
for ((i = 0; i < LANES; i++)); do
  IFS='|' read -r -a items <<< "${LANE_LISTS[$i]#|}"
  run_lane "$i" "${items[@]}" &
done
wait
echo "# all instances done $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
echo
python3 "$TABLE" "$ROOT_OUT" --tag "$TAG"
