#!/usr/bin/env bash
# E1 serial scaling (report Section 2) on one core, plus the p = 1 MPI baselines of the
# strong-scaling problems. Called by cpu_sweep.sh (WHAT=serial) inside a job that has the
# whole node to itself, so the timings are not shared with other jobs' memory traffic. Needs:
#   B    build directory containing poisson_serial (and poisson_mpi for the baselines)
#   CSV  output file (one row per run)
# Fixed-iteration runs choose K so each takes ~10 s (K = 2e9/n, 20 <= K <= 2000), 3 repeats;
# time-to-solution runs use --tol 1e-8.
# The serial binary is pinned to core 0, as `mpirun -np 1` pins the MPI baseline: unpinned,
# the OS sometimes moved it away from the NUMA domain holding its memory, and those runs were
# 30-40% slower (Atlas job 1038236).
set -u
REPS="${REPS:-3}"
S="$B/poisson_serial"
PIN=""
command -v taskset >/dev/null 2>&1 && PIN="taskset -c 0"
run_serial() { $PIN "$S" "$@"; }

kfor() {  # iterations for n unknowns
  local k=$(( 2000000000 / $1 ))
  (( k < 20 )) && k=20
  (( k > 2000 )) && k=2000
  echo "$k"
}
wfor() { (( $1 >= 100 )) && echo 10 || echo 2; }

echo "== E1 fixed iterations, 2D"
for N in 128 256 512 1024 2048 4096 8192 16384; do
  K=$(kfor $((N * N)))
  for rep in $(seq "$REPS"); do run_serial --dim 2 --n "$N" --iters "$K" --warmup "$(wfor "$K")" --csv "$CSV" --tag e1; done
done
echo "== E1 fixed iterations, 3D"
for N in 16 32 64 128 256 512; do
  K=$(kfor $((N * N * N)))
  for rep in $(seq "$REPS"); do run_serial --dim 3 --n "$N" --iters "$K" --warmup "$(wfor "$K")" --csv "$CSV" --tag e1; done
done
echo "== E1 time to solution (tol 1e-8)"
for N in 128 256 512 1024 2048; do run_serial --dim 2 --n "$N" --tol 1e-8 --csv "$CSV" --tag e1-tol; done
for N in 16 32 64 128 256; do run_serial --dim 3 --n "$N" --tol 1e-8 --csv "$CSV" --tag e1-tol; done

if [ "${WITH_MPI1:-1}" = 1 ] && [ -x "$B/poisson_mpi" ]; then
  echo "== p = 1 MPI baselines of the strong-scaling problems (should match serial within a few %)"
  for spec in "3 256" "3 512" "2 8192" "2 16384"; do
    set -- $spec
    n=$(( $1 == 3 ? $2 * $2 * $2 : $2 * $2 ))
    K=$(( 200000000 / n )); (( K < 20 )) && K=20
    for rep in $(seq "$REPS"); do
      run_serial --dim "$1" --n "$2" --iters "$K" --warmup 2 --csv "$CSV" --tag baseline
      OMP_NUM_THREADS=1 mpirun -np 1 "$B/poisson_mpi" --dim "$1" --n "$2" --iters "$K" --warmup 2 --csv "$CSV" --tag baseline
    done
  done
fi
