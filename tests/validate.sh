#!/usr/bin/env bash
# Validation tests of PROJECT_PLAN.md §5, run against the binaries in BUILD (default build/):
#   V1   manufactured solution: error ~ h^2 (serial, 2D and 3D)
#   V2a  one A*x, bitwise identical to serial for p = 1..NP_MAX, odd p, uneven splits,
#        every decomposition and halo variant
#   V2b  full CG solve: same iterations as serial, ||x_p - x_s|| / ||x_s|| <= 1e-10;
#        with --dot repro (double-double reductions) the solution must be bit-identical
#   V3   OpenMP threads reproduce V1 and V2b; the CUDA build (if built and a GPU is visible)
#        reproduces V1, V2a and V2b
# Usage: tests/validate.sh [BUILD]     env: MPIRUN (default mpirun), NP_MAX (default 8),
#        QUICK=1 for a shorter run, GPU_NP (default 2) ranks for the GPU tests.
set -u
BUILD="${1:-build}"
MPIRUN="${MPIRUN:-mpirun}"
NP_MAX="${NP_MAX:-8}"
QUICK="${QUICK:-0}"
GPU_NP="${GPU_NP:-2}"
SERIAL="$BUILD/poisson_serial"
MPI="$BUILD/poisson_mpi"
CUDA="$BUILD/poisson_cuda"
for b in "$SERIAL" "$MPI"; do
  [ -x "$b" ] || { echo "missing $b: run make first" >&2; exit 2; }
done

pass=0
fail=0
failed=()
check() {  # check <label> <command...>: runs it, keeps its last line, counts PASS/FAIL
  local label="$1"
  shift
  local out
  out="$("$@" 2>&1)"
  local rc=$?
  local last  # the result line (MPI libraries may print notices after it)
  last="$(printf '%s\n' "$out" | grep -E -- '-> (PASS|FAIL)' | tail -1)"
  if [ $rc -eq 0 ] && [[ "$last" == *PASS* ]]; then
    pass=$((pass + 1))
    echo "  ok    $last"
  else
    fail=$((fail + 1))
    failed+=("$label")
    echo "  FAIL  $label (exit $rc)"
    printf '%s\n' "$out" | tail -8 | sed 's/^/        /'
  fi
}

export OMP_NUM_THREADS=1
echo "== V1: manufactured solution, serial (observed order -> 2)"
if [ "$QUICK" = 1 ]; then n2=255; n3=63; else n2=1023; n3=255; fi
"$SERIAL" --dim 2 --test mms --max-n "$n2" | sed 's/^/  /'
check "V1 serial 2D" "$SERIAL" --dim 2 --test mms --max-n "$n2"
check "V1 serial 3D" "$SERIAL" --dim 3 --test mms --max-n "$n3"
check "V1 serial 3D fused" "$SERIAL" --dim 3 --test mms --max-n 63 --cg fused

echo "== V2a: A*x bitwise identical to serial (uneven grids, odd p)"
if [ "$QUICK" = 1 ]; then plist="1 2 3 4"; else plist="$(seq 1 "$NP_MAX")"; fi
for p in $plist; do
  for halo in sendrecv nonblocking overlap rma pscw; do
    check "V2a 3D p=$p $halo" $MPIRUN -np "$p" "$MPI" --dim 3 --nx 23 --ny 17 --nz 13 --test apply --halo "$halo"
    check "V2a 2D p=$p $halo" $MPIRUN -np "$p" "$MPI" --dim 2 --nx 37 --ny 29 --test apply --halo "$halo"
  done
  for decomp in pencil slab; do
    check "V2a 3D p=$p $decomp" $MPIRUN -np "$p" "$MPI" --dim 3 --nx 23 --ny 17 --nz 13 --test apply --decomp "$decomp" --halo overlap
  done
done

echo "== V2b: full CG solve against serial"
for p in 1 2 3 4 6 8; do
  [ "$p" -le "$NP_MAX" ] || continue
  for cg in standard fused; do
    check "V2b 3D p=$p $cg" $MPIRUN -np "$p" "$MPI" --dim 3 --nx 31 --ny 27 --nz 23 --test solve --cg "$cg"
    check "V2b 2D p=$p $cg" $MPIRUN -np "$p" "$MPI" --dim 2 --nx 129 --ny 97 --test solve --cg "$cg"
  done
done
np=$(( NP_MAX < 4 ? NP_MAX : 4 ))
check "V2b 3D p=$np overlap" $MPIRUN -np "$np" "$MPI" --dim 3 --n 33 --test solve --halo overlap
check "V2b 3D p=$np rma fused" $MPIRUN -np "$np" "$MPI" --dim 3 --n 33 --test solve --halo rma --cg fused
check "V2b 3D p=$np slab sendrecv" $MPIRUN -np "$np" "$MPI" --dim 3 --n 33 --test solve --decomp slab --halo sendrecv
check "V2b 3D p=$np pscw" $MPIRUN -np "$np" "$MPI" --dim 3 --n 33 --test solve --halo pscw

echo "== V2b with reproducible (double-double) dot products: bit-identical to serial"
for p in 2 3 5 8; do
  [ "$p" -le "$NP_MAX" ] || continue
  check "V2b repro 3D p=$p" $MPIRUN -np "$p" "$MPI" --dim 3 --nx 31 --ny 27 --nz 23 --test solve --dot repro
done
check "V2b repro 2D p=$np fused" $MPIRUN -np "$np" "$MPI" --dim 2 --nx 129 --ny 97 --test solve --dot repro --cg fused
check "V1 serial 3D repro" "$SERIAL" --dim 3 --test mms --max-n 63 --dot repro

echo "== V3: OpenMP threads"
for t in 2 4; do
  r=$(( NP_MAX / t ))
  [ "$r" -ge 1 ] || r=1
  [ "$r" -le 3 ] || r=3
  check "V3 omp t=$t mms" env OMP_NUM_THREADS=$t $MPIRUN -np 1 "$MPI" --dim 3 --test mms --max-n 63
  check "V3 omp t=$t p=$r solve" env OMP_NUM_THREADS=$t $MPIRUN -np "$r" "$MPI" --dim 3 --nx 33 --ny 29 --nz 31 --test solve --halo overlap
  check "V3 omp t=$t p=$r apply" env OMP_NUM_THREADS=$t $MPIRUN -np "$r" "$MPI" --dim 2 --nx 101 --ny 77 --test apply --halo rma
  check "V3 omp t=$t p=$r repro" env OMP_NUM_THREADS=$t $MPIRUN -np "$r" "$MPI" --dim 3 --n 33 --test solve --dot repro --halo pscw
done

if [ -x "$CUDA" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "== V3: CUDA"
  for gm in aware staged; do
    check "V3 cuda 3D apply $gm" $MPIRUN -np "$GPU_NP" "$CUDA" --dim 3 --nx 37 --ny 29 --nz 23 --test apply --gpu-mpi "$gm"
    check "V3 cuda 2D apply $gm" $MPIRUN -np "$GPU_NP" "$CUDA" --dim 2 --nx 129 --ny 97 --test apply --gpu-mpi "$gm"
    check "V3 cuda 3D solve $gm" $MPIRUN -np "$GPU_NP" "$CUDA" --dim 3 --nx 31 --ny 27 --nz 23 --test solve --gpu-mpi "$gm"
  done
  check "V3 cuda mms 3D" $MPIRUN -np 1 "$CUDA" --dim 3 --test mms --max-n 127
  check "V3 cuda mms 2D" $MPIRUN -np 1 "$CUDA" --dim 2 --test mms --max-n 511
else
  echo "== V3: CUDA skipped (no $CUDA or no visible GPU)"
fi

echo "== summary: $pass passed, $fail failed"
for f in "${failed[@]+"${failed[@]}"}"; do echo "  failed: $f"; done
[ "$fail" -eq 0 ]
