#!/usr/bin/env bash
# GPU experiment E7 (PROJECT_PLAN.md §6) with the MPI+CUDA build. Called by vanda_gpu*.pbs:
#   B     build directory containing poisson_cuda
#   CSV   output file
#   NGPU  GPUs in the job (1 or 2);  NODES  nodes in the job (default 1)
#   GPU_P GPU counts to run (default 1..NGPU)
#   LAUNCHER (multi-node jobs) "tm" to start the remote ranks with PBS's pbs_tmrsh, which keeps
#         them inside the job (its GPU and cpuset on that node); otherwise ssh
#   MPI_NET (multi-node jobs) extra mpirun options for the transport between the nodes, split on
#         spaces (see vanda_gpu_2node.pbs and cpu_sweep.sh: the module's default is unusable)
#   GPU_MODES halo modes to run (default "aware staged")
# Strong scaling on 1..NGPU GPUs and weak scaling per GPU, each with CUDA-aware and
# host-staged halo exchange, REPS repeats (default 3).
set -u
REPS="${REPS:-3}"
GPU_P="${GPU_P:-$(seq -s ' ' 1 "$NGPU")}"
GPU_MODES="${GPU_MODES:-aware staged}"
gpu() {  # gpu <ranks> <args...>
  local np=$1
  shift
  # HPC-X Open MPI has no PBS support: give it the job's nodes when there are several.
  local hosts=()
  if (( ${NODES:-1} > 1 )) && [ -n "${PBS_NODEFILE:-}" ]; then
    hosts=(--hostfile "$PBS_NODEFILE" --prefix "$(dirname "$(dirname "$(command -v mpirun)")")"
           -x PATH -x LD_LIBRARY_PATH)
    [ -n "${UCX_WARN_UNUSED_ENV_VARS:-}" ] && hosts+=(-x UCX_WARN_UNUSED_ENV_VARS)
    [ "${LAUNCHER:-ssh}" = tm ] &&
      hosts+=(--mca plm_rsh_agent /opt/pbs/bin/pbs_tmrsh --mca plm_rsh_no_tree_spawn 1)
    # shellcheck disable=SC2206  # MPI_NET is a space-separated list of options without spaces
    [ -n "${MPI_NET:-}" ] && hosts+=($MPI_NET)
  fi
  # timeout: a hung run must not use up the job's walltime
  timeout 600 mpirun -np "$np" ${hosts[@]+"${hosts[@]}"} --map-by "ppr:$(( np < 2 ? 1 : np / ${NODES:-1} )):node" \
    --bind-to none "$B/poisson_cuda" --csv "$CSV" --tag "${TAG:-}" "$@" || echo "!! run failed (exit $?): p=$np $*"
}
for gm in $GPU_MODES; do
  for p in $GPU_P; do
    echo "== E7 strong scaling on $p GPU(s), $gm"
    TAG=gpu-strong
    for rep in $(seq "$REPS"); do
      gpu "$p" --dim 3 --n 256 --iters 500 --gpu-mpi "$gm"
      gpu "$p" --dim 3 --n 512 --iters 200 --gpu-mpi "$gm"
      gpu "$p" --dim 2 --n 8192 --iters 500 --gpu-mpi "$gm"
      gpu "$p" --dim 2 --n 16384 --iters 200 --gpu-mpi "$gm"
    done
    echo "== E7 weak scaling on $p GPU(s), $gm"
    TAG=gpu-weak
    for rep in $(seq "$REPS"); do
      gpu "$p" --dim 3 --nlocal 512 --iters 200 --gpu-mpi "$gm"
      gpu "$p" --dim 2 --nlocal 16384 --iters 200 --gpu-mpi "$gm"
    done
  done
done
