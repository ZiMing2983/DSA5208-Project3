#!/usr/bin/env bash
# CPU experiments E3-E6 (report Sections 6 and 7.1) for one node count. Called by the *_cpu_*.pbs
# scripts with:
#   B       build directory containing poisson_mpi
#   CSV     output file (one row per run)
#   NODES   nodes in the job;  CORES  cores per node
#   SOCKETS sockets per node the job may use: 2 (default) for whole nodes, 1 when each node gives
#           the job only CORES = 36 cores (vanda_cpu_8socket.pbs)
#   WHAT    experiments to run (default "strong weak comm hybrid"); "serial" adds E1 and the
#           p = 1 baselines (serial_sweep.sh). Use it only in a single-node job that owns the
#           whole node: the solver is memory-bound, and on a shared node other jobs take
#           memory bandwidth (on Atlas's shared serial queue this made T1 about 3x too slow).
#   P_STRONG / P_WEAK / P_COMM   rank counts (defaults depend on NODES, see below)
# Every configuration is repeated REPS times (default 3). Iterations: strong scaling uses
# K = 2e8 * p / n (50..1000) so short runs still take long enough to time; weak scaling
# uses K = 100.
# Placement of pure-MPI runs: on one node the ranks are spread round-robin over the NUMA
# domains (every rank gets its share of the memory channels); on several whole nodes, half of each
# node's ranks go to each socket; with SOCKETS=1, a node's ranks fill the cores PBS gave the job
# there. Hybrid runs give every rank T consecutive cores.
# Lock for jobs of 256 or more cores: two job shapes may be queued for the same core count (four
# whole nodes, vanda_cpu_4node.pbs, and one socket on each of eight nodes, vanda_cpu_8socket.pbs),
# and only the first to start should run the sweep. A job stops at once if the lock
# results/.claim_p<cores> exists, and takes it (mkdir, atomic on NFS) once its multi-node check has
# passed; the job id and start time go in the file "job" inside it. Remove the directory to run
# that sweep again.
set -u
# An absolute path, so that ranks started on other nodes find the binary whatever their directory
B="$(cd "$B" && pwd)" || exit 1
REPS="${REPS:-3}"
WHAT="${WHAT:-strong weak comm hybrid}"
TOTAL=$(( NODES * CORES ))
pow2_upto() { local p=1 out=""; while (( p <= $1 )); do out="$out $p"; p=$(( p * 2 )); done; echo $out; }
if (( NODES == 1 )); then
  P_STRONG="${P_STRONG:-$(pow2_upto "$CORES") $CORES}"
  P_WEAK="${P_WEAK:-$(pow2_upto "$CORES") $CORES}"
else
  P_STRONG="${P_STRONG:-$(( NODES * 64 )) $TOTAL}"
  P_WEAK="${P_WEAK:-$(( NODES * 64 )) $TOTAL}"
fi
P_COMM="${P_COMM:-$(( NODES * 64 < TOTAL ? NODES * 64 : TOTAL ))}"
SOCKETS="${SOCKETS:-2}"
# Placement for runs on all the job's cores (the multi-node check and the OSU Allreduce)
if (( SOCKETS == 2 )); then FULL_MAP="ppr:$(( CORES / 2 )):socket"; else FULL_MAP="ppr:${CORES}:node"; fi

CLAIM=""
if (( NODES > 1 && TOTAL >= 256 )) && [ -n "${PBS_JOBID:-}" ]; then
  CLAIM="$(dirname "$CSV")/.claim_p$TOTAL"
  if [ -d "$CLAIM" ]; then
    echo "== the p = $TOTAL sweep already ran or is running (lock $CLAIM: $(cat "$CLAIM/job" 2>/dev/null)): nothing to do"
    exit 0
  fi
fi

# Vanda's HPC-X Open MPI has no PBS (tm) support: on several nodes it must be given the job's
# nodes in a hostfile, and it starts its daemons there with an rsh-style launcher. The preferred
# one is PBS's pbs_tmrsh, which starts them through PBS like a tm launch, inside the job's cgroup
# (its cpusets, devices and accounting); plain ssh is the fallback.
# The transport between nodes: the module's default (ob1 + TCP BTL, UCX given no network device)
# takes 5 ms per message between nodes, and its UCC collectives hang across nodes (net_test.pbs,
# job 1550775). "ucx" (UCX point-to-point, RMA and collectives over TCP on the fast Ethernet
# interface) measured 22 us and passed every check there; "ob1" (the TCP BTL with its progress
# thread, 31 us, and UCX given the interface so that UCC works) is the fallback.
# set_hosts builds the mpirun arguments for one launcher and transport; the multi-node check below
# picks them. Single-node jobs use none of this: the module's defaults (ob1 + shared memory).
HOSTS=()
LAUNCHER=local
TRANSPORT=default
TMRSH="${TMRSH:-/opt/pbs/bin/pbs_tmrsh}"
NET_IF="${VANDA_NET_IF:-ens6f1}"
set_hosts() {  # set_hosts tm|ssh ucx|ob1
  HOSTS=(--hostfile "$PBS_NODEFILE" --prefix "$(dirname "$(dirname "$(command -v mpirun)")")"
         -x PATH -x LD_LIBRARY_PATH -x OMP_NUM_THREADS -x OMP_PLACES -x OMP_PROC_BIND)
  [ -n "${UCX_WARN_UNUSED_ENV_VARS:-}" ] && HOSTS+=(-x UCX_WARN_UNUSED_ENV_VARS)
  [ "$1" = tm ] && HOSTS+=(--mca plm_rsh_agent "$TMRSH" --mca plm_rsh_no_tree_spawn 1)
  case "$2" in
    ucx) HOSTS+=(--mca pml ucx -x UCX_NET_DEVICES="$NET_IF" -x UCX_TLS=tcp,sm,self) ;;
    ob1) HOSTS+=(--mca btl_tcp_progress_thread 1 -x UCX_NET_DEVICES="$NET_IF") ;;
  esac
  LAUNCHER=$1
  TRANSPORT=$2
}

# mpi <ranks> <threads> <solver args...>: spreads the ranks evenly over the nodes.
mpi() {
  local np=$1 t=$2
  shift 2
  local ppn=$(( (np + NODES - 1) / NODES ))
  (( np < NODES )) && ppn=1
  local map
  if [ -n "${MAP_OVERRIDE:-}" ]; then
    map=(--map-by "$MAP_OVERRIDE" --bind-to core)
  elif (( t == 1 && NODES == 1 )); then
    map=(--map-by numa --bind-to core)
  elif (( t == 1 && SOCKETS == 2 && ppn % 2 == 0 )); then
    map=(--map-by "ppr:$(( ppn / 2 )):socket" --bind-to core)
  elif (( t == 1 )); then
    map=(--map-by "ppr:${ppn}:node" --bind-to core)
  else
    map=(--map-by "ppr:${ppn}:node:PE=${t}" --bind-to core)
  fi
  # timeout: a hung run (as with a broken transport) must not use up the job's walltime
  OMP_NUM_THREADS=$t OMP_PLACES=cores OMP_PROC_BIND=close timeout 600 \
    mpirun -np "$np" ${HOSTS[@]+"${HOSTS[@]}"} "${map[@]}" "$B/poisson_mpi" --csv "$CSV" \
    --tag "${TAG:-}" "$@"
  local s=$?
  if (( s != 0 )); then
    FAILED=$(( FAILED + 1 ))
    echo "!! run failed (exit $s): p=$np threads=$t $*"
  fi
  return 0
}
FAILED=0
kstrong() {  # kstrong <n> <p>
  local k=$(( 200000000 * $2 / $1 ))
  (( k < 50 )) && k=50
  (( k > 1000 )) && k=1000
  echo "$k"
}
uniq_sorted() { tr ' ' '\n' <<<"$*" | grep -v '^$' | sort -n -u | tr '\n' ' '; }

# Multi-node check, before any timing. First the network: the 8-byte latency between two nodes must
# be below 500 us (tens of microseconds is normal; the module's default transport gives 5000).
# Then, on all the job's cores, the distributed A*x must equal the serial A*x bit for bit (2D and
# 3D; this exercises every halo face across the nodes), and a CG solve must agree with the serial
# one (iterations within 1, loose tolerance: this catches a broken Allreduce, not rounding). The
# first launcher and transport that pass are used for the sweep. If none passes, the job stops
# with exit 1, so a job waiting on this one with depend=afterok is deleted instead of starting.
# Then, for the report only: does CG with reproducible dot products still match the serial
# solution bit for bit at this rank count (standard CG, and fused CG with PSCW halos)?
check_run() {  # check_run <solver args...>
  OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close timeout 300 \
    mpirun -np "$TOTAL" "${HOSTS[@]}" --map-by "$FULL_MAP" --bind-to core \
    "$B/poisson_mpi" "$@"
}
latency_ok() {  # skipped (passes) if the OSU benchmarks are missing
  [ -x "${HPCX_OSU_DIR:-}/osu_latency" ] || return 0
  local us
  us=$(OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close timeout 120 \
         mpirun -np 2 "${HOSTS[@]}" --map-by ppr:1:node --bind-to core \
         "$HPCX_OSU_DIR/osu_latency" -m 8:8 -i 2000 -x 200 | awk '$1 == 8 {print $2}')
  echo "   8-byte latency between two nodes: ${us:-no result} us"
  [ -n "$us" ] && awk -v u="$us" 'BEGIN {exit !(u < 500)}'
}
if (( NODES > 1 )) && [ -n "${PBS_NODEFILE:-}" ]; then
  ok=0
  for L in tm ssh; do
    if [ "$L" = tm ] && [ ! -x "$TMRSH" ]; then continue; fi
    for T in ucx ob1; do
      set_hosts "$L" "$T"
      echo "== multi-node check, $TOTAL ranks on $NODES nodes, launcher $L, transport $T ($NET_IF)"
      if latency_ok && check_run --dim 3 --n 96 --test apply && check_run --dim 2 --n 1024 --test apply &&
         check_run --dim 3 --n 48 --test solve --check-tol 1e-6; then
        ok=1
        break 2
      fi
    done
  done
  if (( ok == 0 )); then
    echo "!! multi-node check failed with every launcher and transport: stopping before the sweep"
    exit 1
  fi
  echo "== multi-node check passed; the sweep uses launcher $LAUNCHER, transport $TRANSPORT ($NET_IF)"
  if [ -n "$CLAIM" ]; then
    if ! mkdir "$CLAIM" 2>/dev/null; then
      echo "== another job took the p = $TOTAL lock first ($CLAIM: $(cat "$CLAIM/job" 2>/dev/null)): nothing to do"
      exit 0
    fi
    echo "${PBS_JOBID%%.*} started $(date '+%Y-%m-%d %H:%M') on $NODES nodes x $CORES cores" > "$CLAIM/job"
    echo "== took the p = $TOTAL lock $CLAIM"
  fi
  echo "== reproducible CG on $TOTAL ranks (informational)"
  check_run --dim 3 --n 48 --test solve --dot repro
  check_run --dim 3 --n 48 --test solve --dot repro --cg fused --halo pscw
fi

# Network check in multi-node jobs (job log only, not the CSV): latency and bandwidth between
# two nodes, and the latency of the 8- and 16-byte Allreduce that CG does every iteration, on
# all the job's cores. Uses the OSU benchmarks that ship with HPC-X.
if (( NODES > 1 )) && [ -x "${HPCX_OSU_DIR:-}/osu_bw" ]; then
  echo "== network: OSU latency and bandwidth between 2 nodes, Allreduce on $TOTAL ranks"
  export OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close
  for b in osu_latency osu_bw; do
    timeout 300 mpirun -np 2 ${HOSTS[@]+"${HOSTS[@]}"} --map-by ppr:1:node --bind-to core "$HPCX_OSU_DIR/$b"
  done
  timeout 300 mpirun -np "$TOTAL" ${HOSTS[@]+"${HOSTS[@]}"} --map-by "$FULL_MAP" \
    --bind-to core "$HPCX_OSU_DIR/osu_allreduce" -m 8:16
fi

# Transport control in multi-node jobs: p = 64 on the first node only, with the multi-node
# transport and the single-node job's placement (round-robin over the NUMA domains). Compared
# with the single-node job's p = 64 rows (module defaults: ob1 + shared memory), it shows how much
# of a multi-node result comes from the transport rather than from the extra nodes. Rows tagged
# "control"; CONTROL_P=0 skips it.
CONTROL_P="${CONTROL_P:-64}"
if (( NODES > 1 && CONTROL_P > CORES )); then
  echo "== transport control skipped: p = $CONTROL_P does not fit on one node's $CORES cores"
  CONTROL_P=0
fi
if (( NODES > 1 && CONTROL_P > 0 )) && [ -n "${PBS_NODEFILE:-}" ]; then
  first=$(head -1 "$PBS_NODEFILE")
  echo "== transport control: p = $CONTROL_P on $first only, transport $TRANSPORT"
  TAG=control
  for spec in "3 512" "2 8192"; do
    set -- $spec
    n=$(( $1 == 3 ? $2 * $2 * $2 : $2 * $2 ))
    for rep in $(seq "$REPS"); do
      OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=close timeout 600 \
        mpirun -np "$CONTROL_P" "${HOSTS[@]}" --host "$first:$CORES" --map-by numa --bind-to core \
        "$B/poisson_mpi" --csv "$CSV" --tag "$TAG" --dim "$1" --n "$2" --iters "$(kstrong "$n" "$CONTROL_P")" ||
        { FAILED=$(( FAILED + 1 )); echo "!! control run failed: p=$CONTROL_P $spec"; }
    done
  done
fi

for what in $WHAT; do
  case "$what" in
    placement)  # rank placement within one node, 3D 512^3 (single-node jobs only)
      for pol in core socket numa l3cache; do
        TAG="placement-$pol"
        echo "== rank placement --map-by $pol"
        for p in 2 4 8 12 16 24 32 48; do
          (( p <= CORES )) || continue
          for rep in $(seq "$REPS"); do
            MAP_OVERRIDE=$pol mpi "$p" 1 --dim 3 --n 512 --iters "$(kstrong $((512 * 512 * 512)) "$p")"
          done
        done
      done ;;
    serial)  # E1 + T1 baselines on this (exclusive) node
      echo "== E1 serial scaling and p = 1 baselines"
      B="$B" CSV="$CSV" bash "$(dirname "$0")/serial_sweep.sh" ;;
    strong)  # E3: fixed global problems
      TAG=strong
      echo "== E3 strong scaling, p in $(uniq_sorted $P_STRONG)"
      for p in $(uniq_sorted $P_STRONG); do
        for spec in "3 256" "3 512" "2 8192" "2 16384"; do
          set -- $spec
          n=$(( $1 == 3 ? $2 * $2 * $2 : $2 * $2 ))
          for rep in $(seq "$REPS"); do mpi "$p" 1 --dim "$1" --n "$2" --iters "$(kstrong "$n" "$p")"; done
        done
      done ;;
    weak)  # E4: fixed work per rank (global N_k = L x ranks along k)
      TAG=weak
      echo "== E4 weak scaling, p in $(uniq_sorted $P_WEAK)"
      for p in $(uniq_sorted $P_WEAK); do
        for rep in $(seq "$REPS"); do
          mpi "$p" 1 --dim 3 --nlocal 128 --iters 100
          mpi "$p" 1 --dim 2 --nlocal 2048 --iters 100
        done
      done ;;
    comm)  # E5: decomposition x halo variant x CG variant at p = P_COMM
      TAG=comm
      for p in $(uniq_sorted $P_COMM); do
        echo "== E5 communication variants, p = $p"
        K=$(kstrong $((512 * 512 * 512)) "$p")
        for rep in $(seq "$REPS"); do
          for decomp in block pencil slab; do
            for halo in sendrecv nonblocking overlap rma pscw; do
              mpi "$p" 1 --dim 3 --n 512 --iters "$K" --decomp "$decomp" --halo "$halo"
            done
            mpi "$p" 1 --dim 3 --n 512 --iters "$K" --decomp "$decomp" --cg fused
          done
          # The cost of reproducibility: double-double dot products (standard and fused CG)
          mpi "$p" 1 --dim 3 --n 512 --iters "$K" --dot repro
          mpi "$p" 1 --dim 3 --n 512 --iters "$K" --dot repro --cg fused
          for decomp in block slab; do
            mpi "$p" 1 --dim 2 --n 16384 --iters "$(kstrong $((16384 * 16384)) "$p")" --decomp "$decomp"
          done
        done
      done ;;
    hybrid)  # E6: R ranks x T threads per node, R*T = CORES
      TAG=hybrid
      echo "== E6 hybrid MPI+OpenMP on $NODES node(s)"
      for t in 1 2 4 8 9 12 18 24 36 48 72 96; do
        (( t <= CORES && CORES % t == 0 )) || continue
        r=$(( CORES / t ))
        for rep in $(seq "$REPS"); do mpi $(( r * NODES )) "$t" --dim 3 --n 512 --iters 100; done
      done ;;
    *) echo "unknown experiment $what" >&2 ;;
  esac
done

echo "== done: $FAILED failed run(s)"
# Exit 0 whenever the sweep reaches the end (failed runs are counted above), so a job that waits
# for this one with qsub -W depend=afterok is gated on the sweep finishing, not on its last run.
exit 0
