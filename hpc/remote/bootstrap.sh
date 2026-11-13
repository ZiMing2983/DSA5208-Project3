#!/usr/bin/env bash
# Run this ON a cluster login node (Atlas or Vanda) to record its environment:
#     bash ~/dsa5208-p3/hpc/remote/bootstrap.sh
# It writes ~/dsa5208-p3/hpc/env/<cluster>_env.txt, plus the official sample job
# scripts from `hpc cpu` / `hpc gpu` where they exist.
# Read-only: it submits no jobs and uses no core-hours.
# No `set -u`: the site environment scripts sourced below use unset variables.

# Non-interactive ssh shells may lack `module`, `qstat` and `hpc`; load the login profile.
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1

case "$(hostname -f 2>/dev/null || hostname)" in
  *atlas*) CLUSTER=atlas ;;
  *vanda*|stdct-*) CLUSTER=vanda ;;
  *)       CLUSTER="$(hostname -s)" ;;
esac
OUT_DIR="$HOME/dsa5208-p3/hpc/env"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/${CLUSTER}_env.txt"

# Atlas needs its EasyBuild environment before `module` works.
if [ -f /app1/ebenv ]; then source /app1/ebenv >/dev/null 2>&1; fi

section() { printf '\n===== %s =====\n' "$*"; }
{
  section "login node"
  hostname -f 2>/dev/null || hostname
  date
  echo "user=$USER home=$HOME"
  uname -srm
  section "login node CPU (compute nodes may differ)"
  lscpu 2>/dev/null | grep -E "^(Model name|Socket|Core|Thread|CPU\(s\)|NUMA node\(s\))"
  section "storage (hpc space)"
  hpc space 2>&1 || df -h "$HOME"
  section "hpc helper commands"
  hpc 2>&1 | head -60
  section "queues (qstat -Q)"
  qstat -Q 2>&1
  section "queue limits (qstat -Qf)"
  for q in $(qstat -Q 2>/dev/null | awk 'NR>2 {print $1}'); do
    echo "--- $q"
    qstat -Qf "$q" 2>/dev/null | grep -E "queue_type|enabled|started|resources_(max|min|default|available)|max_run|max_queued|max_user|route_destinations|acl_"
  done
  section "cluster status (gstat)"
  gstat 2>&1 | head -80
  section "modules: MPI / CUDA / compilers / build tools"
  module avail 2>&1 | tr -s ' ' '\n' | grep -i -E "mpi|cuda|nvhpc|hpcx|gcc|gnu|intel|kokkos|cmake|python" | sort -u
  section "my jobs"
  qstat -u "$USER" 2>&1
} > "$OUT" 2>&1

# The sample-script helpers may print or write files, so run them inside env/.
( cd "$OUT_DIR" && hpc cpu > "${CLUSTER}_hpc_cpu_sample.txt" 2>&1 ) || true
( cd "$OUT_DIR" && hpc gpu > "${CLUSTER}_hpc_gpu_sample.txt" 2>&1 ) || true
echo "wrote $OUT_DIR/${CLUSTER}_*"
