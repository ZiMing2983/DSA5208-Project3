# Toolchain for Vanda jobs: NVIDIA HPC SDK (nvc++, nvcc) + HPC-X, CUDA-aware Open MPI.
# Source it from a job script.
type module >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1
module load nvhpc-hpcx-cuda12/26.1
export CG_MACHINE=vanda
# Each rank on a CPU node otherwise prints two "UCX WARN unused environment variable" lines
# (UCX_CUDA_IPC_ENABLE_MNNVL is set but has no GPU to apply to), which buried the job logs.
export UCX_WARN_UNUSED_ENV_VARS=n

# The node's fast Ethernet interface (ens6f1, 100 GbE, on the nodes probed so far), for multi-node
# runs. The module sets OMPI_MCA_pml=ob1, OMPI_MCA_btl=self,vader,tcp and UCX_NET_DEVICES to an
# empty string. Between two nodes that takes 5 ms per message (the TCP BTL is serviced only every
# 10 ms) and the UCC collectives, which run on UCX, cannot reach the other node: an 8-rank
# Allreduce and every CG run hung (hpc/env/net_test_1550775.txt). Multi-node jobs therefore pass
# mpirun "--mca pml ucx -x UCX_NET_DEVICES=$VANDA_NET_IF" (UCX over TCP on this interface: 22 us,
# every check passes); see cpu_sweep.sh. Single-node runs keep the module's defaults.
VANDA_NET_IF=$(for d in /sys/class/net/*; do
  [ "$(cat "$d/operstate" 2>/dev/null)" = up ] &&
    [ "$(cat "$d/speed" 2>/dev/null || echo 0)" -ge 10000 ] 2>/dev/null && basename "$d"
done | head -1)
export VANDA_NET_IF="${VANDA_NET_IF:-ens6f1}"
