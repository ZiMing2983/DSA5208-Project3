# NUS HPC setup (Atlas + Vanda)

Status as of Sat 14 Nov 2026, 18:30:

| Piece | Status |
|---|---|
| NUS VPN | ✔ Cisco Secure Client, profile **NUS Student - TLS - Auto Select Nearest Location**. Both clusters are reachable while it is connected. |
| Atlas | ✔ Login works. Surveyed in `hpc/env/atlas_env.txt`, and the project is synced to `~/dsa5208-p3`. |
| Vanda | ✔ Closed for urgent maintenance from about 21:16 on 9 Nov, open again by 22:19 on 12 Nov (both logged in `hpc/env/vanda_downtime.txt`). Login works, the project is synced, all three builds compile with the NVIDIA toolchain, and the first jobs were submitted on 12 Nov. |
| SSH shortcuts | ✔ `hpc/ssh_config` defines `atlas` and `vanda` with your NUSNET ID and a shared connection that lasts 4 h. `~/.ssh/config` is not touched. |
| Login guard | ✔ `hpc/hpc.sh login` reads the cluster's pre-login notice first. If the notice announces maintenance, the helper stops before the password prompt and appends the notice to `hpc/env/<host>_downtime.txt`. |
| Local MPI + OpenMP build | ✔ Run `source hpc/env_local.sh` first, because Anaconda's `mpicxx` otherwise calls a compiler that isn't installed. |
| Smoke-test jobs | ✔ `hpc/pbs/smoke_gpu.pbs` passed on Vanda (2 A40s; CUDA-aware MPI works). `hpc/pbs/smoke_cpu.pbs` (2 whole nodes) is now optional: every multi-node job checks its own launch first. |
| Solver + experiment jobs | ✔ All done. Validation passes on the Mac (141), Atlas (156) and Vanda (189 CPU; 149 on the GPU node). Vanda jobs: 1-node CPU (1548132), GPU (1548137), 2-node GPU (1550781) and, on 14 Nov after a 30-hour queue and a scheduler outage, the 2-node (1552864, 128/144 ranks) and 4-node (1549748, 256/288 ranks) CPU jobs. Both multi-node CPU jobs completed strong and weak scaling (3 repeats) and reached their walltime inside E5, whose multi-node runs are therefore single runs (1–2 at 256); E6 on 4 nodes did not run. The 8-socket fallback (1552875) was not needed: its multi-node check failed (see *Fallback* below). See [Running the experiments](#running-the-experiments). |
| Google Cloud | Not needed. Project 3 runtime figures must come from Atlas or Vanda. |

## When Vanda reopens

1. **Don't type your password while the notice is up.** A closed cluster can refuse even a correct password, and repeated refusals may lock your NUSNET account, which the VPN, email and Canvas also use. `hpc/hpc.sh login vanda` checks for you: while the notice is showing, it stops with exit code 2 and never asks for the password.
2. Once the check passes, type your NUSNET password at the prompt. If Vanda still refuses it with no notice showing, while the same password works on Atlas, your account may not be enabled for Vanda yet. Raise a request through NUS IT's nTouch portal.
3. Run the commands in [After logging in](#after-logging-in), or ask the AI coding assistant to run them.
4. **Keep `hpc/env/vanda_downtime.txt`.** In Lecture 12 the instructor says the deadline will be extended if Vanda has more downtime. The log holds dated copies of each notice; add the IDs of any jobs an outage kills.

## After logging in

```bash
hpc/hpc.sh sync vanda                                               # copy the project to ~/dsa5208-p3
hpc/hpc.sh run vanda 'bash -l dsa5208-p3/hpc/remote/bootstrap.sh'   # record queues, limits, modules (read-only)
hpc/hpc.sh fetch vanda                                              # -> hpc/env/vanda_env.txt (+ hpc cpu/gpu samples)
hpc/hpc.sh run vanda 'cd dsa5208-p3 && qsub hpc/pbs/smoke_cpu.pbs && qsub hpc/pbs/smoke_gpu.pbs'
hpc/hpc.sh run vanda 'qstat -u $USER'                               # repeat until both jobs are gone
hpc/hpc.sh fetch vanda                                              # -> hpc/env/smoke_cpu.txt, smoke_gpu.txt
```

- `run`, `sync` and `fetch` only reuse the shared connection that `login` opens. Without one they stop with an error instead of asking for a password.
- Keep the `bash -l`: a non-login shell may lack `module` and `qstat`.

## Running the experiments

Every job builds the code on its compute node into `build/<machine>-<jobid>`. It writes CSV rows to `results/` and its log to `hpc/env/`; fetch both with `hpc/hpc.sh fetch <host>`.

**Atlas, while Vanda is down.** Atlas has no GPU queue open to you and allows at most 96 cores per job.

```bash
hpc/hpc.sh login atlas
hpc/hpc.sh sync atlas
hpc/hpc.sh run atlas 'cd dsa5208-p3 && for j in atlas_validate atlas_node; do qsub hpc/pbs/$j.pbs; done'
hpc/hpc.sh run atlas 'qstat -u $USER'
hpc/hpc.sh fetch atlas
```

| Job | Queue / size | Runs |
|---|---|---|
| `atlas_validate.pbs` | `parallel`, 12 cores, ≤ 45 min | build + V1, V2a, V2b, V3 (OpenMP) → `hpc/env/validate_atlas_<job>.txt` |
| `atlas_node.pbs` | `parallel`, one whole node, ≤ 3 h | E1 + p = 1 baselines, E3–E6 within one node, rank-placement comparison → `results/node_atlas_<job>.csv` |

Atlas lets each user run at most 96 cores at once in the `parallel` queue, so these two jobs run one after the other. There is deliberately no `serial`-queue job: that queue shares nodes between users, and the memory-bound baselines measured there came out about 3× too slow.

**Vanda (open again since 12 Nov).** The toolchain, checked on 12 Nov, is `nvhpc-hpcx-cuda12/26.1`:
- **Compilers:** `nvc++` 26.1, plus `nvcc` from the SDK's CUDA 12.9 toolkit, targeting the A40 (sm_86).
- **MPI:** HPC-X 2.25.1, whose Open MPI is CUDA-aware (`opal_cuda_support:true`).
- **CUDA version:** the SDK's default `nvcc` is CUDA 13.1, so the Makefiles select 12.9 explicitly to match HPC-X.
- **Job headers:** jobs need no `-P` line. Each GPU comes with a 36-core socket, and a full CPU node is `ncpus=72:mem=480gb`, as in `hpc cpu` / `hpc gpu`.
- **MPI transport:** each compute node has one active link, `ens6f1`: 100 Gb/s Ethernet with 9000-byte frames, plus an RDMA device (`irdma`) on the same port. The module sets `OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=self,vader,tcp` and `UCX_NET_DEVICES` to an empty string. Within one node that is fine: all single-node results use it. **Between nodes it is unusable.** The first 2-node GPU job (1549749) measured 5.0 ms per message (15 ms above the 64 KB eager limit) and spent its whole 45 minutes in the OSU latency test. `hpc/pbs/net_test.pbs` (job 1550775, 2 × 8 cores) then compared transports between two CPU nodes:

  | Transport between nodes | 8-byte latency | Bandwidth (1 pair) | 8-rank Allreduce | CG checks and timing (8 ranks) |
  |---|---|---|---|---|
  | module default (ob1 + TCP BTL; UCC/hcoll collectives) | 5000 µs | ≤ 2.5 GB/s | hangs | hang |
  | collectives UCC/hcoll off | hangs | | | |
  | UCX given `ens6f1` (collectives work), ob1 + TCP BTL | 5000 µs | ≤ 2.1 GB/s | 35–38 µs | pass, 13.1 ms/iter |
  | TCP BTL with its progress thread | 31 µs | ≤ 6.9 GB/s | hangs | hang |
  | **`--mca pml ucx`, UCX over TCP on `ens6f1`** | **22 µs** | ≤ 2.0 GB/s | 23–33 µs | **pass (sendrecv, RMA, PSCW), 14.4 ms/iter** |
  | `--mca pml ucx`, UCX over RoCE (`irdma3`, UD only) | 7 µs | ≤ 3.4 GB/s | 9–11 µs | pass, 19.0 ms/iter |

  The default's TCP BTL is serviced only every 10 ms (the pattern is exact: 5.0 ms one way, 15.0 ms for the 3-message rendezvous), and UCC, which runs on UCX, cannot reach the other node. **Multi-node jobs therefore use `--mca pml ucx -x UCX_NET_DEVICES=ens6f1 -x UCX_TLS=tcp,sm,self`** (GPU jobs add `cuda_copy,cuda_ipc`). TCP rather than RoCE: the site turned UCX networking off, only the unreliable-datagram transport is available on `irdma`, and the CG timing was no better. `hpc/pbs/env_vanda.sh` finds the interface (`VANDA_NET_IF`). On GPU nodes the same setting gives 13.5 µs between hosts and 26 µs between GPU buffers (job 1550781).
- **GPU buffers within a node** go through `ob1` (the module default), which stages them through host memory in small pieces. That is why `smoke_gpu` measured 1.4 GB/s with CUDA-aware MPI but 3.1 GB/s when the code stages the halo itself.
- **Multi-node launches:** HPC-X's Open MPI 4.1.9 has **no PBS support** (its launchers are only `rsh` and `slurm`), so inside a job `mpirun` sees just the first node. The first 2-node smoke test failed this way. Multi-node runs therefore pass `--hostfile $PBS_NODEFILE --prefix … -x PATH -x LD_LIBRARY_PATH`. They start the remote daemons with **`/opt/pbs/bin/pbs_tmrsh`** (`--mca plm_rsh_agent`), which exists on the compute nodes but not on the login node.
  - `pbs_tmrsh` starts processes through PBS, inside the job's cgroup, so they get the job's cores, GPUs and accounting.
  - Plain ssh (the account's intra-cluster key) is only the fallback. It starts processes outside the cgroup, where a rank sees every GPU of a node that may be shared with another job, so the 2-node GPU job refuses to use it.
  - Before any timing, each multi-node CPU job checks itself (`cpu_sweep.sh`): the 8-byte latency between two nodes must be below 500 µs, then on all its cores distributed A·x must be bit-identical to the serial result in 2D and 3D, and a CG solve must agree with the serial one. It tries launchers `pbs_tmrsh` then ssh and, for each, transports UCX-over-TCP then ob1 with the TCP progress thread, and uses the first combination that passes. If none passes, it stops with exit 1 at a cost of a few minutes. Every run has a timeout, so a hang cannot use up the walltime.
  - Each multi-node CPU job also repeats p = 64 on its first node only with the multi-node transport (rows tagged `control`), to compare with the single-node job's p = 64 rows and separate the transport's effect from the extra nodes'.
  - Compute nodes have no `qstat`. A probe job showed that PBS confines a job to exactly its assigned cores and GPUs (cgroup v2: `/sys/fs/cgroup/pbs_jobs.service/jobs/<id>`).
- **Queue (Fri 13 Nov evening):** `qstat` shows users only their own jobs, so other jobs' remaining time is unknown. `hpc gstat` reported 488 jobs waiting in batch_cpu from 61 users, against 154 running. `pbsnodes -aSj` showed that 74 of the 139 usable CPU nodes each ran one job on all 72 cores, 50 were shared by two jobs (37 of them with only about 8 cores spare), and one node (cn-035) was completely free. The scheduler kept cn-035 idle even for a 2-minute, 1-core job pinned to it ("Insufficient amount of resource: host"), presumably holding it for a large job higher in the queue. A job that needs several whole nodes at once therefore waits a long time: the 4-node job had waited 19 hours by 19:30.
- **Failed starts:** PBS retries a job that fails to start. After 21 failures it puts a *system* hold on it ("job held, too many failed attempts to run"), which only the administrators can release (`qrls` answers "Unauthorized Request"), so resubmit instead. The 2-node CPU job 1549747 failed 21 times in a row at 18:20 on 13 Nov, each start ending within 2 s with exit status −10 (PBS's code for an invalid user or group ID on the execution host: a node could not look up the account). The job script never ran and no core-hours were charged. It was resubmitted unchanged as 1552864.
- **Fallback for the ≥ 256-core runs:** `vanda_cpu_8socket.pbs` asks for one 36-core socket on each of 8 nodes, which can start on nodes that another job shares. It runs the same experiments as the 4-node job, with `SOCKETS=1` placement in `cpu_sweep.sh` (each node's ranks fill the cores PBS gave the job there, recorded at the top of its log). Both jobs stay queued. A job with ≥ 256 cores stops at once if the lock `results/.claim_p288` exists, and takes the lock (`mkdir`, atomic over NFS) as soon as its multi-node check passes. So only the first to start runs the sweep, and the other stops within a minute or two when it starts; delete it if it is still queued. Remove the lock to rerun that sweep. The two layouts differ in one way that matters for the report: each socket still runs 32–36 ranks, so the memory bandwidth per rank is the same, but with 8 nodes more of the halo exchange crosses the network. **Outcome (14 Nov):** the 4-node job took the lock at 12:32 and ran the sweep. The fallback (1552875) started at 12:29 on 8 nodes, passed the latency check (19–52 µs) but failed every A·x check, because Open MPI could not bind ranks inside a partial-node cpuset ("hwloc_set_cpubind returned Error": with `--map-by ppr:36:node --bind-to core` it binds to cores outside the job's allocation). A future partial-node job would need `--bind-to none` or an explicit `--cpu-set` from the cgroup's allowed list.

Submit the validation job first, and let the 1-node CPU job wait for it to pass. The multi-node jobs check their own launch before timing anything. Each job writes its own CSV, so jobs can run at the same time without two nodes appending to one file over NFS. The analysis treats `cpu_vanda+2node.csv`, `cpu_vanda+4node.csv` and `cpu_vanda+8node.csv` as part of `cpu_vanda.csv`'s campaign and uses its T₁; the node count comes from the `+<N>node` suffix.

```bash
hpc/hpc.sh run vanda 'cd dsa5208-p3 && V=$(qsub hpc/pbs/vanda_validate.pbs) && qsub -W depend=afterok:$V hpc/pbs/vanda_cpu_1node.pbs && qsub hpc/pbs/vanda_cpu_2node.pbs && qsub hpc/pbs/vanda_cpu_4node.pbs'
hpc/hpc.sh run vanda 'cd dsa5208-p3 && qsub hpc/pbs/smoke_gpu.pbs && qsub hpc/pbs/vanda_gpu.pbs && qsub hpc/pbs/vanda_gpu_2node.pbs'
```

Together the jobs cost about 500 CPU-h and 4 GPU-h out of the 10,000 / 1,000 budget. The 8-socket fallback adds only 5–10 CPU-h: whichever of it and the 4-node job starts second stops within a minute or two.

| Job | Size | Runs |
|---|---|---|
| `vanda_validate.pbs` | 36 cores | build + validation with the NVIDIA toolchain; its exit status gates the CPU jobs |
| `vanda_cpu_1node.pbs` | 72 cores (whole node) | E1 + the T₁ baselines of every speedup, E3 p = 2…72, E4 p = 1…72, E5 at p = 64, E6 on 1 node |
| `vanda_cpu_2node.pbs` | 144 cores, ≤ 45 min | multi-node check, OSU network test, transport control, then E3/E4 at p = 128, 144 and E5 at p = 128 → `results/cpu_vanda+2node.csv` |
| `vanda_cpu_4node.pbs` | 288 cores, ≤ 70 min | multi-node check, OSU, transport control, then E3/E4 at p = 256, 288, E5 at p = 256 and E6 on 4 nodes (the ≥ 256-core runs) → `results/cpu_vanda+4node.csv` |
| `vanda_cpu_8socket.pbs` | 8 × 36 cores (one socket per node), ≤ 60 min | fallback for the 4-node job, same experiments with E6 per socket and no transport control; the first of the two to start runs, the other stops (lock `results/.claim_p288`) → `results/cpu_vanda+8node.csv` |
| `vanda_gpu.pbs` | 2 × A40, one node | GPU validation (V3) + E7 on 1 and 2 GPUs |
| `vanda_gpu_2node.pbs` | 1 A40 on each of 2 nodes | GPU-confinement check, latency check, OSU between the GPUs, E7 with the halo over 100 GbE → `results/gpu_vanda+2node.csv` |
| `net_test.pbs` | 2 × 8 cores, ≤ 40 min | compares MPI transports between two nodes (table above) → `hpc/env/net_test_<job>.txt` only |

Then, on the Mac: `python3 analysis/plot_scaling.py` and `python3 analysis/make_tables.py`.

**What success looks like (smoke tests):**
- `smoke_cpu.txt` lists 2 different nodes and shows `Allreduce ... PASS` twice, once launched with `pbs_tmrsh` and once with ssh. The job's exit status is that of the `pbs_tmrsh` run.
- `smoke_gpu.txt` shows the GPU model, `CUDA-aware MPI: ... at run time=1`, `GPU buffer sent ... PASS`, and a table comparing CUDA-aware and host-staged bandwidth.

## Cluster facts

**Vanda** (Vanda Cluster User Guide, June 2025):

| Item | Detail |
|---|---|
| CPU nodes | 168 × (2 × 36-core Intel Xeon Platinum 8452Y, 512 GB). One "vnode" is one socket: 36 cores and about 250 GB. |
| GPU nodes | 102 × (2 × NVIDIA A40, 48 GB, 696 GB/s). A40 FP64 runs at 1/64 of the FP32 rate, but CG is memory-bound, so bandwidth sets its speed. |
| Network | 100 Gb Ethernet, not InfiniBand. The guide warns that large MPI jobs may scale poorly; the communication analysis should show this. |
| Job header | No queue line and no `-P` line: jobs are routed by their request and charged to your personal budget. Put `mem=` on every chunk. Full node: `select=1:ncpus=72:mpiprocs=72:mem=500gb`. |
| Budget | 10,000 CPU-h and 1,000 GPU-h per year. A request the remaining budget can't cover fails with "Required Budget is not available". `hpc project` shows the balance (it may ask for `amgr login` first). |
| Batch limits | Up to 10 sockets (360 cores) and 2 GPUs per job, walltime up to 168 h, 4 jobs at once. |
| Interactive limits | `qsub -I`: up to 12 h, at most 2 sockets and 2 GPUs. |
| Free queue | `-q auto_free`: not charged, at most 2 sockets and 2 GPUs, needs `mem=`. It may give AMD GPUs unless you add `gpu-type=nvidia`. |
| Storage | `/home/svu/$USER`: 40 GB, separate from Atlas. `/scratch/$USER`: 500 GB, files purged after 60 days. |
| Building | Compile on a compute node, inside the job script or in `qsub -I -l select=1:ncpus=1:mem=4gb -l walltime=00:10:00`. The login node is a virtual machine, so `-march=native` there would target the wrong CPU. |

**Atlas** (surveyed 9 Nov, `hpc/env/atlas_env.txt`):

| Item | Detail |
|---|---|
| `parallel` queue | 12–96 cores per job, at most 96 cores and 4 jobs running per user, walltime up to 720 h. Very busy on 9 Nov: 964 jobs, 635 of them held. **≥ 256-core runs are impossible here, so the scaling figures need Vanda.** |
| `serial` queue | 1 core, 2–15 GB, up to 40 jobs running per user. Fine for single-core baselines and development runs. |
| Modules | Run `source /app1/ebenv` first. Toolchains include `gompi/2023b` and `intel/2023b`, and CUDA up to 12.9.1. |
| Storage | Home quota 20 GB. |

## Everyday use

- **Workflow:** edit code on the Mac → `hpc/hpc.sh sync vanda` → build and submit on Vanda → `hpc/hpc.sh fetch vanda` to pull back the CSVs.
- **Sync never deletes anything** on the cluster. It also never uploads `results/`, `build/` or `hpc/env/`, so output produced on the cluster can't be overwritten.
- **Keep `hpc/env/` out of the public repo.** It holds cluster logs, and Lecture 12 says not to include log files or terminal output in the submission.
- **Local test runs:** `source hpc/env_local.sh`, then `make -C hpc/smoke cpu && OMP_NUM_THREADS=2 mpirun -np 4 hpc/smoke/hello_hybrid`.
- **Close the shared connection:** `hpc/hpc.sh logout vanda`
