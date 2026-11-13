# Parallel Conjugate Gradient for the 2D/3D Poisson equation (MPI + OpenMP + CUDA)

DSA5208 Project 3. Gao Ziming (A0245234L).

The code solves −∇²u = f on the unit square and the unit cube, with u = 0 on the boundary. It discretises the problem with the 5-point (2D) or 7-point (3D) finite-difference stencil and runs a **matrix-free Conjugate Gradient** solver. There are three builds:

| Build | Binary | Parallelism |
|---|---|---|
| serial | `build/poisson_serial` | none; the T₁ baseline of every speedup |
| MPI + OpenMP | `build/poisson_mpi` | Cartesian domain decomposition, halo exchange, `MPI_Allreduce`; OpenMP threads inside each rank |
| MPI + CUDA | `build/poisson_cuda` | one rank per GPU, all vectors on the device, CUDA-aware or host-staged halo exchange |

The design, experiments and report outline are in [PROJECT_PLAN.md](PROJECT_PLAN.md). The cluster setup and job submission are covered in [hpc/README.md](hpc/README.md).

## Layout

```text
├── Makefile                 serial | mpi | cuda | test
├── include/  grid.hpp kernels.hpp cg.hpp halo.hpp options.hpp cuda_kernels.hpp
├── src/      grid.cpp kernels.cpp cg.cpp halo.cpp options.cpp serial_main.cpp mpi_main.cpp
│   └── cuda/ kernels.cu mpi_cuda_main.cpp
├── tests/    validate.sh            validation tests V1, V2a, V2b, V3
├── hpc/      cluster setup + pbs/   job scripts for Atlas and Vanda (experiments E1-E7)
├── analysis/ plot_scaling.py make_tables.py common.py
└── results/  CSV files written by the solvers (one row per run)
```

## Building

```bash
make            # serial + MPI/OpenMP
make cuda       # MPI + CUDA (needs nvcc; CUDA_ARCH=sm_86 for the A40 by default)
make test       # runs tests/validate.sh
```

| Where | Before `make` |
|---|---|
| Mac (Anaconda MPICH + Apple clang) | `source hpc/env_local.sh` |
| Atlas | `source hpc/pbs/env_atlas.sh` (GCC 13.2 + Open MPI 4.1.6) |
| Vanda | `source hpc/pbs/env_vanda.sh` (NVIDIA HPC SDK + HPC-X, CUDA-aware) |

Notes on the build:
- The serial build uses the compiler behind `mpicxx`, so MPI with p = 1 compares like with like.
- All builds turn off fused multiply-adds: `-ffp-contract=off` for GCC and clang, `-Mnofma` for nvc++, `--fmad=false` for nvcc. The stencil then rounds identically everywhere, and validation test V2a can demand bit-identical results. The solver is memory-bound, so FMAs would not speed it up.
- On the clusters, compile on a compute node, because `-march=native` there targets the right CPU. The job scripts do this into `build/<machine>-<jobid>`.

## Running

```bash
build/poisson_serial --dim 3 --n 128 --iters 100                  # fixed-iteration timing run
build/poisson_serial --dim 2 --n 1023 --tol 1e-8                  # time to solution
mpirun -np 8 build/poisson_mpi --dim 3 --n 256 --iters 200 --halo overlap --csv results/x.csv
mpirun -np 8 build/poisson_mpi --dim 3 --nlocal 128 --iters 100   # weak scaling: 128^3 per rank
OMP_NUM_THREADS=4 mpirun -np 2 build/poisson_mpi --dim 3 --n 256 --iters 100
mpirun -np 2 build/poisson_cuda --dim 3 --n 512 --iters 200 --gpu-mpi aware
```

| Option | Meaning |
|---|---|
| `--dim 2\|3` | problem dimension |
| `--n N`, `--nx --ny --nz` | interior points per dimension (they may differ) |
| `--nlocal L` | weak scaling: global N_k = L × (ranks along k) |
| `--iters K` | alone: **fixed mode**, K timed iterations after `--warmup W` (default 10) untimed ones. With `--tol`: the maximum number of iterations |
| `--tol T` | stop when ‖r‖/‖b‖ < T (default 1e-8 when `--iters` is absent) |
| `--decomp block\|pencil\|slab` | process grid from `MPI_Dims_create`: all dimensions, no split in z, or a split in x only |
| `--halo sendrecv\|nonblocking\|overlap\|rma\|pscw` | blocking `MPI_Sendrecv` per dimension; `Irecv`/`Isend` + `Waitall`; non-blocking with interior points computed during the exchange; one-sided `MPI_Put` with `MPI_Win_fence` (synchronises every rank); one-sided `MPI_Put` with `MPI_Win_post/start/complete/wait` on an MPI group of the neighbours (synchronises only neighbours) |
| `--cg standard\|fused` | standard CG (two Allreduces per iteration) or Chronopoulos–Gear CG (one 2-value Allreduce) |
| `--dot fast\|repro` | `repro`: dot products accumulated in double-double (TwoSum), combined over threads in an OpenMP critical section (Vanda's `nvc++` has no OpenMP user-defined reductions; double-double addition gives the same result in any order) and over ranks by a custom `MPI_Op`. The CG solution is then **bit-identical** for any number of ranks and threads. CPU builds only. |
| `--gpu-mpi aware\|staged` | CUDA build: device buffers passed straight to MPI, or copied through pinned host buffers |
| `--csv FILE`, `--tag LABEL`, `--machine NAME` | append one result row; label it; name the machine (default `$CG_MACHINE` or the host name) |
| `--test mms\|apply\|solve` | validation modes (below) |

In fixed mode, a small problem can converge before the last iteration. The solver then restarts from x = 0 instead of iterating on underflowing numbers, so the work per iteration stays the same.

### Output

Each run prints a summary. With `--csv` it also appends one row with these columns:

```text
machine,backend,dim,nx,ny,nz,n,nnz,p,threads,gpus,decomp,halo,cg,iters,rel_res,err_inf,
t_total,t_comp,t_halo,t_reduce,px,py,pz,warmup,tol,t_iter,tag,dot,t_h2d,t_d2h
```

`t_h2d` and `t_d2h` are the CUDA build's one-off host→device and device→host copies, timed with CUDA events; they are 0 for CPU runs.

- **Timers:** `t_total` covers the timed iterations. The phase times split it into stencil + vector updates (`t_comp`), halo exchange (`t_halo`; in the overlap variant only the part that computation did not hide) and `MPI_Allreduce` (`t_reduce`).
- **Over ranks:** `t_total` is the slowest rank's time. The phase times are averages over ranks, so they add up to about `t_total`. Time a rank spends waiting for slower ranks shows up in `t_reduce`, because the Allreduce is where it waits.
- **Errors:** `err_inf` is max |u_h − u_ex| over the grid.

## Validation

`make test` (or `tests/validate.sh BUILD`) runs the tests of PROJECT_PLAN.md §5:

| Test | What it checks |
|---|---|
| V1 `--test mms` | The manufactured solution u = ∏ t(1−t)eᵗ (not ∏ sin πx, which is a discrete eigenvector, so CG would converge in one step). It is solved for N + 1 = 16, 32, …, and the error must fall as h², an observed order of 2 ± 0.1. |
| V2a `--test apply` | One application of A to a non-smooth vector on p ranks must be **bit-identical** to the serial result. It is run for p = 1…8 on grids of 23×17×13 and 37×29, which are uneven and include prime extents, for every halo variant and decomposition. |
| V2b `--test solve` | A full CG solve on p ranks must take the same number of iterations as serial and satisfy ‖x_p − x_s‖/‖x_s‖ ≤ 1e-10. Locally the difference is about 1e-15, coming only from the order of the reduction. With `--dot repro` the difference must be exactly 0: the solve is bit-identical to serial for p = 2, 3, 5, 8 and with OpenMP threads. |
| V3 | V1, V2a and V2b again with 2 and 4 OpenMP threads. The CUDA build runs them on the GPU when one is visible. |

These are four tests. The script runs each one over a sweep of configurations: rank counts, halo variants, decompositions, 2D and 3D. All configurations pass: 141 on the Mac (p ≤ 8). The cluster jobs run the same script with p ≤ 12. The CUDA runs of V3 need Vanda's GPUs. The report presents the four tests with their sweeps rather than a count of configurations; Lecture 12 asks why anyone would need 100+ tests.

## Experiments (cluster job scripts)

| Script | Machine | Runs |
|---|---|---|
| `hpc/pbs/atlas_validate.pbs` | Atlas, 12 cores | build + validation |
| `hpc/pbs/atlas_node.pbs` | Atlas, 1 whole node | E1 + p = 1 baselines, then E3–E6 within one node (Atlas allows ≤ 96 cores per job) |
| `hpc/pbs/vanda_cpu_1node.pbs` | Vanda, 1 whole node | E1 + T₁ baselines, then E3–E6 on one node |
| `hpc/pbs/vanda_cpu_{2,4}node.pbs` | Vanda, 144/288 cores | E3 strong, E4 weak, E5 communication variants (+ E6 on 4 nodes) |
| `hpc/pbs/vanda_gpu.pbs`, `vanda_gpu_2node.pbs` | Vanda, 2 × A40 | GPU validation + E7 |

The loops themselves live in `hpc/pbs/{serial,cpu,gpu}_sweep.sh`. Each job script lists its core-hour or GPU-hour cost in its header.

Two measurement rules come from the first Atlas runs:
- **Baselines run in a job that has the whole node.** On a node shared with other jobs (Atlas's `serial` queue), their memory traffic made the memory-bound serial run about 3× slower, which would inflate every speedup.
- **Ranks are spread out.** On one node, pure-MPI ranks are spread over the NUMA domains. On several nodes, half of each node's ranks go to each socket. Packing ranks onto the first cores puts them all on one NUMA domain's memory channels: on Atlas, 3D 512³ with 4 ranks took 329 ms per iteration packed and 119 ms spread. The single-node jobs also run a `placement` experiment that compares `--map-by core|socket|numa|l3cache`.
- **The serial baseline is pinned to core 0**, as `mpirun -np 1` pins the MPI baseline. Unpinned, the OS sometimes moved it away from its memory, and those runs were 30–40% slower.

## Analysis

```bash
python3 analysis/plot_scaling.py results/*.csv --out report/figures   # E1-E7 figures (log axes, ideal lines)
python3 analysis/make_tables.py results/*.csv --out report/tables     # speedup / efficiency tables (Markdown)
```

`make_tables.py` also writes the compact tables that the report includes (`report/tables/report_*.md`). The report itself is `report/report.md`; `python3 report/build.py` inlines those tables and builds `report/report.pdf` with pandoc and XeLaTeX.

Both scripts take the median over repeated runs. For T₁, speedups use the serial run from the same CSV file, which means the same job on the same whole node. When that file has no serial run, they use MPI with p = 1.

## AI usage

An AI assistant was used as a coding assistant (the solver, the validation, job and analysis scripts, and diagnosing problems on the clusters) and as a report-drafting assistant.
