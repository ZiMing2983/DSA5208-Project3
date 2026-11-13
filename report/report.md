---
title: "Parallel Conjugate Gradient for the 2D and 3D Poisson Equation with MPI, OpenMP and CUDA"
subtitle: "DSA5208 Project 3"
author: "Gao Ziming, student ID A0245234L"
date: "November 2026"
---

**Code:** <https://github.com/ZiMing2983/DSA5208-Project3>. The repository's README explains how to build and run every program and job script. Unless stated otherwise, every runtime in this report was measured on NUS's **Vanda** cluster between 12 and 13 November 2026.

**Summary.** I solve $-\nabla^2 u = f$ on the unit square and the unit cube with a matrix-free Conjugate Gradient (CG) solver, written three times: serial C++, MPI + OpenMP, and MPI + CUDA. One CG iteration does about 20 floating-point operations per unknown but moves at least 104 bytes, so the solver is limited by memory bandwidth, not arithmetic. That one fact explains most of the results. Within a 72-core Vanda node, speedup stops at 17–20 because the cores share the node's memory bandwidth; communication takes only about 5% of the time. The domain decomposition matters (slabs are 21% slower than blocks at 64 ranks in 3D), while the five halo-exchange variants differ by at most 4% inside a node. One NVIDIA A40 GPU, whose memory bandwidth is similar to that of a whole CPU node, is 1.0–1.1× as fast as the 72-core node and 18–21× as fast as one core; two GPUs give 1.8–2.0× on the large problems, and keep 92–100% efficiency when each GPU keeps the same share of a larger problem. Between nodes, the MPI library's default transport on Vanda takes 5 ms per message, which would make multi-node CG useless; with a better-configured transport (UCX over TCP, Section 5) it takes 22 µs. **[TODO: multi-node]** results up to 288 cores.

## 1. The problem and the algorithm

### 1.1 The Poisson equation and its discretisation

The problem is
$$-\nabla^2 u = f \quad \text{in } \Omega = (0,1)^d, \qquad u = 0 \text{ on } \partial\Omega, \qquad d \in \{2, 3\}.$$
I place $N_k$ interior grid points along dimension $k$, with spacing $h_k = 1/(N_k + 1)$; the $N_k$ may differ, which the weak-scaling runs of Section 6.3 need. Second-order central differences replace each second derivative, $\partial^2 u/\partial x^2 \approx (u_{i-1} - 2u_i + u_{i+1})/h_x^2$. In 3D this gives the **7-point stencil**
$$(Au)_{i,j,k} = \frac{2u_{i,j,k} - u_{i-1,j,k} - u_{i+1,j,k}}{h_x^2} + \frac{2u_{i,j,k} - u_{i,j-1,k} - u_{i,j+1,k}}{h_y^2} + \frac{2u_{i,j,k} - u_{i,j,k-1} - u_{i,j,k+1}}{h_z^2},$$
and in 2D the 5-point stencil (the first two terms). A *stencil* is a fixed pattern of neighbouring grid points combined with fixed weights; here the neighbours outside the domain hold the boundary value 0.

The discrete problem is a linear system $Au = b$ with $n = \prod_k N_k$ unknowns, where $b$ holds $f$ at the grid points. Each row of $A$ has $2d + 1$ nonzero entries, fewer next to the boundary, so for $N_k = N$ the number of nonzeros is $\text{nnz} = 5N^2 - 4N$ in 2D and $7N^3 - 6N^2$ in 3D. The matrix is **symmetric positive definite (SPD)**: $A = A^\top$ and $x^\top A x > 0$ for every $x \neq 0$. Its **condition number**, the ratio of its largest to its smallest eigenvalue, is $\kappa(A) \approx 4/(\pi^2 h^2)$ for $h_k = h$, so it grows as $N^2$ [9, 10]. The solver is **matrix-free**: $A$ is never stored, and multiplying by $A$ means applying the stencil.

### 1.2 Conjugate Gradient, its cost, and where communication arises

CG [1] is the standard iterative method for SPD systems. Each iteration does one product with $A$ and two dot products:

```text
x = 0;  r = b;  p = r;  rho = r.r                    [Allreduce]
repeat until sqrt(rho) / ||b|| < tol (or for K iterations):
    halo exchange of p                               [neighbours only]
    q     = A p                                      [stencil, local]
    alpha = rho / (p.q)                              [Allreduce 1]
    x    += alpha p;   r -= alpha q;   rho' = r.r    [local, then Allreduce 2]
    p     = r + (rho'/rho) p;   rho = rho'           [local]
```

In exact arithmetic CG reduces the error by a factor $\varepsilon$ in at most about $\tfrac12\sqrt{\kappa}\,\ln(2/\varepsilon)$ iterations [9], that is $O(N)$ iterations. One iteration costs $O(n)$, so a full solve costs $O(n \cdot N) = O(n^{1 + 1/d})$.

**Arithmetic intensity**, the number of floating-point operations per byte moved to or from memory, decides what limits this loop. Per unknown and iteration the 3D loop does 20 operations (10 in the stencil, 10 in the two dot products and three vector updates) and moves at least 13 doubles = 104 bytes: the stencil reads $p$ and writes $q$ (2), the dot product reads $p$ and $q$ (2), the update of $x$ and $r$ reads four vectors and writes two (6), and the update of $p$ reads two and writes one (3). The intensity is therefore about 0.19 flop/byte. A loop is limited by arithmetic only if its intensity exceeds the hardware's *machine balance*, its peak arithmetic rate divided by its peak memory bandwidth [11]: a few operations per byte for a Vanda CPU node, and about 0.8 for the A40 GPU in double precision. **The solver is memory-bandwidth-bound** on both, and its speed is set by how fast the hardware moves data. In the rest of the report, "GB/s (model)" means $104\,n / t_\text{iter}$: the bandwidth the hardware must sustain to run one iteration in $t_\text{iter}$.

To run CG on $p$ processes, I use **domain decomposition**: the grid is cut into $p$ boxes and each MPI process (*rank*) owns one box and the matching parts of every vector. Two kinds of communication follow:

- **Halo exchange.** The stencil at a point next to a box face needs values owned by the neighbouring rank. Each box is therefore surrounded by a one-cell layer of **ghost cells** (the *halo*), which holds copies of the neighbours' boundary values and is refreshed before every product with $A$. Only neighbouring ranks talk, and the volume is proportional to the box's surface.
- **Global reductions.** Each dot product is a sum over all ranks, done with `MPI_Allreduce`, a *collective* operation that adds one value from every rank and returns the sum to all of them. CG needs two per iteration, and each is a global synchronisation point: no rank can continue until every rank has contributed.

Figure \ref{fig:decomp} shows one rank's box with its ghost cells, and the three box shapes this project compares.

![Domain decomposition. (a) A 2D grid of $8 \times 8$ points on $2 \times 2$ ranks: rank 3 owns the blue points; its ghost cells (hatched) hold copies of the neighbouring ranks' boundary values, and beyond the physical boundary (open squares) the ghost value is 0. The 5-point stencil (red) at a point next to the box face reads a ghost cell. (b)–(d) The slab, pencil and block decompositions of a 3D grid on 8 ranks, with one rank's box shaded. \label{fig:decomp}](figures/decomposition.pdf){width=100%}

A message of $m$ bytes costs about $\alpha + \beta m$, where the **latency** $\alpha$ is the fixed cost of a message and the **bandwidth** $1/\beta$ is the rate at which bytes move. At the problem sizes used here halo messages are large, so bandwidth dominates their cost; Allreduce messages carry 8 bytes, so latency dominates, and a tree-based Allreduce takes about $\log_2 p$ latency steps. The time of one iteration on $p$ ranks is then approximately
$$T_\text{iter}(p) \approx T_\text{comp}(n/p) + \sum_\text{faces}(\alpha + \beta\, m_\text{face}) + 2\,\alpha_\text{red} \log_2 p.$$
The halo volume depends on the shape of the boxes (Table \ref{tab:halo}). The **surface-to-volume ratio**, halo bytes over owned unknowns, is what the decomposition controls: thin slabs send few messages but much data, cubes send more messages but the least data.

| decomposition of an $N^3$ grid | process grid | messages per rank and iteration | halo bytes per rank and iteration | communication / computation |
|----------------------|----------------|------------------|--------------------------|------------------|
| slab (1D) | $p \times 1 \times 1$ | 2 | $2 \cdot 8 N^2$, constant in $p$ | $\propto p/N$ |
| pencil (2D) | $p_x \times p_y \times 1$ | 4 | $4 \cdot 8 N^2/\sqrt{p}$ | $\propto \sqrt{p}/N$ |
| block (3D) | $p_x \times p_y \times p_z$ | 6 | $6 \cdot 8 N^2/p^{2/3}$ | $\propto p^{1/3}/N$ |

Table: Halo traffic of the three decompositions of a 3D grid with $N^3$ unknowns over $p$ ranks (interior ranks, double precision). \label{tab:halo}

### 1.3 How the data is generated

The data is synthetic. I choose the exact solution and derive the right-hand side from it (a **manufactured solution**):
$$u_\text{ex}(x) = \prod_{k=1}^{d} g(x_k), \qquad g(t) = t(1 - t)\,e^{t},$$
which is zero on the boundary. Since $-g''(t) = t(t + 3)e^{t} =: q(t)$,
$$f(x) = -\nabla^2 u_\text{ex} = \sum_{k=1}^{d} q(x_k) \prod_{j \neq k} g(x_j).$$
The program evaluates $f$ at the grid points to form $b$, so no data is read from disk, and any problem size or shape can be generated on the fly. That is what weak scaling needs: it grows the problem with the number of processes, so that each keeps the same amount of work. Because $u_\text{ex}$ is known, the discretisation error $\max |u_h - u_\text{ex}|$ can be measured; it must fall as $h^2$ (test V1, Section 4). I avoided the common choice $u_\text{ex} = \prod_k \sin(\pi x_k)$: on the grid it is an exact eigenvector of $A$, so CG would converge in one iteration and every timing would be meaningless.

Performance runs do not solve to a tolerance. They run a fixed number $K$ of iterations after 10 untimed warm-up iterations, so runs with different $p$ do identical work. If a small problem converges before the last iteration, the solver restarts from $x = 0$ instead of iterating on numbers that underflow. Because the data is synthetic, there are no conclusions to draw about a dataset; the discussion is about the algorithm and the machine.

## 2. Serial performance (E1)

I ran the serial build on one core, pinned to core 0, inside a job that owned a whole Vanda node, so no other job competed for memory bandwidth. Figure \ref{fig:e1} and Table \ref{tab:e1} show the time per iteration for $n$ from $1.6 \cdot 10^4$ to $2.7 \cdot 10^8$ unknowns, and the number of iterations and the time to reach a relative residual of $10^{-8}$ for the sizes that finish in minutes. Both the matrix dimension $n$ and the number of nonzeros vary, by four orders of magnitude.

![Serial runtime on Vanda (one core of an Intel Xeon Platinum 8452Y, inside a whole-node job): time per CG iteration against the number of unknowns $n$ with a slope-1 reference line (left), CG iterations to a relative residual of $10^{-8}$ against grid points per dimension $N$ (middle), and time to solution against $n$ (right). 2D: $N = 128 \ldots 16384$; 3D: $N = 16 \ldots 512$. Fitted log-log slopes in the legends. \label{fig:e1}](figures/e1_vanda.pdf){width=100%}

<!-- include: tables/report_e1_vanda.md | Serial runtime on Vanda (one core, whole-node job). GB/s (model) = 104 bytes per unknown divided by the time per iteration. Times to solution are for a relative residual of $10^{-8}$, run for the sizes that finish within minutes. \label{tab:e1} -->

- **Time per iteration grows linearly with $n$** (and with nnz, which is about $(2d+1)\,n$): the fitted slopes are 1.07 (2D) and 1.11 (3D), close to the predicted 1. The excess comes from the caches. One core sustains 44–70 GB/s while the four CG vectors fit in its 2 MB L2 cache ($n$ up to about $6 \cdot 10^4$), 32–36 GB/s while they fit in the socket's 67.5 MB L3 cache ($n$ up to about $2 \cdot 10^6$), and 21–27 GB/s once they come from main memory, so the time per unknown rises by a factor of about 3 across the range.
- **Iterations grow as $N$**: the slope is 1.01 in both 2D and 3D, as $\sqrt{\kappa} \propto N$ predicts.
- **Time to solution grows as $n^{1.70}$ (2D) and $n^{1.43}$ (3D)**, against the predicted $n^{1 + 1/d}$ = $n^{1.5}$ and $n^{1.33}$. The difference is again the cache effect in the time per iteration, which is largest over the sizes small enough to solve.
- One core reaches about 4 GFLOP/s on the largest problems (20 operations × $1.34 \cdot 10^8$ unknowns in 0.66 s for 3D $512^3$), a small fraction of its arithmetic peak, while moving 21 GB/s. A full solve of 3D $512^3$ would take about 1,900 iterations (extrapolating the iteration counts, which grow as $N$), or 21 minutes. This is the case for parallelisation, and the bandwidth-bound character of the loop is the property that shapes it.

## 3. Parallel design

### 3.1 Domain decomposition and halo exchange (MPI)

`MPI_Dims_create` chooses a balanced process grid and `MPI_Cart_create` builds a non-periodic Cartesian communicator. The same code gives three decompositions: **block** (all dimensions split), **pencil** (no split along $z$) and **slab** (split along $x$ only). When $N_k$ is not divisible by the number of ranks along $k$, the first $N_k \bmod p_k$ ranks get one extra plane. `MPI_Cart_shift` returns `MPI_PROC_NULL` for a neighbour beyond the physical boundary, so boundary faces need no special case: a send to `MPI_PROC_NULL` does nothing, and that ghost layer keeps the value 0. Each vector is stored with its ghost layer in one array, with the last index ($k$ in 3D) contiguous in memory. Before sending, each face is copied (*packed*) into a contiguous buffer, and after receiving it is copied into the ghost layer.

I implemented five halo-exchange variants (`--halo`), to measure what the communication pattern costs:

1. **sendrecv**: blocking `MPI_Sendrecv` with the lower and then the upper neighbour, one dimension after the other. Pairing each send with a receive avoids deadlock.
2. **nonblocking** (the default): `MPI_Irecv` and `MPI_Isend` on all faces at once, then `MPI_Waitall`.
3. **overlap**: as nonblocking, but while the messages are in flight the stencil is applied to the interior points, which need no ghost cells; after `MPI_Waitall` it is applied to the boundary shell.
4. **rma**: one-sided communication, also called remote memory access (RMA). Each rank exposes its receive buffers in an MPI window, and the neighbours write into it with `MPI_Put` between two `MPI_Win_fence` calls (as in the Lecture 8 put/get demo). A fence synchronises every rank of the communicator.
5. **pscw**: the same `MPI_Put`s, synchronised with the post–start–complete–wait (PSCW) calls `MPI_Win_post/start/complete/wait` on an MPI group that contains only the neighbouring ranks, so each rank waits only for its neighbours.

### 3.2 Global reductions

Standard CG needs two Allreduces per iteration. Since they carry 8 bytes each, they cost latency, and at large $p$ they can dominate. I implemented two alternatives (`--cg`, `--dot`):

- **Chronopoulos–Gear CG** [2] (`--cg fused`) rearranges the recurrences so that both dot products of an iteration ($r \cdot r$ and $w \cdot r$ with $w = Ar$) are computed together and summed with **one** Allreduce of two values. It needs one extra vector and slightly more vector updates, so it trades memory traffic for latency.
- **Reproducible dot products** (`--dot repro`). Floating-point addition is not associative (Lecture 8), so a parallel sum changes in its last bits when $p$ or the number of threads changes, and so does the CG solution. With `--dot repro` every rank accumulates its local sum in double-double arithmetic (a pair of doubles $hi + lo$, about 106 bits): Knuth's TwoSum [6, 7] keeps the exact rounding error of every addition. Thread partial sums are combined in an OpenMP critical section, which gives the same result in any order (Vanda's compiler does not support OpenMP user-defined reductions), and ranks are combined by `MPI_Allreduce` with a custom `MPI_Op` on a two-double MPI datatype. The stencil and the vector updates already round identically for every decomposition (Section 4), so with `--dot repro` the whole CG run is **bit-identical** for any number of ranks and threads.

### 3.3 Threads within a rank (OpenMP)

Inside each rank, the stencil and every vector loop is an OpenMP `parallel for collapse(2) schedule(static)` over the $(i, j)$ rows, with an `omp simd` loop over the contiguous index; dot products use `reduction(+:...)`. MPI is initialised with `MPI_THREAD_FUNNELED`: only the master thread calls MPI, between parallel loops. A Vanda node has two sockets with 36 cores each, and each socket has its own memory, so a core reads memory attached to the other socket more slowly: the node is a **NUMA** (non-uniform memory access) machine with two NUMA domains. Linux places a memory page in the NUMA domain of the thread that first writes it (**first touch**). So every vector is initialised inside the same static-scheduled loops that later use it, and each thread's pages end up in its own domain. Ranks and threads are pinned with `OMP_PLACES=cores`, `OMP_PROC_BIND=close` and `mpirun --map-by ppr:R:node:PE=T --bind-to core`.

### 3.4 GPUs (MPI + CUDA)

The CUDA build runs one rank per GPU: `MPI_Comm_split_type` gives each rank its index within the node, which selects the device. All CG vectors stay in GPU memory for the whole solve; only the right-hand side is copied in and the solution copied out. The kernels are written by hand, because cuBLAS is a BLAS library, which the project rules exclude:

- the stencil: blocks of $32 \times 8$ threads tile each $(k, j)$ plane, so consecutive threads read consecutive addresses (*coalesced* access), and the third grid dimension runs over the planes $i$;
- the fused update $x \mathrel{+}= \alpha p$, $r \mathrel{-}= \alpha q$ together with $r \cdot r$, and the update of $p$;
- the dot product: each block of threads sums its values with warp shuffles (exchanges among the 32 threads of a *warp*, the group of threads that executes together) and shared memory, then adds its partial sum to the result with one `atomicAdd`;
- packing and unpacking of the halo faces.

The scalar result of each dot product is copied to the host, which also synchronises the GPU, and summed over ranks with `MPI_Allreduce`. The halo exchange has two variants (`--gpu-mpi`). **CUDA-aware MPI** is an MPI library that accepts GPU memory addresses: the device buffers go straight to `MPI_Isend`/`MPI_Irecv`, and the library moves the data. **Host-staged** exchange copies each face to a pinned (page-locked) host buffer with `cudaMemcpy`, sends it with ordinary MPI, and copies it back on the other side. The one-off host-to-device and device-to-host copies are timed with CUDA events.

### 3.5 Summary of the design

| aspect | choice | alternatives measured | reason |
|------------|----------------------------|-------------------------|-----------------------------------|
| decomposition | block (`MPI_Dims_create`) | pencil, slab | smallest halo for $p \gg 1$ (Table \ref{tab:halo}) |
| uneven splits | first $N \bmod P$ ranks get one more plane | — | any $p$ and $N$ work, as V2a tests |
| halo | non-blocking point-to-point | blocking, overlap, RMA with fence, RMA with PSCW | simple, no deadlock, lets the library progress all faces at once |
| reductions | 2 Allreduces per iteration | Chronopoulos–Gear (1 Allreduce), reproducible double-double sums | standard CG is the baseline; the others trade memory traffic for latency and reproducibility |
| threads | OpenMP static loops, first touch | 1–72 threads per rank (E6) | keeps memory pages local to each socket |
| GPU | 1 rank per GPU, vectors resident on the device, hand-written kernels | CUDA-aware vs host-staged halo | no host-device traffic inside the loop; BLAS not allowed |
| floating point | no fused multiply-add (`-ffp-contract=off`, `-Mnofma`, `--fmad=false`) | — | the stencil rounds identically on every build, so V2a can require bitwise equality; the loop is memory-bound, so FMA would not make it faster |

Table: Design choices and their reasons. \label{tab:design}

### 3.6 How times are measured

Each rank times three phases with a monotonic clock: computation (stencil and vector updates), halo exchange, and Allreduce. The time per iteration of a run is that of the slowest rank, while the phase times are averaged over ranks, so they add up to approximately the total. A rank that waits for slower ranks waits inside the next Allreduce, so load imbalance appears in the Allreduce time; in the overlap variant, the halo time counts only the waiting that computation did not hide. Every configuration runs three times and I report the median (the time-to-solution runs of Section 2 ran once). The serial time $T_1$ of every speedup comes from the serial build, pinned to one core, in the same whole-node job as the parallel runs. MPI on one rank, where every neighbour is `MPI_PROC_NULL` and no message is sent, is within 1.5% of the serial build (within 0.1% on the 2D $16384^2$, 3D $256^3$ and 3D $512^3$ problems), so the MPI build adds no measurable overhead of its own. Pure-MPI ranks are spread round-robin over the two NUMA domains on one node and split evenly between the sockets on several nodes: packing them onto the first cores would put small runs on one socket's memory channels and understate their speed (Figure \ref{fig:placement}).

## 4. Validation

The instructor asks for a few meaningful tests rather than 100+. I use four tests; the validation script runs each of them over a sweep of configurations (rank counts, halo variants, decompositions, 2D and 3D), and the sweep is the test's parameter range. Table \ref{tab:tests} lists them, and every configuration passed on Vanda: 189 on a CPU node and 149 on a GPU node.

| test | what it checks | sweep | pass criterion | result on Vanda |
|-------------|---------------|------------------------|------------------------|--------------------------|
| V1 manufactured solution | the discretisation and the serial solver | 2D $N + 1 = 16 \ldots 1024$, 3D $N + 1 = 16 \ldots 256$, solved to $10^{-10}$ | observed order $\log_2(e_N / e_{2N}) = 2 \pm 0.1$ | 2.000 in 2D and 3D (Table \ref{tab:v1}) |
| V2a distributed stencil | decomposition and halo exchange | $p = 1 \ldots 12$ on uneven grids ($23 \times 17 \times 13$ and $37 \times 29$), all halo variants and decompositions | $A x$ on $p$ ranks **bit-identical** to serial | identical in every case |
| V2b distributed solve | the reductions and the whole solver | the same $p$, standard and fused CG | same iteration count as serial and $\lVert x_p - x_s \rVert / \lVert x_s \rVert \le 10^{-10}$; exactly 0 with `--dot repro` | at most $1.8 \cdot 10^{-15}$; exactly 0 with `--dot repro` for $p = 2, 3, 5, 8$ and with threads |
| V3 other builds | OpenMP and CUDA | V1, V2a, V2b with 2 and 4 threads, and on 1–2 A40s (CUDA-aware and staged) | as V1, V2a, V2b | as above; on the GPU: order 2.000 (2D) and 1.999 (3D), 0 of 24,679 values of $Ax$ differ, solve within $6.4 \cdot 10^{-16}$ |

Table: The four validation tests. \label{tab:tests}

V2a can demand bitwise equality because a distributed product with $A$ performs exactly the same floating-point operations in the same order as the serial one, once fused multiply-adds are disabled; any wrong ghost value or off-by-one face shows up as a difference. V2b cannot demand equality for the standard dot products: the global sums are added in a different order on different $p$ (floating-point addition is not associative), and the differences of order $10^{-15}$ are exactly this round-off. With the double-double dot products the difference becomes 0, which confirms that the reduction order was its only source.

| $N + 1$ | $h$ | 2D: CG iterations | 2D: $\max\lvert u_h - u_\text{ex}\rvert$ | 3D: CG iterations | 3D: $\max\lvert u_h - u_\text{ex}\rvert$ |
|---|---|---|---|---|---|
| 16 | $6.25 \cdot 10^{-2}$ | 51 | $3.62 \cdot 10^{-4}$ | 64 | $1.56 \cdot 10^{-4}$ |
| 32 | $3.13 \cdot 10^{-2}$ | 106 | $9.07 \cdot 10^{-5}$ | 132 | $3.93 \cdot 10^{-5}$ |
| 64 | $1.56 \cdot 10^{-2}$ | 215 | $2.27 \cdot 10^{-5}$ | 269 | $9.82 \cdot 10^{-6}$ |
| 128 | $7.81 \cdot 10^{-3}$ | 434 | $5.68 \cdot 10^{-6}$ | 548 | $2.46 \cdot 10^{-6}$ |
| 256 | $3.91 \cdot 10^{-3}$ | 874 | $1.42 \cdot 10^{-6}$ | 1115 | $6.14 \cdot 10^{-7}$ |
| 512 | $1.95 \cdot 10^{-3}$ | 1760 | $3.55 \cdot 10^{-7}$ | | |
| 1024 | $9.77 \cdot 10^{-4}$ | 3547 | $8.87 \cdot 10^{-8}$ | | |

Table: Test V1: the error of the discrete solution falls by a factor of 4 each time $h$ halves (second order), and the iteration count doubles ($\propto N$). Serial build, relative residual $10^{-10}$. \label{tab:v1}

## 5. Experimental setup

All runtimes come from Vanda (Table \ref{tab:hw}). Every experiment runs as a batch job under PBS, the cluster's job scheduler. Each job script builds the code on the compute node it runs on, records the environment, and writes one CSV row per run; the figures and tables in this report are generated from those CSV files by the analysis scripts in the repository.

**Experiments.** The labels follow the project plan and the figure titles: E1 serial scaling (Section 2), E2 validation (Section 4), E3 strong scaling and E4 weak scaling (Sections 6.1–6.3), E5 the communication variants (Section 6.4), E6 the hybrid MPI + OpenMP sweep (Section 7.1) and E7 the GPU runs (Section 7.2). **Strong scaling** keeps the problem size fixed and adds processes; **weak scaling** grows the problem with the number of processes, so that each process keeps the same amount of work.

| | CPU node | GPU node |
|------------|--------------------------------------------------|--------------------------------------|
| processors | 2 × Intel Xeon Platinum 8452Y, 36 cores each (72 cores, one thread per core) | 2 × NVIDIA A40 (48 GB GDDR6, 696 GB/s peak memory bandwidth [12]) |
| memory | 512 GB DDR5 in two NUMA domains; Intel specifies eight DDR5-4800 channels per socket, at most 614 GB/s per node [13]; L2 2 MB per core, L3 67.5 MB per socket | |
| network | 100 Gb/s Ethernet (`ens6f1`, 9000-byte frames) | the same |
| software | NVIDIA HPC SDK 26.1 (`nvc++`), HPC-X with Open MPI 4.1.9 (module `nvhpc-hpcx-cuda12/26.1`, CUDA-aware), CUDA 12.9, driver 575.57 | the same |
| flags | `-O3 -Mnofma -Kieee -mp` | `nvcc -O3 -arch=sm_86 --fmad=false` |

Table: Hardware and software on Vanda. \label{tab:hw}

**The MPI transport between nodes needed fixing.** The module configures Open MPI to send messages between nodes with its own TCP transport (its `ob1` point-to-point layer over the TCP byte-transfer layer) and gives UCX, the faster communication library that ships with HPC-X, no network devices. With that default, I measured **5.0 ms per message** between two nodes, exactly half of a 10 ms progress interval, and collective operations hung. Every CG iteration needs a halo exchange and two Allreduces, each of which would wait at least that long, so communication alone would take tens of milliseconds per iteration: longer than the computation of a multi-node run. A transport comparison on two CPU nodes, with the OSU micro-benchmarks (standard MPI latency, bandwidth and collective tests [14]), found that **UCX over TCP** on the 100 GbE interface gives an 8-byte latency of 22 µs, a working 8-rank Allreduce in 23–33 µs, and passes all the multi-node validation checks; on GPU nodes the same setting gives 13.5 µs between hosts and 26 µs between GPU buffers. All multi-node runs use it (`--mca pml ucx`); every multi-node job first checks that the latency is below 500 µs and that the distributed $Ax$ is bit-identical to the serial one, before it times anything. Single-node runs keep the module's default, which uses shared memory inside a node.

**Problem sizes.** Strong scaling uses a "small" and a "large" problem in each dimension: $8192^2$ and $16384^2$ in 2D, $256^3$ and $512^3$ in 3D. Each run does $K = 2 \cdot 10^8\, p / n$ iterations, between 50 and 1000, so that short runs are still long enough to time. Weak scaling keeps $2048^2$ (2D) or $128^3$ (3D) unknowns per rank and runs 100 iterations.

## 6. MPI performance

### 6.1 Strong scaling within one node

Figure \ref{fig:strong} and Table \ref{tab:strong} show strong scaling (fixed problem size, more ranks) up to the 72 cores of one node. The **speedup** is $S_p = T_1 / T_p$ and the **parallel efficiency** is $E_p = S_p / p$ (Lecture 7).

![Strong scaling on Vanda, pure MPI, block decomposition, non-blocking halo, standard CG: time per iteration with ideal $T_1/p$ lines (left), speedup with Amdahl fits (middle) and parallel efficiency (right), for 2D $8192^2$, 2D $16384^2$, 3D $256^3$ and 3D $512^3$. $T_1$ is the serial build on one core of the same node. **[TODO: multi-node]** points at $p = 128 \ldots 288$. \label{fig:strong}](figures/strong_vanda.pdf){width=100%}

<!-- include: tables/report_strong_vanda.md | Strong scaling on Vanda: speedup / parallel efficiency against the serial run (pure MPI, block decomposition, non-blocking halo, standard CG; median of 3 runs). \label{tab:strong} -->

<!-- include: tables/report_strong_baselines_vanda.md -->

Efficiency stays near 1 up to 8 ranks and then falls: the speedup within one node levels off at **17–20**, whatever the problem. Communication does not explain this. At 64 ranks on 3D $512^3$, the halo exchange takes 1.1 ms and the Allreduces 0.6 ms of a 34.3 ms iteration (5%; Table \ref{tab:commshare} in Section 6.4 gives every rank count). The limit is memory bandwidth. One core sustains 21–27 GB/s (Section 2); at 64 ranks the node moves $104 \times 1.34 \cdot 10^8$ bytes in 34.3 ms, which is **407 GB/s**, two thirds of the node's theoretical 614 GB/s and about 19 times the single-core figure. A bandwidth-bound loop speeds up only until the cores together saturate the memory channels:
$$T_p \approx \max\!\left(\frac{T_1}{p},\; \frac{104\, n}{b_\text{node}}\right), \qquad S_\text{max} \approx \frac{b_\text{node}}{b_\text{core}} \approx \frac{400}{21} \approx 19,$$
which matches the measured plateau. The 2D problems level off slightly lower (17) because their single-core baseline is a little more efficient (26 GB/s).

The 3D $512^3$ problem is slightly superlinear at 8 ranks ($E = 1.03$). I attribute this to the cache: each of the 8 ranks owns a $256^3$ block whose $256 \times 256$ planes (0.5 MB) let the three planes the 7-point stencil touches stay in the core's 2 MB L2 cache, while the single-core run works with 2 MB planes. 3D $256^3$, in contrast, loses 20% already at 2 ranks. Its single-core baseline is relatively efficient (24.5 GB/s against 21.0 GB/s for $512^3$, plausibly because its 0.5 MB planes stay in L2), and at 2 ranks the Allreduce time, 4.1 ms of 45 ms, shows that one rank waits for the other: the two ranks, one per socket, did not run at the same speed. The three repeats agree within 0.3%, so this is not noise.

Rank placement matters as much as the rank count (Figure \ref{fig:placement}). With `--map-by core`, Open MPI fills the cores of the first socket before it uses the second, so 8 ranks share the memory channels of one socket and take 106 ms per iteration; spread over the two NUMA domains (`--map-by numa` or `socket`) they take 81 ms. At 48 ranks the packed placement puts 36 ranks on one socket and 12 on the other, and is still 40% slower (51 ms against 37 ms). All results above use the spread placement. Placement by L3 cache gives the same result as by socket on this processor, because each socket has a single L3.

![Rank placement within one Vanda node, 3D $512^3$, pure MPI: time per iteration against the number of ranks for four `mpirun --map-by` policies. The dashed line is the time on all 72 cores. \label{fig:placement}](figures/placement_vanda.pdf){width=80%}

**Karp–Flatt and Amdahl** (Lecture 7). Amdahl's law $S(p) = 1/(f + (1 - f)/p)$ fitted to the single-node points gives serial fractions $f = 0.022$–$0.050$ (dotted lines in Figure \ref{fig:strong}), and the experimentally determined serial fraction of Karp and Flatt [4], $e(p) = (1/S_p - 1/p)/(1 - 1/p)$, grows from 0.00–0.02 at $p \le 8$ to about 0.04 at $p = 64$–72 (Table \ref{tab:kf}; 3D $256^3$ starts higher, as explained above). A serial fraction that grows with $p$ means that the overhead grows with $p$. Here the code has no serial part to speak of: the "serial fraction" is the shared memory bandwidth, which acts like a serial resource inside one node. This is the sense in which Lecture 7 calls both laws abstractions: Amdahl's fit predicts $S(\infty) = 20$–46, but a second node brings its own memory channels, so the limit does not carry over to several nodes.

<!-- include: tables/report_karpflatt_vanda.md | Karp–Flatt experimentally determined serial fraction $e(p)$ of the strong-scaling runs on Vanda (same runs as Table \ref{tab:strong}). \label{tab:kf} -->

### 6.2 Strong scaling across nodes

**[TODO: multi-node]** The 2-node (144 cores) and 4-node (288 cores) jobs are queued on Vanda; their rows will appear in Tables \ref{tab:strong} and \ref{tab:kf} and in Figure \ref{fig:strong}. Each added node brings its own memory bandwidth, so the bandwidth model predicts that the speedup resumes growing beyond one node, at about 19 per node, minus the cost of the halo exchange and the Allreduces over the 100 GbE network. Points to cover: the speedup and efficiency relative to one core and relative to one full node; the phase breakdown at 144 and 288 ranks; the share of the Allreduce, whose 8-byte latency between nodes is 22 µs or more (Section 5); and the transport control (64 ranks on one node with the multi-node transport).

<!-- include: tables/report_control_vanda.md | Transport control on Vanda: 64 ranks on one node with the multi-node transport (UCX over TCP), against the single-node job's run with the module's default (shared memory). \label{tab:control} -->

### 6.3 Weak scaling

Weak scaling keeps the work per rank fixed: $2048^2$ unknowns per rank in 2D and $128^3$ in 3D, with the global grid growing with $p$. Ideally the time per iteration stays constant; the **weak-scaling efficiency** is $E_w = T(1)/T(p)$, and Gustafson's **scaled speedup** [5] is $p \cdot T(1)/T(p)$. I measure weak scaling per iteration: the number of CG iterations to a fixed tolerance grows with the global $N$ (Section 2), an algorithmic effect that is not a parallel overhead.

![Weak scaling on Vanda, pure MPI, block decomposition: time per iteration with the ideal constant time (left), weak-scaling efficiency $T(1)/T(p)$ (middle) and Gustafson's scaled speedup $p\,T(1)/T(p)$ (right), for $2048^2$ (2D) and $128^3$ (3D) unknowns per rank. **[TODO: multi-node]** points. \label{fig:weak}](figures/weak_vanda.pdf){width=100%}

<!-- include: tables/report_weak_vanda.md | Weak scaling on Vanda with $2048^2$ (2D) or $128^3$ (3D) unknowns per rank: time per iteration (ms), weak-scaling efficiency $E_w = T(1)/T(p)$ and Gustafson's scaled speedup $p \cdot E_w$ (pure MPI, block decomposition). \label{tab:weak} -->

Within one node (Figure \ref{fig:weak} and Table \ref{tab:weak}), weak scaling falls to $E_w = 0.25$ (2D) and 0.19 (3D) at 64 ranks, for the same reason as strong scaling: 64 ranks doing 64 times the work share one node's memory bandwidth, so each rank gets about 6 GB/s instead of the 27–33 GB/s that one rank sustains when it runs alone. The scaled speedup saturates at 16 (2D) and 12 (3D), again the bandwidth ratio between the node and one core. The communication share is small: at 64 ranks, the 3D weak-scaling problem is the $512^3$ grid of Section 6.1, whose halo and Allreduce take 5% of the time. **[TODO: multi-node]** Across nodes the per-node work and bandwidth stay constant, so the relevant measure is the efficiency relative to one full node, $T(64 \text{ ranks}, 1 \text{ node})/T(p)$; the gap from 1 is the cost of inter-node communication.

### 6.4 The impact of communication

Table \ref{tab:commshare} shows how much of every strong-scaling iteration is spent communicating. Within one node, the halo exchange takes at most 3.3% of an iteration on the two large problems and up to 6.7% on 3D $256^3$, whose small boxes have more surface per point. The Allreduce share, which also counts the time a rank waits for slower ranks, is 1–4%, except on 3D $256^3$ (4–9%), whose two sockets did not run at the same speed (Section 6.1). **[TODO: multi-node]** the rows at 144–288 ranks, where the halo crosses the network.

<!-- include: tables/report_commshare_vanda.md | Communication share of the strong-scaling runs on Vanda: time in the halo exchange / time in the Allreduces, as a percentage of the time per iteration (phase times averaged over ranks; the Allreduce time includes waiting for slower ranks). Same runs as Table \ref{tab:strong}. \label{tab:commshare} -->

Figure \ref{fig:comm64} and Table \ref{tab:comm64} break the time per iteration of 3D $512^3$ on 64 ranks into its phases, for every decomposition, halo variant and reduction variant.

![Communication variants on Vanda, 3D $512^3$, 64 ranks on one node: time per iteration (total of the slowest rank) and the phase times averaged over ranks (stencil and vector updates, halo exchange, Allreduce) for each decomposition / halo variant, with Chronopoulos–Gear ("fused") CG and reproducible ("repro") dot products. Log time axis; the fastest variant is at the top. \label{fig:comm64}](figures/comm_vanda_3d_p64.pdf){width=82%}

<!-- include: tables/report_comm_vanda_3d_p64.md | Communication variants on Vanda, 3D $512^3$, 64 ranks on one node: time per iteration in ms, with the halo-exchange time in parentheses (median of 3 runs). \label{tab:comm64} -->

- **The decomposition is what matters.** With slabs ($64 \times 1 \times 1$), each rank exchanges two full $512 \times 512$ faces, 4.2 MB per iteration, against 0.79 MB for blocks ($4 \times 4 \times 4$) and 1.05 MB for pencils ($8 \times 8 \times 1$). The measured halo time follows the volume: 6.0 ms for slabs against 1.15 ms for blocks, a ratio of 5.2 for a volume ratio of 5.3, and slabs are 21% slower overall. Pencils are the fastest (33.7 ms) although they move 33% more halo data than blocks. They never split the contiguous $k$ direction, so each rank's rows stay 512 values long instead of 128: the stencil and vector loops run over longer contiguous stretches, and their computation takes 32.1 ms against 32.6 ms for blocks, which is most of the difference. The halo time is also slightly lower (1.11 ms against 1.15 ms), because every face a pencil packs is made of contiguous runs, whereas the block decomposition must also pack the $k$-faces element by element.
- **The halo-exchange variants hardly matter within a node.** For each decomposition, blocking, non-blocking, fence and PSCW differ by less than 1%. Inside one node, MPI moves halo data by copying through shared memory, which uses the same memory bandwidth as the stencil, so there is little to gain from different synchronisation. The **overlap** variant is 1–4% slower: computing the interior and the boundary shell separately splits the stencil into up to seven boxes with shorter loops, and that costs more than the 1.1 ms of halo it could hide.
- **The Allreduce is cheap within a node** (0.3–0.7 ms per iteration, or 2 × 0.15–0.35 ms). Chronopoulos–Gear CG saves about 0.07 ms of it but moves more memory in its vector updates, so it is 1% slower here; it is designed for the regime where the Allreduce latency dominates.
- **Reproducibility costs 2%.** Double-double dot products add arithmetic to loops that are bandwidth-bound anyway, and the custom `MPI_Op` makes each Allreduce a little slower (0.95 ms instead of 0.66 ms per iteration): 35.1 ms instead of 34.4 ms, for results that are bit-identical for any $p$.
- **In 2D, communication is negligible** at 64 ranks on $16384^2$: the halo takes 0.25 ms (block) or 0.50 ms (slab) of a 65 ms iteration.

**[TODO: multi-node]** The same comparison at 128 and 256 ranks, where the halo crosses the network: whether the overlap variant now hides network time, whether PSCW beats a global fence, and whether one Allreduce (fused CG) pays off once each Allreduce costs tens of microseconds.

## 7. MPI + X

### 7.1 MPI + OpenMP

Lecture 9 asks how a 72-core Vanda node is best used: 72 ranks of one thread, 1 rank of 72 threads, or something in between. Figure \ref{fig:hybrid} and Table \ref{tab:hybrid} show the sweep for 3D $512^3$ on one node.

![Hybrid MPI + OpenMP on one Vanda node, 3D $512^3$, 72 cores: time per iteration against OpenMP threads per rank; labels give ranks × threads. Threads of a rank occupy consecutive cores. **[TODO: multi-node]** the same sweep on 4 nodes. \label{fig:hybrid}](figures/hybrid_vanda.pdf){width=80%}

<!-- include: tables/report_hybrid_vanda.md | Hybrid MPI + OpenMP sweep on Vanda, 3D $512^3$: time per iteration in ms for each split of the cores into ranks × threads. \label{tab:hybrid} -->

All splits lie within 11% of each other, between 33.2 ms (8 ranks × 9 threads) and 36.9 ms (3 × 24): no split escapes the memory-bandwidth limit of the node, which sets the 33–35 ms. Pure MPI (34.4 ms) is 3% slower than the best split, and a single process with 72 threads (35.2 ms) 6% slower. Fewer ranks mean fewer and larger halo messages, but within a node the halo exchange is only 3% of the time, so there is little to gain. The slowest splits are those with the fewest ranks (3 × 24, 2 × 36, 1 × 72): every parallel loop of an iteration ends at a barrier of 24–72 threads, and with 3 × 24 the middle rank's threads also span both sockets. The practical conclusion is that on a bandwidth-bound code the split hardly matters within one node; the hybrid model's advantage, fewer ranks and messages, should matter more across nodes, where each message is expensive.

### 7.2 MPI + CUDA

Figure \ref{fig:gpu} and Table \ref{tab:gpu} compare one core, one full CPU node, one A40 and two A40s, on the four strong-scaling problems.

![CPU against GPU on Vanda: time per iteration for the four strong-scaling problems on one CPU core, one 72-core CPU node, one NVIDIA A40, two A40s in one node and one A40 on each of two nodes, with CUDA-aware and host-staged halo exchange (left; log axis). Right: memory bandwidth achieved on one A40 (104-byte-per-unknown model) against the 696 GB/s peak. \label{fig:gpu}](figures/gpu_vanda.pdf){width=100%}

<!-- include: tables/report_gpu_vanda.md | CPU against GPU on Vanda: time per iteration in ms. The CPU rows use pure MPI with the block decomposition; the GPU rows use one rank per GPU. \label{tab:gpu} -->

- **One A40 is as fast as a whole CPU node**, 1.0–1.1× its speed, and 18–21× as fast as one core. The GPU sustains 445–488 GB/s, 64–70% of its 696 GB/s peak, against about 400 GB/s for the CPU node. The two have similar memory bandwidth, and that, not the GPU's arithmetic, sets the speed: on 3D $512^3$ the A40 performs $20 \times 1.34 \cdot 10^8$ operations in 31.4 ms, 85 GFLOP/s, about 15% of its double-precision peak (1/64 of its 37 TFLOP/s single-precision peak).
- **Two GPUs give 2.0× in 2D** whether they share a node or not: a 2D halo is one row of 8192 or 16384 values, negligible next to the stencil.
- **In 3D the halo matters.** For $512^3$ on two GPUs, each exchanges a $512 \times 512$ face (2.1 MB) per iteration. Host-staged exchange within one node takes 0.46 ms and gives 2.00×; CUDA-aware MPI takes 2.2 ms and gives only 1.80×. Within a node, the module's default transport appears to pass GPU buffers through host memory in small pieces: a smoke test measured 1.4 GB/s CUDA-aware against 3.1 GB/s staged. Between two nodes, with UCX's CUDA support, CUDA-aware and staged exchange take the same 1.1–1.2 ms and both give 1.9×. So the MPI transport configuration decided whether CUDA-aware MPI paid off, more than whether the two GPUs shared a node. For the small $256^3$ problem, the halo is a larger share of a shorter iteration, and two GPUs give only 1.5–1.8×.
- **Weak scaling on two GPUs is nearly ideal** (Table \ref{tab:gpuweak}). With $16384^2$ (2D) or $512^3$ (3D) unknowns per GPU, two GPUs keep the 2D time per iteration within 1% of one GPU's, wherever they are ($E_w = 0.99$–1.00). In 3D, host-staged exchange within a node keeps $E_w = 0.99$, CUDA-aware MPI within a node gives 0.92 because of its slow 2.7 ms halo, and across two nodes both give 0.96.
- **The host-device copies are paid once.** Copying the right-hand side of 3D $512^3$ to one GPU takes 120 ms and copying the solution back 70 ms, about 6 iterations' worth. A solve of that size needs about 1,900 iterations, so the copies cost 0.3% of it.

<!-- include: tables/report_gpuweak_vanda.md | GPU weak scaling on Vanda: time per iteration, weak-scaling efficiency $E_w$ against one A40, and halo-exchange time, with $16384^2$ (2D) or $512^3$ (3D) unknowns per GPU. \label{tab:gpuweak} -->

Vanda allows at most two GPUs per job, so GPU scaling stops at two.

## 8. Discussion and conclusions

**How the problem is parallelised.** The grid is divided into boxes, one per MPI rank (domain decomposition); each rank applies the stencil to its own box and exchanges a one-cell halo with its neighbours before every product with $A$; the dot products are summed over all ranks with `MPI_Allreduce`. Within a rank, OpenMP threads share the loops over the box, and on GPUs each rank drives one A40 with hand-written CUDA kernels, keeping all vectors on the device. The same decomposition and halo code serve the CPU and GPU builds.

**Which parts need the most communication.** The halo exchange moves the most data, in proportion to the surface of each box, which is why the decomposition shape matters: slabs moved 5.3 times the data of blocks at 64 ranks and were 21% slower. The Allreduces move almost nothing, 8 bytes twice per iteration, but they synchronise every rank and their cost is latency, which grows with $\log p$ and is much higher between nodes: an 8-rank Allreduce across two nodes takes 23–33 µs. As strong scaling shrinks each box, the halo shrinks with it while the Allreduce latency does not, so at large $p$ the Allreduce becomes the main cost. **[TODO: multi-node]** Quantify with the 288-core phase breakdown.

**What limits scaling.** Within one node, memory bandwidth: the loop does 0.19 operations per byte, so 72 cores are only 17–20 times faster than one, and a GPU is only as much faster as its memory is. Communication within a node costs about 5%. Between nodes, the network adds latency to every message; with Vanda's default MPI settings that latency was 5 ms, which would have made every multi-node run meaningless, and checking and fixing the transport (22 µs with UCX) was a prerequisite for the scaling study. **[TODO: multi-node]** what the 2- and 4-node runs show.

**Limitations.** Each configuration ran in a single job on one node (or one pair of nodes), so the variation between nodes is not sampled. The three repeats of a configuration agree closely (median spread 0.3%, 90th percentile 2.7%, largest 11%), and every result is their median; the time-to-solution runs ran once. Some explanations above, such as the cache effects behind the superlinear point, the slower hybrid splits and the staging inside the MPI library, are attributions consistent with the measurements rather than direct measurements. The network is shared with other jobs, so multi-node timings may vary more than single-node ones. **[TODO: multi-node]** the run-to-run spread of the multi-node jobs.

**What I would do next.** The number of CG iterations grows with $N$, so even perfect parallel scaling leaves a time to solution that grows as $n^{1 + 1/d}$; a multigrid preconditioner would make the iteration count nearly independent of $N$. To hide the Allreduce latency at large $p$, pipelined CG [3] overlaps the reduction with the next product with $A$ (using `MPI_Iallreduce`), and s-step CG performs several iterations per reduction. On the GPUs, CUDA streams could overlap the halo exchange with the interior stencil, and remote direct memory access (RDMA) over the nodes' RoCE devices (RDMA over Converged Ethernet), which the site has not enabled for MPI, would avoid the host for GPU messages.

## 9. Declaration of AI usage

I used an AI assistant in two roles in this project:

- **As a coding assistant.** It helped write and debug the solver (the serial, MPI + OpenMP and MPI + CUDA builds), the validation script, the PBS job scripts for the NUS clusters, and the analysis and plotting scripts. It also helped diagnose the problems met on Vanda: the MPI library's missing PBS support, its 5 ms default transport between nodes, and a job held by the scheduler.
- **As a report-drafting assistant.** It helped draft and edit the text of this report from the measured results.

Every runtime in this report was measured on the NUS clusters, and I am responsible for the design choices, the experiments and the conclusions.

## References

[1] M. R. Hestenes and E. Stiefel, "Methods of conjugate gradients for solving linear systems," *Journal of Research of the National Bureau of Standards*, 49(6):409–436, 1952.

[2] A. T. Chronopoulos and C. W. Gear, "s-step iterative methods for symmetric linear systems," *Journal of Computational and Applied Mathematics*, 25(2):153–168, 1989.

[3] P. Ghysels and W. Vanroose, "Hiding global synchronization latency in the preconditioned Conjugate Gradient algorithm," *Parallel Computing*, 40(7):224–238, 2014.

[4] A. H. Karp and H. P. Flatt, "Measuring parallel processor performance," *Communications of the ACM*, 33(5):539–543, 1990.

[5] J. L. Gustafson, "Reevaluating Amdahl's law," *Communications of the ACM*, 31(5):532–533, 1988.

[6] D. E. Knuth, *The Art of Computer Programming, Vol. 2: Seminumerical Algorithms*, 3rd ed., Addison-Wesley, 1997 (Section 4.2.2).

[7] T. Ogita, S. M. Rump and S. Oishi, "Accurate sum and dot product," *SIAM Journal on Scientific Computing*, 26(6):1955–1988, 2005.

[8] Message Passing Interface Forum, *MPI: A Message-Passing Interface Standard, Version 4.0*, 2021.

[9] J. R. Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*, Technical report, Carnegie Mellon University, 1994.

[10] Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.

[11] S. Williams, A. Waterman and D. Patterson, "Roofline: an insightful visual performance model for multicore architectures," *Communications of the ACM*, 52(4):65–76, 2009.

[12] NVIDIA, *NVIDIA A40 datasheet*, 2020.

[13] Intel, *Intel Xeon Platinum 8452Y Processor specifications*, ark.intel.com.

[14] The Ohio State University, *OSU Micro-Benchmarks*, mvapich.cse.ohio-state.edu.

[15] DSA5208 lecture notes, Lectures 7–12, National University of Singapore, 2026.

## Appendix A. Reproducing the results

The repository's README gives the full instructions; in short:

```text
make                 # serial + MPI/OpenMP builds;  make cuda  for the MPI + CUDA build
make test            # validation tests V1-V3
build/poisson_serial --dim 3 --n 256 --iters 100
mpirun -np 64 build/poisson_mpi --dim 3 --n 512 --iters 200 --decomp block --halo nonblocking
mpirun -np 2  build/poisson_cuda --dim 3 --n 512 --iters 200 --gpu-mpi aware
python3 analysis/plot_scaling.py && python3 analysis/make_tables.py && python3 report/build.py
```

The experiments run as PBS jobs on Vanda (`hpc/pbs/`): `vanda_cpu_1node.pbs` (E1, $T_1$, E3–E6 and rank placement on one node), `vanda_cpu_2node.pbs` and `vanda_cpu_4node.pbs` (E3–E6 on 144 and 288 cores; `vanda_cpu_8socket.pbs` runs the 288-core experiments on one socket of each of 8 nodes when four whole nodes are not free), `vanda_gpu.pbs` and `vanda_gpu_2node.pbs` (validation on the GPU and E7). Each script lists its cost in its header; together they use about 500 of the 10,000 CPU-hours and 4 of the 1,000 GPU-hours of a student's yearly Vanda budget.
