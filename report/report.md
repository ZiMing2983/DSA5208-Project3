---
title: "Parallel Conjugate Gradient for the 2D and 3D Poisson Equation with MPI, OpenMP and CUDA"
subtitle: "DSA5208 Project 3"
author: "Gao Ziming, student ID A0245234L"
date: "November 2026"
---

**Code:** <https://github.com/ZiMing2983/DSA5208-Project3>. The README in the repository explains how to build and run every program and job script. All runtimes in this report were measured on the NUS Vanda cluster between 12 and 14 November 2026.

**Summary.** I solve $-\nabla^2 u = f$ on the unit square and the unit cube with a matrix-free Conjugate Gradient (CG) solver, written in three versions (serial C++, MPI + OpenMP and MPI + CUDA) and measured on up to 288 cores and two GPUs. One CG iteration performs about 20 floating-point operations per unknown but moves at least 104 bytes, so the solver is limited by memory bandwidth rather than by arithmetic, and most of the results follow from that. On one core, the time per iteration grows linearly with the number of unknowns and the iteration count grows with the grid size, as theory predicts. On one 72-core node the speedup levels off at 17–20, because all the cores share the memory bandwidth of the node. Each added node brings its own bandwidth: on four nodes (256–288 cores) the two large problems reach a speedup of 67–77, which is 98–101% of four times the one-node speedup, and the two smaller problems gain even more (78–80 for 2D $8192^2$, and 136–138 for 3D $256^3$, whose share per rank then fits in cache). In weak scaling, four nodes keep the time per iteration within 2–4% of the one-node value. For three of the four problems, communication takes 2–7% of an iteration on one node and 5–17% on four, and the Allreduce becomes its largest part as nodes are added. One NVIDIA A40, whose memory bandwidth is similar to that of a CPU node, is about as fast as a whole node, and two A40s are 1.8–2.0 times as fast as one on the large problems. Four validation tests confirm second-order accuracy and show that the distributed results match the serial ones.

## 1. The problem and the algorithm

### 1.1 The Poisson equation and its discretisation

The problem is
$$-\nabla^2 u = f \quad \text{in } \Omega = (0,1)^d, \qquad u = 0 \text{ on } \partial\Omega, \qquad d \in \{2, 3\}.$$

I place $N_k$ interior grid points along dimension $k$, with spacing $h_k = 1/(N_k + 1)$. The $N_k$ may differ between dimensions, which the weak-scaling runs of Section 6.3 need. Each second derivative is replaced by a second-order central difference, $\partial^2 u/\partial x^2 \approx (u_{i-1} - 2u_i + u_{i+1})/h_x^2$, which in 3D gives the **7-point stencil**
$$(Au)_{i,j,k} = \frac{2u_{i,j,k} - u_{i-1,j,k} - u_{i+1,j,k}}{h_x^2} + \frac{2u_{i,j,k} - u_{i,j-1,k} - u_{i,j+1,k}}{h_y^2} + \frac{2u_{i,j,k} - u_{i,j,k-1} - u_{i,j,k+1}}{h_z^2},$$
and in 2D the 5-point stencil, which consists of the first two terms. (A *stencil* is a fixed pattern of neighbouring grid points combined with fixed weights.) Neighbours outside the domain hold the boundary value 0.

The discrete problem is a linear system $Au = b$ with $n = \prod_k N_k$ unknowns, where $b$ holds $f$ at the grid points. Each row of $A$ has $2d + 1$ nonzero entries (fewer next to the boundary), so for $N_k = N$ the number of nonzeros is $\text{nnz} = 5N^2 - 4N$ in 2D and $7N^3 - 6N^2$ in 3D. The matrix is **symmetric positive definite (SPD)**, meaning that $A = A^\top$ and $x^\top A x > 0$ for every $x \neq 0$. Its **condition number**, the ratio of its largest to its smallest eigenvalue, is $\kappa(A) \approx 4/(\pi^2 h^2)$ for $h_k = h$, so it grows as $N^2$ [9, 10]. I never store $A$: the solver is *matrix-free*, and multiplying by $A$ simply means applying the stencil.

### 1.2 Conjugate Gradient, its cost, and where communication arises

CG [1] is the standard iterative method for SPD systems. Solving $Au = b$ is equivalent to minimising $\phi(x) = \tfrac12 x^\top A x - b^\top x$, whose gradient is $Ax - b = -r$. CG minimises $\phi$ along a sequence of search directions $p_0, p_1, \ldots$ that are **$A$-conjugate**, that is, $p_i^\top A p_j = 0$ for $i \neq j$, so that a step along a new direction never undoes the progress made along the earlier ones. The step length $\alpha$ minimises $\phi$ along $p$, and the factor $\rho'/\rho$ makes the next direction conjugate to the previous one. Conjugacy to all the earlier directions then follows automatically, which is why CG only needs to keep the current $x$, $r$ and $p$. Both scalars are computed from dot products, so every iteration needs global sums. Each iteration performs one product with $A$ and two dot products:

```text
x = 0;  r = b;  p = r;  rho = r.r                    [Allreduce]
repeat until sqrt(rho) / ||b|| < tol (or for K iterations):
    halo exchange of p                               [neighbours only]
    q     = A p                                      [stencil, local]
    alpha = rho / (p.q)                              [Allreduce 1]
    x    += alpha p;   r -= alpha q;   rho' = r.r    [local, then Allreduce 2]
    p     = r + (rho'/rho) p;   rho = rho'           [local]
```

Every line except the halo exchange and the two Allreduces is local: each rank applies it to its own part of the grid, as OpenMP loops on a CPU (Section 3.3) or as CUDA kernels on a GPU (Section 3.4). A dot product is first summed inside the rank, so only one value per rank enters each Allreduce. Since every rank receives the same $\rho$, all ranks decide to stop at the same iteration without any further communication.

In exact arithmetic CG reduces the error by a factor $\varepsilon$ in at most about $\tfrac12\sqrt{\kappa}\,\ln(2/\varepsilon)$ iterations [9], that is, in $O(N)$ iterations. Since one iteration costs $O(n)$, a full solve costs $O(n \cdot N) = O(n^{1 + 1/d})$.

What limits this loop is its **arithmetic intensity**, the number of floating-point operations per byte moved to or from memory. Per unknown and iteration, the 3D loop performs 20 operations: 10 in the stencil, and 10 in the two dot products and three vector updates. It moves at least 13 doubles, or 104 bytes. The stencil reads $p$ and writes $q$ (2 doubles), the dot product reads $p$ and $q$ (2), the update of $x$ and $r$ reads four vectors and writes two (6), and the update of $p$ reads two and writes one (3). The intensity is therefore about 0.19 flop/byte. A loop is limited by arithmetic only if its intensity exceeds the *machine balance* of the hardware, which is its peak arithmetic rate divided by its peak memory bandwidth [11]. This balance is a few operations per byte for a Vanda CPU node and 0.4–0.8 for the A40 GPU in double precision, so the solver is **memory-bandwidth-bound** on both, and its speed is set by how fast the hardware can move data. In the rest of the report, "GB/s (model)" means $104\,n / t_\text{iter}$, the bandwidth the hardware must sustain to complete one iteration in $t_\text{iter}$.

To run CG on $p$ processes I use **domain decomposition**. The grid is cut into $p$ boxes, and each MPI process (a *rank*) owns one box and the matching parts of every vector. This leads to two kinds of communication. The first is the **halo exchange**: the stencil at a point next to a box face needs values owned by the neighbouring rank, so each box is surrounded by a one-cell layer of **ghost cells** (the *halo*), which holds copies of the neighbours' boundary values and is refreshed before every product with $A$. Only neighbouring ranks communicate here, and the volume is proportional to the surface of the box. The second is the **global reductions**. Each dot product is a sum over all ranks, computed with `MPI_Allreduce`, a *collective* operation that adds one value from every rank and returns the sum to all of them. CG needs two of them per iteration, and each one is a global synchronisation point, since no rank can continue until every rank has contributed. Figure \ref{fig:decomp} shows one rank's box with its ghost cells, together with the three box shapes compared in this project.

![Domain decomposition. (a) A 2D grid of $8 \times 8$ points on $2 \times 2$ ranks: rank 3 owns the blue points; its ghost cells (hatched) hold copies of the neighbouring ranks' boundary values, and beyond the physical boundary (open squares) the ghost value is 0. The 5-point stencil (red) at a point next to the box face reads a ghost cell. (b)–(d) The slab, pencil and block decompositions of a 3D grid on 8 ranks, with one rank's box shaded. \label{fig:decomp}](figures/decomposition.pdf){width=100%}

A message of $m$ bytes costs about $\alpha + \beta m$, where the **latency** $\alpha$ is the fixed cost of a message and the **bandwidth** $1/\beta$ is the rate at which bytes move. Most 3D halo messages are tens of kilobytes to megabytes, so their cost is mainly bandwidth. A 2D face is a single row of only 4–16 KB on 64 ranks or more, so once it crosses the network its latency matters more. Allreduce messages carry 8 bytes, so their cost is latency, and a tree-based Allreduce takes about $\log_2 p$ steps, each costing a latency $\alpha_\text{red}$. The time of one iteration on $p$ ranks is then approximately
$$T_\text{iter}(p) \approx T_\text{comp}(n/p) + \sum_\text{faces}(\alpha + \beta\, m_\text{face}) + 2\,\alpha_\text{red} \log_2 p.$$
(In the CG listing above, $\alpha$ is the step length and $p$ the search direction; everywhere else, $p$ is the number of ranks.) The halo volume depends on the shape of the boxes (Table \ref{tab:halo}), because the decomposition sets the **surface-to-volume ratio**, the number of halo bytes per owned unknown. Thin slabs send few messages but a lot of data, while blocks send more messages but the least data.

| decomposition of an $N^3$ grid | process grid | messages per rank and iteration | halo bytes per rank and iteration | communication / computation |
|----------------------|----------------|------------------|--------------------------|------------------|
| slab (1D) | $p \times 1 \times 1$ | 2 | $2 \cdot 8 N^2$, constant in $p$ | $\propto p/N$ |
| pencil (2D) | $p_x \times p_y \times 1$ | 4 | $4 \cdot 8 N^2/\sqrt{p}$ | $\propto \sqrt{p}/N$ |
| block (3D) | $p_x \times p_y \times p_z$ | 6 | $6 \cdot 8 N^2/p^{2/3}$ | $\propto p^{1/3}/N$ |

Table: Halo traffic of the three decompositions of a 3D grid with $N^3$ unknowns over $p$ ranks, for a rank with neighbours on every side (double precision). \label{tab:halo}

### 1.3 How the data is generated

The data is synthetic. I choose the exact solution and derive the right-hand side from it, a technique known as a **manufactured solution**:
$$u_\text{ex}(x) = \prod_{k=1}^{d} g(x_k), \qquad g(t) = t(1 - t)\,e^{t},$$
which is zero on the boundary. Since $-g''(t) = t(t + 3)e^{t} =: q(t)$,
$$f(x) = -\nabla^2 u_\text{ex} = \sum_{k=1}^{d} q(x_k) \prod_{j \neq k} g(x_j).$$
The program evaluates $f$ at the grid points to form $b$. No data is read from disk, and a problem of any size or shape can be generated on the fly, which weak scaling needs. Because $u_\text{ex}$ is known, the discretisation error $\max |u_h - u_\text{ex}|$ can be measured, and it must fall as $h^2$ (test V1, Section 4). I avoided the common choice $u_\text{ex} = \prod_k \sin(\pi x_k)$, because on the grid it is an exact eigenvector of $A$, so CG would converge in a single iteration and every timing would be meaningless.

Performance runs do not solve to a tolerance. Each run first performs untimed warm-up iterations (10, or 2 in the single-core runs with $K < 100$) and then times a fixed number $K$ of iterations (Section 5), and I report the time per iteration. This way every timed iteration does the same work, whatever the value of $p$ and however fast CG converges. If a small problem converges before the last iteration, the solver restarts from $x = 0$ rather than iterate on numbers that underflow.

**Strong scaling** keeps the problem size fixed while ranks are added, and **weak scaling** grows the problem with the number of ranks so that each rank keeps the same amount of work. Pure-MPI runs use one core per rank. The figure titles label the experiments as follows: E1 is serial scaling (Section 2), E2 validation (Section 4), E3 and E4 strong and weak scaling (Section 6), E5 the communication variants (Section 6.4), E6 the hybrid MPI + OpenMP sweep (Section 7.1) and E7 the GPU runs (Section 7.2).

## 2. Serial performance (E1)

I ran the serial build on one core, pinned to core 0, inside a job that owned a whole Vanda node, so that no other job competed for memory bandwidth. Figure \ref{fig:e1} and Table \ref{tab:e1} show the time per iteration for $n$ from $4.1 \cdot 10^3$ to $2.7 \cdot 10^8$ unknowns, with nnz from $2.7 \cdot 10^4$ to $1.3 \cdot 10^9$, so both the dimension of the matrix and the number of nonzeros vary by nearly five orders of magnitude. For the sizes that finish within minutes, they also show the number of iterations and the time needed to reach a relative residual of $10^{-8}$.

![Serial runtime on Vanda (one core of an Intel Xeon Platinum 8452Y, inside a whole-node job): time per CG iteration against the number of unknowns $n$ with a slope-1 reference line (left), CG iterations to a relative residual of $10^{-8}$ against grid points per dimension $N$ (middle), and time to solution against $n$ (right). Left: 2D $N = 128 \ldots 16384$, 3D $N = 16 \ldots 512$; middle and right: 2D $N = 128 \ldots 2048$, 3D $N = 16 \ldots 256$. Fitted log-log slopes in the legends; the dashed lines in the middle and right panels have the theoretical slopes (1 and $1 + 1/d$) and pass through the largest solved size. At equal $n$, the 2D and 3D times per iteration (left) nearly coincide, although the 3D matrices have 38–40% more nonzeros. \label{fig:e1}](figures/e1_vanda.pdf){width=100%}

<!-- include: tables/report_e1_vanda.md | Serial runtime on Vanda (one core, whole-node job). GB/s (model) = 104 bytes per unknown divided by the time per iteration. Times to solution are for a relative residual of $10^{-8}$, run for the sizes that finish within minutes. The $T_1$ of the speedups in Section 6 comes from separate baseline runs in the same job, which agree with these within 0.2%. \label{tab:e1} -->

The time per iteration grows linearly with $n$: the fitted slopes are 1.07 (2D) and 1.11 (3D), close to the predicted 1. The small excess comes from the caches. One core sustains 44–70 GB/s while the four CG vectors fit in its 2 MB L2 cache ($n$ up to about $6 \cdot 10^4$), 32–36 GB/s while they fit in the 67.5 MB L3 cache of the socket ($n$ up to about $2 \cdot 10^6$), and 21–27 GB/s once they have to come from main memory. As a result, the time per unknown rises by a factor of about 2 (2D) and 3 (3D) across the range.

The time follows $n$ rather than nnz. Because $A$ is never stored, its nonzeros cost arithmetic but no memory traffic. At equal $n$ the 3D problems have 38–40% more nonzeros than the 2D ones, so if the time followed nnz, 3D would take 38–40% longer. Instead, 3D $256^3$ takes only 6% longer per iteration than 2D $4096^2$ (71.3 vs. 67.1 ms), and 3D $64^3$ takes even less time than 2D $512^2$ (0.79 vs. 0.85 ms), although that 2D point is the noisiest one in the study (Section 8).

The number of iterations grows as $N$, with a fitted slope of 1.01 in both 2D and 3D, as $\sqrt{\kappa} \propto N$ predicts. The time to solution grows as $n^{1.70}$ (2D) and $n^{1.43}$ (3D), compared with the predicted $n^{1 + 1/d}$, i.e. $n^{1.5}$ and $n^{1.33}$. The difference again comes from the cache effect on the time per iteration, which is largest over the sizes that are small enough to solve.

One core is far from its arithmetic peak. On the largest problems it reaches about 4 GFLOP/s (20 operations × $1.34 \cdot 10^8$ unknowns in 664 ms for 3D $512^3$) while moving 21 GB/s. A full solve of 3D $512^3$ would take about 1,900 iterations (extrapolating the iteration counts, which grow as $N$), or 21 minutes. This is what motivates the parallelisation, and the bandwidth-bound nature of the loop determines how well it can scale.

## 3. Parallel design

### 3.1 Domain decomposition and halo exchange (MPI)

`MPI_Dims_create` chooses a balanced process grid, and `MPI_Cart_create` builds a non-periodic Cartesian communicator [8]. The same code supports three decompositions: *block*, in which all dimensions are split, *pencil*, with no split along $z$, and *slab*, which splits along $x$ only. When $N_k$ is not divisible by $p_k$, the number of ranks along dimension $k$, the first $N_k \bmod p_k$ ranks get one extra plane.

`MPI_Cart_shift` returns `MPI_PROC_NULL` for a neighbour beyond the physical boundary. The blocking variant calls `MPI_Sendrecv` on every face, since a transfer to or from `MPI_PROC_NULL` does nothing, whereas the other variants skip such faces. In both cases a ghost layer on the physical boundary is never written and keeps the value 0 that it was given when the vector was allocated. Each vector is stored together with its ghost layer in one array, with the last index ($k$ in 3D) contiguous in memory. Before sending, each face is copied (*packed*) into a contiguous buffer, and after receiving it is copied into the ghost layer.

To measure what the communication pattern costs, I implemented five halo-exchange variants (`--halo`), whose names are used throughout the report:

1. **sendrecv**: blocking `MPI_Sendrecv` with the lower and then the upper neighbour, one dimension after the other. Pairing each send with a receive avoids deadlock.
2. **nonblocking** (the default): `MPI_Irecv` and `MPI_Isend` on all faces at once, then `MPI_Waitall`.
3. **overlap**: as nonblocking, but the stencil is applied to the interior points, which need no ghost cells, while the messages are in flight. After `MPI_Waitall` it is applied to the boundary shell.
4. **rma**: one-sided communication, also called remote memory access (RMA). Each rank exposes its receive buffers in an MPI window, and the neighbours write into it with `MPI_Put` between two `MPI_Win_fence` calls, as in the Lecture 8 put/get demo [15]. A fence synchronises every rank of the communicator.
5. **pscw**: the same `MPI_Put`s, synchronised with the post–start–complete–wait (PSCW) calls `MPI_Win_post/start/complete/wait`. These work on an MPI group containing only the neighbouring ranks, so each rank waits only for its neighbours.

### 3.2 Global reductions

Standard CG needs two Allreduces per iteration. Each carries only 8 bytes, so its cost is latency, and at large $p$ the Allreduces can dominate. I implemented two alternatives (`--cg` and `--dot`).

The first is **Chronopoulos–Gear CG** [2] (`--cg fused`), which rearranges the recurrences so that both dot products of an iteration ($r \cdot r$ and $w \cdot r$, with $w = Ar$) are computed together and summed with a single Allreduce of two values. It needs one extra vector and one extra vector update, but its updates share one loop, and in the model of Section 1.2 it moves the same 13 doubles per unknown as standard CG. It therefore trades one Allreduce per iteration for a little extra arithmetic and storage.

The second is **reproducible dot products** (`--dot repro`). Floating-point addition is not associative (Lecture 8 [15]), so a parallel sum changes in its last bits when $p$ or the number of threads changes, and the CG solution changes with it. With `--dot repro`, each rank accumulates its local sum in double-double arithmetic, as a pair of doubles $hi + lo$ with about 106 bits of precision. Knuth's TwoSum [6, 7] computes the exact rounding error of every addition, and these errors are summed in the low part. The partial sums of the threads are combined in an OpenMP critical section in whatever order the threads arrive, because Vanda's compiler does not support OpenMP user-defined reductions. Double-double addition is not exactly associative either, but the rounded result changes with the order only if a sum lies extremely close to a rounding boundary. Across ranks, `MPI_Allreduce` combines the partial sums with a custom `MPI_Op` on a two-double MPI datatype. Since the stencil and the vector updates already round identically for every decomposition (Section 4), the whole CG run with `--dot repro` was **bit-identical** in every configuration I tested (2–8 ranks, 2–4 threads), and in practice it should be for any number.

### 3.3 Threads within a rank (OpenMP)

Inside each rank, the stencil and every vector loop is an OpenMP `parallel for collapse(2) schedule(static)` over the $(i, j)$ rows, with an `omp simd` loop over the contiguous index; dot products use `reduction(+:...)`. MPI is initialised with `MPI_THREAD_FUNNELED`, so only the master thread calls MPI, between parallel loops.

A Vanda node has two sockets with 36 cores each, and each socket has its own memory, so a core reads memory attached to the other socket more slowly. The node is therefore a **NUMA** (non-uniform memory access) machine with two NUMA domains. Linux places a memory page in the NUMA domain of the thread that first writes it (**first touch**), so I zero every vector, ghost cells included, when it is allocated, using the same static-scheduled loops that later work on it. Each thread's pages then end up in its own domain.

Ranks and threads are pinned with `OMP_PLACES=cores`, `OMP_PROC_BIND=close` and `mpirun --map-by ppr:R:node:PE=T --bind-to core`, which places $R$ ranks per node, each bound to $T$ cores.

### 3.4 GPUs (MPI + CUDA)

The CUDA build runs one rank per GPU. Each rank selects its GPU by its index within the node, which it reads from the launcher's environment (`OMPI_COMM_WORLD_LOCAL_RANK`) before `MPI_Init`, as CUDA-aware transports prefer, or otherwise obtains from `MPI_Comm_split_type`. All CG vectors stay in GPU memory for the whole solve. Only the right-hand side is copied in and the solution copied out, and these one-off copies are timed with CUDA events. The project rules exclude BLAS (Basic Linear Algebra Subprograms) libraries such as NVIDIA's cuBLAS, so I wrote the CUDA kernels by hand. In the stencil kernel, blocks of $32 \times 8$ threads tile each $(k, j)$ plane, so that consecutive threads read consecutive addresses (*coalesced* access), and the third grid dimension runs over the planes $i$. One kernel fuses the updates $x \mathrel{+}= \alpha p$ and $r \mathrel{-}= \alpha q$ with the dot product $r \cdot r$, and another updates $p$. For a dot product, each block of threads sums its values with warp shuffles (exchanges among the 32 threads of a *warp*, the group of threads that executes together) and shared memory, and then adds its partial sum to the result with one `atomicAdd`. Two more kernels pack and unpack the halo faces.

The scalar result of each dot product is copied to the host, which also synchronises the GPU, and summed over ranks with `MPI_Allreduce`. The halo exchange has two variants (`--gpu-mpi`). With **CUDA-aware MPI**, an MPI library that accepts GPU memory addresses, the device buffers are passed straight to `MPI_Isend` and `MPI_Irecv`, and the library moves the data. In the **host-staged** variant, each face is copied to a pinned (page-locked) host buffer with `cudaMemcpy`, sent with ordinary MPI, and copied back to the GPU on the other side.

### 3.5 Summary of the design

| aspect | choice | alternatives measured | reason |
|------------|----------------------------|-------------------------|-----------------------------------|
| decomposition | block (`MPI_Dims_create`) | pencil, slab | smallest halo for $p \gg 1$ (Table \ref{tab:halo}) |
| uneven splits | the first $N_k \bmod p_k$ ranks along dimension $k$ get one more plane | — | any $p$ and $N$ work, as V2a tests |
| halo | nonblocking point-to-point | sendrecv, overlap, rma (fence), pscw | simple, no deadlock, lets the library progress all faces at once |
| reductions | 2 Allreduces per iteration | Chronopoulos–Gear (1 Allreduce), reproducible double-double sums | standard CG is the baseline; the others trade an Allreduce for arithmetic, or speed for reproducibility |
| threads | OpenMP static loops, first touch | 1–72 threads per rank (E6) | keeps memory pages local to each socket |
| GPU | 1 rank per GPU, vectors resident on the device, hand-written kernels | CUDA-aware and host-staged halo | no bulk host-device traffic inside the loop (only the dot-product scalars, and the halo faces when host-staged); BLAS not allowed |
| floating point | no fused multiply-add (FMA: $ab + c$ computed with one rounding instead of two; `-ffp-contract=off`, `-Mnofma`, `--fmad=false`) | — | the stencil rounds identically on every build, so V2a can require bitwise equality; the loop is memory-bound, so FMA would not make it faster |

Table: Design choices and their reasons. \label{tab:design}

### 3.6 How times are measured

Each rank times three phases with a monotonic clock: computation (the stencil and vector updates), halo exchange, and Allreduce. The time per iteration of a run is that of the slowest rank, while the phase times are averaged over ranks so that they add up to approximately the total. A rank that has to wait for slower ranks does so inside the next Allreduce, so load imbalance shows up as Allreduce time. In the overlap variant, the halo time counts only the waiting that the computation did not hide.

Every configuration was run three times, and I report the median; the time-to-solution runs of Section 2 were run once. The serial time $T_1$ in every speedup comes from the serial build, pinned to one core, in the same whole-node job as the single-node parallel runs. MPI on one rank, where every neighbour is `MPI_PROC_NULL` and no message is sent, is within 1.5% of the serial build (within 0.1% on 2D $16384^2$, 3D $256^3$ and 3D $512^3$), so the MPI build adds no measurable overhead of its own. Pure-MPI ranks are spread round-robin over the two NUMA domains on one node, and split evenly between the sockets on several nodes. Packing them onto the first cores instead would put small runs on the memory channels of a single socket and understate their speed (Figure \ref{fig:placement}).

## 4. Validation

I validated the solver with the four tests in Table \ref{tab:tests}, all of which passed on Vanda. The validation script runs each test over a sweep of configurations covering rank counts, halo variants, decompositions, and 2D and 3D. The 189 configurations checked on a CPU node and the 149 checked on a GPU node are parameter values of these four tests rather than separate tests. Most of them belong to V2a, which accounts for 146 of the 189 on the CPU node: it covers every $p$ from 1 to 12 with each of the five halo variants in 2D and 3D, plus the slab and pencil decompositions and two threaded runs. I made this sweep wide because each combination exercises a different code path (which faces exist, which neighbours are `MPI_PROC_NULL`, and where an uneven split puts the extra plane), and an error in one of them would not show up in the others.

| test | what it checks | sweep | pass criterion | result on Vanda |
|-------------|---------------|------------------------|------------------------|--------------------------|
| V1 manufactured solution | the discretisation and the serial solver | 2D $N + 1 = 16 \ldots 1024$, 3D $N + 1 = 16 \ldots 256$, solved to $10^{-10}$ | observed order $\log_2(e_N / e_{2N}) = 2 \pm 0.1$ | 2.000 in 2D and 3D (Table \ref{tab:v1}) |
| V2a distributed stencil | decomposition and halo exchange | $p = 1 \ldots 12$ on uneven grids ($23 \times 17 \times 13$ and $37 \times 29$), all halo variants and decompositions | $A x$ on $p$ ranks **bit-identical** to serial | identical in every case |
| V2b distributed solve | the reductions and the whole solver | the same $p$, standard and fused CG | iteration count within 1 of serial and $\lVert x_p - x_s \rVert / \lVert x_s \rVert \le 10^{-10}$; with `--dot repro`, the same count and exactly 0 | iteration counts equal; at most $1.8 \cdot 10^{-15}$; exactly 0 with `--dot repro` for $p = 2, 3, 5, 8$ and with threads |
| V3 other builds | OpenMP and CUDA | V1, V2a, V2b with 2 and 4 threads, and on 1–2 A40s (CUDA-aware and host-staged) | as V1 and V2b; on the GPU, $Ax$ within $10^{-13} \max\lvert Ax \rvert$ of the CPU result | as on the CPU; on the GPU: order 2.000 (2D) and 1.999 (3D), 0 of 24,679 values of $Ax$ differ, solve within $6.4 \cdot 10^{-16}$ |

Table: The four validation tests. \label{tab:tests}

V2a can demand bitwise equality because, once fused multiply-adds are disabled, a distributed product with $A$ performs exactly the same floating-point operations in the same order as the serial one, so any wrong ghost value or off-by-one face shows up as a difference. V2b cannot demand equality with the standard dot products. The global sums are added in a different order for different $p$, and floating-point addition is not associative, so the observed differences of order $10^{-15}$ are exactly this round-off. With the double-double dot products the difference becomes 0, which confirms that the order of the reduction was its only source.

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

All runtimes come from Vanda (Table \ref{tab:hw}). Every experiment runs as a batch job under PBS, the job scheduler of the cluster. Each job script builds the code on the compute node it runs on, records the environment, and writes one CSV row per run, and the analysis scripts in the repository generate the figures and tables of this report from these CSV files.

| | CPU node | GPU node |
|------------|--------------------------------------------------|--------------------------------------|
| processors | 2 × Intel Xeon Platinum 8452Y, 36 cores each (72 cores, one thread per core) | 2 × NVIDIA A40 (48 GB GDDR6, 696 GB/s peak memory bandwidth [12]) |
| memory | 512 GB DDR5 in two NUMA domains; Intel specifies eight DDR5-4800 channels per socket, at most 614 GB/s per node [13]; L2 2 MB per core, L3 67.5 MB per socket | |
| network | 100 Gb/s Ethernet (`ens6f1`, 9000-byte frames) | the same |
| software | NVIDIA HPC SDK 26.1 (`nvc++`); HPC-X (NVIDIA's MPI and communication libraries) with Open MPI 4.1.9 (module `nvhpc-hpcx-cuda12/26.1`, CUDA-aware); CUDA 12.9; driver 575.57 | the same |
| flags | `-O3 -Mnofma -Kieee -mp` | `nvcc -O3 -arch=sm_86 --fmad=false` |

Table: Hardware and software on Vanda. \label{tab:hw}

The MPI transport between nodes had to be fixed before any multi-node run was possible. (An MPI library's *transport* is the layer that moves message bytes between processes.) Vanda's MPI module makes Open MPI send messages between nodes through its own TCP transport, and gives UCX, the faster communication library that ships with HPC-X, no network device. That TCP transport is serviced only every 10 ms: I measured 5.0 ms per message between two nodes, and collective operations hung. Every CG iteration needs a halo exchange and two Allreduces, each of which would wait at least that long, so communication alone would have taken tens of milliseconds per iteration.

A transport comparison on two nodes with the OSU micro-benchmarks (standard MPI latency, bandwidth and collective tests [14]) showed that UCX over TCP on the 100 GbE interface works. In the multi-node jobs it gave an 8-byte latency of 22 µs, up to 4.1 GB/s between one pair of ranks (a third of the 12.5 GB/s line rate), and an 8-byte Allreduce in 25 µs on 144 ranks (2 nodes) and 91–99 µs on 288 ranks (4 nodes). Between GPU nodes, the same setting moves a 2 MB message in 0.61 ms from host buffers or 0.73 ms from GPU buffers. All multi-node runs therefore use UCX (`--mca pml ucx`), and before timing anything, every multi-node job checks that the latency is below 500 µs and that the distributed $Ax$ is bit-identical to the serial one. Single-node runs keep the default of the module, which uses shared memory inside a node; Sections 6.2 and 6.4 show that UCX would cost at most 1% there.

Strong scaling uses a "small" and a "large" problem in each dimension: $8192^2$ and $16384^2$ in 2D, and $256^3$ and $512^3$ in 3D. Each CPU run does $K = 2 \cdot 10^8\, p / n$ iterations, kept between 50 and 1000, so that short runs are still long enough to time. Weak scaling keeps $2048^2$ (2D) or $128^3$ (3D) unknowns per rank and runs 100 iterations. The GPU runs do 500 or 200 iterations (200 for weak scaling).

The multi-node runs ran on 14 November, after waiting about 30 hours for whole nodes on a busy cluster and through an outage of the scheduler. Their strong- and weak-scaling runs completed with three repeats each, but both jobs then reached their time limits inside E5, which repeats its whole set of variants three times. The 2-node job (144 cores) stopped after one pass over the 3D variants at 128 ranks, and the 4-node job (288 cores) stopped during the second pass at 256 ranks, so E6 did not run on four nodes. The multi-node E5 results are therefore single runs (one or two at 256 ranks) rather than medians of three.

## 6. MPI performance

### 6.1 Strong scaling within one node

Figure \ref{fig:strong} and Table \ref{tab:strong} show strong scaling on up to four nodes; this section discusses the 72 cores of one node, and Section 6.2 the rest. The **speedup** is $S_p = T_1 / T_p$ and the **parallel efficiency** is $E_p = S_p / p$ (Lecture 7 [15]).

![Strong scaling on Vanda, pure MPI, block decomposition, nonblocking halo, standard CG, for 2D $8192^2$, 2D $16384^2$, 3D $256^3$ and 3D $512^3$ on up to 288 ranks (4 nodes; the dotted vertical line marks the end of one node): time per iteration with ideal $T_1/p$ lines (left), speedup with Amdahl's law $S(p) = 1/(f + (1 - f)/p)$ fitted to the single-node points and drawn over all $p$ (middle; $f$ is the serial fraction), and parallel efficiency (right). The points at $p = 1$ are $T_1$, the serial build on one core. \label{fig:strong}](figures/strong_vanda.pdf){width=100%}

<!-- include: tables/report_strong_vanda.md | Strong scaling on Vanda: speedup / parallel efficiency against the serial run (pure MPI, block decomposition, nonblocking halo, standard CG; median of 3 runs). $T_1$ = 270.7, 1088.3, 71.3 and 663.5 ms for the four problems, in column order. \label{tab:strong} -->

Efficiency stays near 1 up to 8 ranks and then falls, and the speedup within one node levels off at 17–20 for every problem. Communication is not the reason for this plateau, since at 64–72 ranks it takes 2–7% of an iteration on three of the problems and 11–15% on the small 3D $256^3$ (Table \ref{tab:commshare} in Section 6.4). The reason is memory bandwidth. One core sustains 21–27 GB/s (Section 2), and a bandwidth-bound loop speeds up only until the cores together saturate the memory channels:
$$T_p \approx \max\!\left(\frac{T_1}{p},\; \frac{104\, n}{b_\text{node}}\right), \qquad S_\text{max} \approx \frac{b_\text{node}}{b_\text{core}},$$
where $b_\text{node}$ and $b_\text{core}$ are the bandwidths sustained by a full node and by one core. At 64 ranks the node moves 407–486 GB/s, depending on the problem, about 19 times the single-core rate of 3D $512^3$. This $b_\text{node}$ is itself measured at the plateau, however, so the ratio restates the plateau rather than predicting it.

The stronger evidence is that the plateau does not depend on the problem. At 64 ranks the node moves 407–486 GB/s for the four strong-scaling problems, 406–431 GB/s for the two weak-scaling problems, and 379–420 GB/s for all ten hybrid splits of Section 7.1. This is 62–79% of the 614 GB/s peak, similar to the 64–70% of its own peak that the A40 reaches. The 2D problems level off slightly lower, at 17, because their single-core baseline is more efficient (26 GB/s) while the node sustains about 440 GB/s on them, and 440/26 ≈ 17. Section 6.2 tests what the model predicts for several nodes.

The 3D $512^3$ problem is slightly superlinear at 8 ranks ($E = 1.03$), but only in two of its three repeats ($E$ = 1.03 and 1.04; the third gives 0.98), so the effect is small and uncertain. A plausible cause is the cache: each of the 8 ranks owns a $256^3$ block, so the three $256 \times 256$ planes (0.5 MB each) that the 7-point stencil reads at a time fit in the 2 MB L2 cache of a core, whereas in the single-core run each plane is 2 MB.

3D $256^3$ behaves differently and already loses 20% at 2 ranks. Its single-core baseline is relatively efficient (24.5 GB/s, compared with 21.0 GB/s for $512^3$), probably because its 0.5 MB planes stay in L2. At 2 ranks the Allreduce takes 4.1 ms of the 45 ms iteration, which means one rank is waiting for the other: the two ranks, one on each socket, did not run at the same speed. The three repeats agree within 0.3%, so this is not noise.

Rank placement matters as much as the rank count (Figure \ref{fig:placement}). With `--map-by core`, Open MPI fills the cores of the first socket before it uses the second. At 8 ranks, the ranks then share the memory channels of one socket and need 106 ms per iteration, compared with 81 ms when they are spread over the two NUMA domains (`--map-by numa` or `socket`). The gap is largest at 24–32 ranks, which the packed placement puts entirely on one socket: 67–70 ms vs. 39–43 ms, up to 1.7 times slower. At 48 ranks, with 36 ranks on one socket and 12 on the other, packing is still 40% slower. All the results above use the spread placement. Placement by L3 cache gives the same result as placement by socket, because each socket has a single L3.

![Rank placement within one Vanda node, 3D $512^3$, pure MPI: time per iteration against the number of ranks for four `mpirun --map-by` policies. Dashed: ideal $T_1/p$ from the serial build; dotted: the time on all 72 cores. The numa, socket and l3cache policies coincide within 1%. \label{fig:placement}](figures/placement_vanda.pdf){width=80%}

Amdahl's law (Lecture 7 [15]) fitted to the single-node points gives serial fractions $f$ = 0.033 for both 2D problems, 0.050 for 3D $256^3$ and 0.022 for 3D $512^3$, which would cap the speedup at $1/f$ = 20–45. The experimentally determined serial fraction of Karp and Flatt [4], $e(p) = (1/S_p - 1/p)/(1 - 1/p)$, grows from 0.00–0.02 at $p \le 8$ to about 0.04 at $p = 64$–72 (Table \ref{tab:kf}). 3D $256^3$ is the exception at small $p$: its slow two-rank run gives $e(2) = 0.27$, after which its $e(p)$ settles at 0.03–0.04 from $p = 8$. A serial fraction that grows with $p$ indicates an overhead that grows with $p$. The code itself has almost no serial part, so within one node the "serial fraction" measures the shared memory bandwidth, which behaves like a serial resource. Across nodes the cap no longer holds, and $e(p)$ falls again (Section 6.2).

<!-- include: tables/report_karpflatt_vanda.md | Karp–Flatt experimentally determined serial fraction $e(p)$ of the strong-scaling runs on Vanda (same runs as Table \ref{tab:strong}). \label{tab:kf} -->

### 6.2 Strong scaling across nodes

Each added node brings its own memory bandwidth, so the bandwidth model predicts that the time per iteration falls in proportion to the number of nodes, as long as communication stays small. The 2-node (128 and 144 ranks) and 4-node (256 and 288 ranks) runs confirm this (Figure \ref{fig:strong}, Table \ref{tab:strong}). The two large problems scale with the node count. 3D $512^3$ takes 34.3 ms per iteration on one node (64 ranks), 17.4 ms on two and 8.65 ms on four, where the model predicted 8.57 ms, a quarter of the one-node time. On four nodes its speedup over one core is 76.7 at 256 ranks and 74.9 at 288; relative to one full node with the same number of ranks per node, this is an efficiency of 0.99 and 0.98. 2D $16384^2$ reaches 67.7 and 67.5, an efficiency of 1.01 and 0.99 relative to one node.

The two smaller problems scale superlinearly across nodes, because each rank's share shrinks into the caches. 2D $8192^2$ reaches 78–80, which is 1.12–1.15 times four times its one-node speedup. 3D $256^3$ reaches 136–138 on 256–288 ranks, 1.7 times four times its one-node speedup. Each rank's four vectors then take about 2 MB, close to the 2 MB L2 cache of a core, compared with 8.4 MB at 64 ranks, so this problem shows the cache effect most strongly.

All four problems exceed the limit of $1/f$ = 20–45 that Amdahl's law fitted to one node would allow; 3D $512^3$, for example, reaches 77 where the fit allows 45. The Karp–Flatt fraction falls from about 0.04 on one node to 0.02 on two and 0.01 on four (Table \ref{tab:kf}). A truly serial part of the code could not shrink in this way, but a resource that each node brings with it can. The multi-node transport does not distort these results either: running 64 ranks on one node with UCX over TCP instead of the default transport costs 0.2–1.0% (Table \ref{tab:control}).

The efficiency relative to one core stays at 0.23–0.30 on the large problems, as on one node, because it is still set by the bandwidth ratio between a node and a single core. The efficiency relative to one full node measures how well the parallelisation itself scales across nodes, and it is 0.98–1.01 for the large problems.

<!-- include: tables/report_control_vanda.md | Transport control on Vanda: 64 ranks on one node, measured inside the 2-node and 4-node jobs with the multi-node transport (UCX over TCP), against the single-node job's run with the module's default transport (shared memory). \label{tab:control} -->

### 6.3 Weak scaling

In weak scaling each rank keeps $2048^2$ unknowns in 2D or $128^3$ in 3D, so the global grid grows with $p$ and, ideally, the time per iteration stays constant. I report the **weak-scaling efficiency** $E_w = T(1)/T(p)$ and Gustafson's **scaled speedup** $p \cdot T(1)/T(p)$ [5]. Weak scaling is measured per iteration: the number of CG iterations to a fixed tolerance grows with the global $N$ (Section 2), but that is an algorithmic effect rather than a parallel overhead.

![Weak scaling on Vanda, pure MPI, block decomposition, with $2048^2$ (2D) and $128^3$ (3D) unknowns per rank on up to 288 ranks (4 nodes; dotted vertical line: end of one node): time per iteration with the ideal constant time (left), weak-scaling efficiency $T(1)/T(p)$ (middle) and Gustafson's scaled speedup $p\,T(1)/T(p)$ (right). \label{fig:weak}](figures/weak_vanda.pdf){width=100%}

<!-- include: tables/report_weak_vanda.md | Weak scaling on Vanda with $2048^2$ (2D) or $128^3$ (3D) unknowns per rank: time per iteration (ms), weak-scaling efficiency $E_w = T(1)/T(p)$ and Gustafson's scaled speedup $p \cdot E_w$ (pure MPI, block decomposition). \label{tab:weak} -->

Within one node (Figure \ref{fig:weak} and Table \ref{tab:weak}), the weak-scaling efficiency falls to $E_w = 0.25$ (2D) and 0.19 (3D) at 64 ranks. At 64 ranks the cause is bandwidth: 64 ranks doing 64 times the work share the memory bandwidth of one node, so each rank gets about 6 GB/s instead of the 27–33 GB/s that one rank sustains when it runs alone. At small $p$ the cause must be different, because the node then moves only 45–157 GB/s, far below its limit, and yet the first step from 1 to 2 ranks already costs 17% (2D) and 29% (3D), although each rank has a socket to itself. As for 3D $256^3$ in Section 6.1, the computation time rises (by 9% and 19%), and one rank waits for the other in the Allreduce (1.3 and 1.1 ms per iteration), so that communication, including the halo, takes 9% (2D) and 16% (3D) of an iteration. The 3D baseline is also helped by the cache: its $128^3$ box has 70 MB of vectors, about the size of the 67.5 MB L3 cache, and runs at 33 GB/s. On one node the scaled speedup saturates at 16 (2D) and 12 (3D), and the 3D value is lower because of this cached baseline. Compared with one core on the full $512^3$ problem, which takes 664 ms per iteration, the 34.4 ms of 64 ranks is a ratio of 19.

Across nodes the work and the bandwidth per node stay constant, so the time per iteration should stay at its one-node value, and it almost does. In 2D it goes from 64.8 ms (64 ranks, 1 node) to 65.4 ms (2 nodes) and 65.9 ms (4 nodes), and in 3D from 34.4 to 35.1 and 35.7 ms. This is a weak-scaling efficiency of 0.98 (2D) and 0.96 (3D) relative to one node, with the same values at 72, 144 and 288 ranks. The scaled speedup grows in proportion to the number of nodes, reaching 63–64 (2D) and 47–48 (3D) on 256–288 ranks, and the 2–4% lost across four nodes is the cost of inter-node communication.

### 6.4 The impact of communication

Table \ref{tab:commshare} shows how much of each strong-scaling iteration is spent communicating. Within one node, the halo exchange takes at most 3.3% of an iteration on the two large problems and up to 6.7% on 3D $256^3$, whose small boxes have more surface per point. The Allreduce share is 1–4%, except on 3D $256^3$ (4–9%).

Most of the Allreduce time is waiting rather than message latency. Chronopoulos–Gear CG halves the number of Allreduces but reduces their time by only 3–10% (from 0.66 to 0.60 ms per iteration for blocks at 64 ranks), which means that an Allreduce itself costs only about 0.02–0.07 ms. The remaining 0.5 ms or so per iteration is ranks waiting for the slowest one, and fewer Allreduces cannot remove that.

Across nodes, communication grows, and the Allreduce grows fastest. At 256–288 ranks on four nodes, the halo takes 0.9–2.1% of an iteration on the 2D problems and about 6% on 3D $512^3$, while the Allreduces take 5–13% and 9–11% respectively. 3D $256^3$ spends 22–23% of an iteration in the halo and 45–50% in the Allreduces, yet still scales superlinearly, because its computation shrinks into the caches. As strong scaling shrinks each box, the halo shrinks with it, but the Allreduce does not. Its latency grows with the number of nodes, much faster than the $\log_2 p$ of Section 1.2: from 25 µs on 144 ranks (2 nodes) to about 95 µs on 288 (4 nodes; Section 5), nearly four times as much, while $\log_2 p$ grows by only 14%. My explanation is that the steps between nodes dominate it, each costing at least the 22 µs network latency, and that four nodes need more of these steps than two. The time ranks spend waiting for one another grows as well, and at 288 ranks the Allreduce is the largest communication cost for every problem.

<!-- include: tables/report_commshare_vanda.md | Communication share of the strong-scaling runs on Vanda: time in the halo exchange / time in the Allreduces, as a percentage of the time per iteration (phase times averaged over ranks; the Allreduce time includes waiting for slower ranks). Same runs as Table \ref{tab:strong}. \label{tab:commshare} -->

Figure \ref{fig:comm64} and Table \ref{tab:comm64} break down the time per iteration of 3D $512^3$ on 64 ranks into its phases, for every decomposition, halo variant and reduction variant.

![Communication variants on Vanda, 3D $512^3$, 64 ranks on one node: time per iteration (total of the slowest rank) and the phase times averaged over ranks (stencil and vector updates, halo exchange, Allreduce) for each decomposition / halo variant, with Chronopoulos–Gear ("fused") CG and reproducible ("repro") dot products. Log time axis; the fastest variant is at the top. \label{fig:comm64}](figures/comm_vanda_3d_p64.pdf){width=82%}

<!-- include: tables/report_comm_vanda_3d_p64.md | Communication variants on Vanda, 3D $512^3$, 64 ranks on one node: time per iteration in ms, with the halo-exchange time in parentheses (median of 3 runs). \label{tab:comm64} -->

The decomposition matters most. With slabs ($64 \times 1 \times 1$), an interior rank exchanges two full $512 \times 512$ faces, 4.2 MB per iteration, whereas an interior block ($4 \times 4 \times 4$) exchanges 0.79 MB and an interior pencil ($8 \times 8 \times 1$) 1.05 MB. Summed over the 64 ranks, slabs move 7.0 times and pencils 1.6 times as much halo data as blocks. Slabs are 21% slower than blocks overall, and two thirds of the extra 7.1 ms is spent in the halo exchange (6.0 vs. 1.15 ms, close to the ratio of 5.3 for an interior rank). The rest is computation (35.0 vs. 32.6 ms), probably because a slab's $514 \times 514$ planes (2.1 MB) are too large for the three planes read by the stencil to stay in the 2 MB L2 cache, the same effect that Section 6.1 suggests for the superlinear point. Pencils are the fastest (33.7 ms), even though they move more halo data than blocks. They never split the contiguous $k$ direction, so each rank's rows stay 512 values long instead of 128, and their computation takes 32.1 ms compared with 32.6 ms for blocks, which accounts for most of the difference. Their halo time is also slightly lower (1.11 vs. 1.15 ms), because every face a pencil packs consists of contiguous runs, whereas a block must also pack its $k$-faces element by element.

Within a node, the choice of halo variant hardly matters: for each decomposition, sendrecv, nonblocking, rma and pscw differ by less than 1%. Inside one node MPI moves halo data by copying it through shared memory, which uses the same memory bandwidth as the stencil, so a different synchronisation scheme has little to gain. The overlap variant is 1–4% slower and hid no communication at all, since its halo time is not below that of the nonblocking variant (1.34 vs. 1.15 ms for blocks). The most likely reason is that the library moves these messages of 128 KB and more only inside an MPI call, here `MPI_Waitall`, because nothing drives its progress while the interior is being computed; hiding them would require `MPI_Test` calls inside the interior loop or an asynchronous progress thread. The variant is slower because splitting the stencil into up to seven boxes with shorter loops adds 1.3 ms of computation for blocks.

Chronopoulos–Gear CG is 1% slower within a node. It saves about 0.07 ms of Allreduce time, but its computation takes 0.27 ms longer and its halo exchange 0.17 ms longer; the longer computation is probably due to its update loop streaming five vectors instead of four. Reproducible dot products cost 2%. Double-double arithmetic adds work to loops that are bandwidth-bound anyway, and the custom `MPI_Op` makes the Allreduces a little slower (0.95 instead of 0.66 ms per iteration), so an iteration takes 35.1 ms instead of 34.4 ms, in exchange for results that were bit-identical for every $p$ tested. In 2D, communication is negligible at 64 ranks on $16384^2$, where the halo takes 0.25 ms (block) or 0.50 ms (slab) of a 65 ms iteration.

Across nodes (Tables \ref{tab:comm128} and \ref{tab:comm256}), the halo of every box face on a node boundary crosses the network. These are single runs (Section 5), so differences of a few percent are uncertain, but three changes stand out. First, slabs fall further behind: they are 33% slower than blocks at 128 ranks (23.1 vs. 17.4 ms) and 62% slower at 256 (14.2 vs. 8.7 ms), because their halo stays at about 4.7 ms per iteration at both 128 and 256 ranks while the computation halves. Pencils stay level with blocks (0.1% and 1.5% faster). Second, one-sided communication becomes slower than two-sided. rma is 3% slower than nonblocking at 128 ranks and 7% at 256, with most of the extra time in its halo (1.26 vs. 0.54 ms at 256 ranks). pscw is 15% and 23% slower, and its extra time appears mostly as Allreduce time (3.2 vs. 0.9 ms per iteration at 128 ranks): the post–start–complete–wait handshakes delay ranks, which then wait for each other at the next Allreduce. Over TCP, UCX has no remote direct memory access, so it probably carries out the puts and these handshakes in software. Third, the other variants change little. Overlap still hides nothing (it is 2–3% slower). Chronopoulos–Gear CG is now level with standard CG (17.4 vs. 17.4 ms at 128 ranks, 8.74 vs. 8.71 ms at 256): its single Allreduce saves 0.22–0.25 ms per iteration, which its longer update loop and halo use up. Reproducible dot products cost 3% at 128 ranks and 4% at 256 (9.08 vs. 8.71 ms), slightly more than on one node, because their custom `MPI_Op` makes each Allreduce longer.

<!-- include: tables/report_comm_vanda_3d_p128.md | Communication variants on Vanda, 3D $512^3$, 128 ranks on 2 nodes: time per iteration in ms, with the halo-exchange time in parentheses (single runs: the job's time limit stopped the repeats). Reproducible dot products were run with blocks only. \label{tab:comm128} -->

<!-- include: tables/report_comm_vanda_3d_p256.md | Communication variants on Vanda, 3D $512^3$, 256 ranks on 4 nodes: time per iteration in ms, with the halo-exchange time in parentheses (one or two runs each: the job's time limit stopped the repeats). Reproducible dot products were run with blocks only. \label{tab:comm256} -->

## 7. MPI + X

### 7.1 MPI + OpenMP

Lecture 9 asks how a 72-core Vanda node is best used: as 72 ranks of one thread, 1 rank of 72 threads, or something in between. Figure \ref{fig:hybrid} shows the sweep for 3D $512^3$ on one node. The same sweep on four nodes did not run, because the 4-node job reached its time limit before it (Section 5).

![Hybrid MPI + OpenMP on one Vanda node, 3D $512^3$, 72 cores: time per iteration against OpenMP threads per rank (log axes); labels give ranks × threads. Threads of a rank occupy consecutive cores. \label{fig:hybrid}](figures/hybrid_vanda.pdf){width=80%}

All splits lie within 11% of each other, between 33.2 ms (8 ranks × 9 threads) and 36.9 ms (3 × 24). No split escapes the memory-bandwidth limit of the node, which sets the 33–35 ms. Pure MPI (34.4 ms) is 3.5% slower than the best split, and a single rank with 72 threads (35.2 ms) is 6% slower. Measured against one core ($T_1$ = 663.5 ms), the best split gives a speedup of 20.0 and an efficiency of 0.28 on 72 cores. Pure MPI gives 19.3 and 0.27 in this sweep (19.1 in the separate strong-scaling run of Table \ref{tab:strong}), and one rank of 72 threads gives 18.9 and 0.26.

The phase times show the trade-off. Going from 72 × 1 to 8 × 9, the time in the halo exchange and the Allreduces falls from 2.2 to 0.95 ms per iteration while the computation stays at 31.8–32.3 ms, which accounts for the 3.5% gain. With fewer than 8 ranks the computation rises, and the extra time follows the size of each rank's planes, as for the slabs of Section 6.4. The 4 × 18 and 6 × 12 splits, with 1.1 MB planes, compute for 33.4–33.6 ms per iteration. The splits with 3 or fewer ranks keep whole $512 \times 512$ planes (2.1 MB), which are probably too large for the three planes the stencil reads to stay in a core's 2 MB L2 cache, and they compute for 34.8–35.2 ms.

On a bandwidth-bound code, the split therefore matters little within one node. The advantage of the hybrid model, fewer ranks and fewer messages, should matter more across nodes, where every message costs more, but I could not test this prediction because the four-node sweep did not run.

### 7.2 MPI + CUDA

Vanda allows at most two GPUs per job, so Figure \ref{fig:gpu} and Table \ref{tab:gpu} compare one core, one, two and four full CPU nodes, one A40 and two A40s on the four strong-scaling problems.

![CPU against GPU on Vanda: time per iteration for the four strong-scaling problems (2D $8192^2$ and $16384^2$, 3D $256^3$ and $512^3$) on one CPU core, one, two and four 72-core CPU nodes, one NVIDIA A40, two A40s in one node and one A40 on each of two nodes, with CUDA-aware and host-staged halo exchange (left; log axis). Dashed: the ideal two-GPU time, half the one-A40 time. Right: memory bandwidth achieved (104-byte-per-unknown model) on one A40 and on one 72-core CPU node, 2D and 3D separately, against the A40's 696 GB/s and the CPU node's 614 GB/s peaks. \label{fig:gpu}](figures/gpu_vanda.pdf){width=100%}

<!-- include: tables/report_gpu_vanda.md | CPU against GPU on Vanda: time per iteration in ms. The CPU rows use pure MPI with the block decomposition; the GPU rows use one rank per GPU. \label{tab:gpu} -->

One A40 is as fast as a whole CPU node: 1.0–1.1 times its speed, and 18–21 times the speed of one core. Here too, bandwidth sets the speed. The GPU sustains 445–488 GB/s, 64–70% of its 696 GB/s peak, compared with 400–480 GB/s for the CPU node at 72 ranks. The two have similar memory bandwidth, and it is this, not the arithmetic of the GPU, that determines the speed. The arithmetic units are far from their limit: on 3D $512^3$ the A40 performs $20 \times 1.34 \cdot 10^8$ operations in 31.4 ms, or 85 GFLOP/s. That is about 15% of its double-precision peak (1/64 of its 37 TFLOP/s single-precision peak), or 30% of the roughly 290 GFLOP/s that this build can reach without fused multiply-adds.

Two GPUs give 2.0× in 2D, whether or not they share a node, because a 2D halo is a single row of 8192 or 16384 values and is negligible next to the stencil. In 3D, CUDA-aware MPI is slower than host staging within a node. For $512^3$ on two GPUs, each GPU exchanges a $512 \times 512$ face (2.1 MB) per iteration. Within one node, the code's own host staging through pinned buffers takes 0.46 ms per iteration and gives 2.0 times the speed of one GPU, whereas CUDA-aware MPI takes 2.2 ms and gives 1.8 times. The default transport of the module appears to pass GPU buffers through host memory in small pieces: a short preliminary test with 64 MB messages measured 1.4 GB/s CUDA-aware, compared with 3.1 GB/s when staged through ordinary host memory. Between two nodes, where UCX handles the GPU buffers, both variants take 1.1–1.2 ms and give 1.9 times. This is the $\alpha + \beta m$ term of Section 1.2 between nodes. One 2 MB message between the same two nodes takes 0.61–0.73 ms, and the halo sends and receives a 2.1 MB face at the same time and also packs and unpacks it, so it takes 1.5–2 times as long, which costs 7% of the 16.5 ms iteration. CUDA-aware MPI therefore never beat the code's own staging here, and the transport configuration, more than whether the two GPUs shared a node, decided whether it cost anything. For the small $256^3$ problem the halo is a larger share of a shorter iteration, and two GPUs give only 1.5–1.8 times.

Weak scaling on two GPUs is nearly ideal (Table \ref{tab:gpuweak}), with $16384^2$ (2D) or $512^3$ (3D) unknowns per GPU. In 2D, two GPUs keep the time per iteration within 1% of one GPU's wherever they are ($E_w$ = 0.99–1.00). In 3D, host-staged exchange within a node keeps $E_w$ = 0.99, CUDA-aware MPI within a node gives 0.92 because of its slow 2.7 ms halo, and across two nodes both give 0.96. Finally, the host-device copies are paid only once. Copying the right-hand side of 3D $512^3$ to one GPU takes 113 ms and copying the solution back takes 70 ms (medians), about 6 iterations' worth. A solve of that size needs about 1,900 iterations, so the copies cost 0.3% of it.

<!-- include: tables/report_gpuweak_vanda.md | GPU weak scaling on Vanda: time per iteration, weak-scaling efficiency $E_w$ against one A40, and halo-exchange time, with $16384^2$ (2D) or $512^3$ (3D) unknowns per GPU. \label{tab:gpuweak} -->

## 8. Discussion and conclusions

The data in this project is synthetic (Section 1.3), a manufactured solution generated on the fly at any size and shape, so there are no conclusions to draw about a dataset. The conclusions below are about the algorithm and the machine.

### 8.1 How the problem was parallelised

The stencil couples each grid point only to its $2d$ nearest neighbours, so I chose domain decomposition. The grid is divided into boxes, one per MPI rank, and each rank owns its box and the matching parts of every vector, which confines communication to the box surfaces. Blocks have the least surface for $p \gg 1$ (Table \ref{tab:halo}), so they are the default. I kept $A$ matrix-free, because storing it (8 bytes per value and 4 per column index) would add 60 (2D) to 84 (3D) bytes per unknown to the 104 bytes the vectors need, in a loop whose speed is set by memory traffic.

The three kinds of work in a CG iteration are parallelised with different techniques. The product with $A$ needs a halo exchange with the neighbouring ranks (nonblocking MPI point-to-point), followed by the stencil on the local box, which is shared among OpenMP threads by static loops or among CUDA threads in $32 \times 8$ blocks. The three vector updates are purely local and need no communication at any level (OpenMP static loops, or fused CUDA kernels). The two dot products use a two-level reduction, first inside the rank (an OpenMP `reduction`, or warp shuffles and one `atomicAdd` per CUDA block) and then over all ranks with `MPI_Allreduce`. The CPU and GPU builds share the same decomposition and halo code.

### 8.2 How much each choice mattered

On one node, placing the ranks on both sockets matters most, since packed ranks are up to 1.7 times slower. The box shape comes next, with slabs 21% slower than blocks at 64 ranks. The halo variant (within 1%, with overlap 1–4% slower), a single fused Allreduce instead of two (1% slower) and the split into ranks × threads (within 11%) hardly matter. On four nodes, slabs fall 62% behind, and the one-sided rma and pscw variants become 7–23% slower than nonblocking point-to-point. On two A40s, host-staged halos beat CUDA-aware ones inside a node (2.0 vs. 1.8 times one GPU on 3D $512^3$) and tie with them between nodes.

### 8.3 Which parts need the most communication

The halo exchange moves the most data, in proportion to the surface of each box, which is why the box shape matters: slabs moved 7.0 times the data of blocks at 64 ranks and were 21% slower, two thirds of that in the halo exchange and the rest in a slower stencil. The Allreduces move almost nothing, 8 bytes twice per iteration, but they synchronise every rank. Their latency grows with the number of nodes (25 µs on two, about 95 µs on four), and they absorb all the waiting for slower ranks. As $p$ grows, the Allreduce therefore takes over: strong scaling shrinks each box, and the halo shrinks with it, but the Allreduce does not. At 288 ranks the Allreduce is the largest communication cost of every problem, taking 10.5% of an iteration on 3D $512^3$ (compared with 6.1% for the halo) and half of one on 3D $256^3$.

### 8.4 What limits scaling

Within one node, the limit is memory bandwidth. The loop performs 0.19 flop/byte, so 72 cores are only 17–20 times faster than one, and a GPU is faster only in proportion to its memory bandwidth. Communication within a node costs 2–7% of the time (11–15% on 3D $256^3$), and most of its Allreduce part is waiting for the slowest rank. Across nodes the bandwidth adds up: the speedup grows in proportion to the number of nodes, and four nodes give 98–101% of four times the one-node speedup on the large problems. Communication grows to 5–17% of an iteration on three of the four problems (about 70% on 3D $256^3$), mainly in the Allreduces. The multi-node results also depended on fixing the transport first. With Vanda's default MPI settings a message between nodes took 5 ms, which would have made every multi-node run meaningless, so checking and fixing the transport (22 µs with UCX) was a prerequisite for the scaling study.

### 8.5 Limitations

Each node count ran in a single job on one set of nodes, so the variation between nodes is not sampled, and the network is shared with other jobs. The three repeats of a configuration agree closely: for all the runs of Sections 6 and 7, the spread $(\max - \min)/\text{median}$ has a median of 0.3%, a 90th percentile of 3.4% and a maximum of 11%, and every result is their median. The 11% is one slow repeat of the single-rank 3D weak-scaling run, the $T(1)$ of Table \ref{tab:weak}; its other two repeats agree within 2%, so its median is sound. Two single-core runs were noisier: the serial 2D $512^2$ point of Table \ref{tab:e1} (39%) and the one-rank MPI run of 3D $256^3$ (18%). The time-to-solution runs were run only once. The time limits cut the multi-node communication sweeps to one or two runs per variant and left out the hybrid sweep on four nodes. Finally, some of the explanations above are attributions consistent with the measurements rather than direct measurements: the cache effects behind the superlinear points and the slower splits, the inter-node steps behind the Allreduce latency, the missing progress in the overlap variant, and the staging inside the MPI library.

### 8.6 Future work

The number of CG iterations grows with $N$, so even perfect parallel scaling leaves a time to solution that grows as $n^{1 + 1/d}$. A multigrid preconditioner, which approximately solves the problem on a hierarchy of coarser grids and hands CG a better-conditioned system, would make the iteration count nearly independent of $N$. To hide the latency of the Allreduce and the waiting it absorbs at large $p$, pipelined CG [3] overlaps the reduction with the next product with $A$ (using `MPI_Iallreduce`), and s-step CG performs several iterations per reduction. On the GPUs, CUDA streams could overlap the halo exchange with the interior stencil, and remote direct memory access (RDMA) over the nodes' RoCE devices (RDMA over Converged Ethernet) would avoid the host for GPU messages, but the site has not enabled it for MPI.

## 9. Declaration of AI usage

I used an AI assistant in three roles in this project, from the first ideas to the final text:

- **For idea generation and project planning.** It helped me generate and compare candidate algorithms and ways to parallelise them, and helped plan the project: the experiments, the schedule and the use of the NUS clusters.
- **As a coding assistant.** It helped write and debug the solver (the serial, MPI + OpenMP and MPI + CUDA builds), the validation script, the PBS job scripts for the NUS clusters, and the analysis and plotting scripts that produce every figure and table. It also submitted and monitored the cluster jobs through my login, and helped diagnose the problems met on Vanda: the MPI library's missing PBS support, its 5 ms default transport between nodes, and jobs delayed or held by the scheduler.
- **As a report-drafting assistant.** It helped draft and edit the text of this report from the measured results, and reviewed the draft against the grading rubric.

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

The experiments run as PBS jobs on Vanda (`hpc/pbs/`):

- `vanda_cpu_1node.pbs`: E1, $T_1$, E3–E6 and rank placement on one node.
- `vanda_cpu_2node.pbs`: E3–E5 at 128 and 144 ranks.
- `vanda_cpu_4node.pbs`: E3–E5 at 256 and 288 ranks. It also contains E6, which the job's time limit cut off.
- `vanda_cpu_8socket.pbs`: a fallback that would have run the 256- and 288-rank experiments on one socket of each of 8 nodes. It was not needed, and its rank binding failed inside partial-node allocations.
- `vanda_gpu.pbs` and `vanda_gpu_2node.pbs`: validation on the GPU, and E7.

Each script lists its cost in its header. Together they used about 600 of the 10,000 CPU-hours and 4 of the 1,000 GPU-hours of a student's yearly Vanda budget.
