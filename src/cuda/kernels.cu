// CUDA kernels for the matrix-free CG (no cuBLAS: BLAS libraries are not allowed).
// Built with --fmad=false so the stencil rounds exactly like the CPU build (-ffp-contract=off).
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "cuda_kernels.hpp"

namespace {

constexpr int kThreads = 256;

// Interior launch: threads along the contiguous index (k in 3D, j in 2D), a second thread
// dimension along j (3D only), and blockIdx.z along i (grid-strided for the reductions).
struct Launch {
  dim3 grid, block;
};

Launch interior_launch(const DevBlock& b, int max_z) {
  Launch L;
  if (b.dim == 3) {
    L.block = dim3(32, 8, 1);
    L.grid = dim3((b.n2 + 31) / 32, (b.n1 + 7) / 8, b.n0 < max_z ? b.n0 : max_z);
  } else {
    L.block = dim3(kThreads, 1, 1);
    L.grid = dim3((b.n1 + kThreads - 1) / kThreads, 1, b.n0 < max_z ? b.n0 : max_z);
  }
  return L;
}

// Storage index of this thread's point in plane i, or -1 if the thread is outside the block.
__device__ __forceinline__ long point(const DevBlock& b, int i) {
  const int t_in = blockIdx.x * blockDim.x + threadIdx.x;
  const int t_mid = blockIdx.y * blockDim.y + threadIdx.y;
  if (b.dim == 3) {
    if (t_in >= b.n2 || t_mid >= b.n1) return -1;
    return (long)i * b.si + (long)(1 + t_mid) * b.sj + (1 + t_in);
  }
  if (t_in >= b.n1) return -1;
  return (long)i * b.si + (1 + t_in);
}

__device__ __forceinline__ double warp_sum(double v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

// Block-wide sum of v, added to *out with one atomic per block. Every thread must call it.
__device__ void block_sum_to(double v, double* out) {
  __shared__ double part[32];
  const int t = threadIdx.y * blockDim.x + threadIdx.x;
  const int lane = t & 31, warp = t >> 5;
  v = warp_sum(v);
  if (lane == 0) part[warp] = v;
  __syncthreads();
  if (warp == 0) {
    const int nwarps = (blockDim.x * blockDim.y + 31) >> 5;
    v = (lane < nwarps) ? part[lane] : 0.0;
    v = warp_sum(v);
    if (lane == 0) atomicAdd(out, v);
  }
}

__global__ void k_apply(DevBlock b, DevCoef c, const double* __restrict__ u,
                        double* __restrict__ q) {
  for (int i = 1 + blockIdx.z; i <= b.n0; i += gridDim.z) {
    const long id = point(b, i);
    if (id < 0) continue;
    double v = c.dg * u[id] - c.cx * (u[id - b.si] + u[id + b.si]) -
               c.cy * (u[id - b.sj] + u[id + b.sj]);
    if (b.dim == 3) v = v - c.cz * (u[id - 1] + u[id + 1]);
    q[id] = v;
  }
}

__global__ void k_dot(DevBlock b, const double* __restrict__ x, const double* __restrict__ y,
                      double* out) {
  double acc = 0.0;
  for (int i = 1 + blockIdx.z; i <= b.n0; i += gridDim.z) {
    const long id = point(b, i);
    if (id >= 0) acc += x[id] * y[id];
  }
  block_sum_to(acc, out);
}

__global__ void k_update_xr(DevBlock b, double alpha, const double* __restrict__ p,
                            const double* __restrict__ q, double* __restrict__ x,
                            double* __restrict__ r, double* out) {
  double acc = 0.0;
  for (int i = 1 + blockIdx.z; i <= b.n0; i += gridDim.z) {
    const long id = point(b, i);
    if (id < 0) continue;
    x[id] += alpha * p[id];
    const double rn = r[id] - alpha * q[id];
    r[id] = rn;
    acc += rn * rn;
  }
  block_sum_to(acc, out);
}

__global__ void k_update_p(DevBlock b, double beta, const double* __restrict__ r,
                           double* __restrict__ p) {
  for (int i = 1 + blockIdx.z; i <= b.n0; i += gridDim.z) {
    const long id = point(b, i);
    if (id >= 0) p[id] = r[id] + beta * p[id];
  }
}

// Face d as an A x C array (C fastest), in the CPU pack_face order:
//   d = 0: (j, k)   d = 1: (i, k)   d = 2: (i, j)      (k is a single 0 in 2D)
__global__ void k_face(DevBlock b, int d, int layer, const double* __restrict__ src,
                       double* __restrict__ dst, int pack) {
  const int nk = b.k1 - b.k0 + 1;
  const int C = (d == 2) ? b.n1 : nk;
  const long A = (d == 0) ? b.n1 : b.n0;
  const long f = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= A * C) return;
  const int a = (int)(f / C), cc = (int)(f % C);
  long id;
  if (d == 0)
    id = (long)layer * b.si + (long)(1 + a) * b.sj + (b.k0 + cc);
  else if (d == 1)
    id = (long)(1 + a) * b.si + (long)layer * b.sj + (b.k0 + cc);
  else
    id = (long)(1 + a) * b.si + (long)(1 + cc) * b.sj + layer;
  if (pack)
    dst[f] = src[id];
  else
    dst[id] = src[f];
}

long face_len(const DevBlock& b, int d) {
  const long nk = b.k1 - b.k0 + 1;
  if (d == 0) return (long)b.n1 * nk;
  if (d == 1) return (long)b.n0 * nk;
  return (long)b.n0 * b.n1;
}

}  // namespace

void gpu_apply_A(const DevBlock& b, const DevCoef& c, const double* u, double* q) {
  const Launch L = interior_launch(b, 65535);
  k_apply<<<L.grid, L.block>>>(b, c, u, q);
}

void gpu_dot(const DevBlock& b, const double* x, const double* y, double* out) {
  const Launch L = interior_launch(b, 16);
  k_dot<<<L.grid, L.block>>>(b, x, y, out);
}

void gpu_update_xr(const DevBlock& b, double alpha, const double* p, const double* q, double* x,
                   double* r, double* out) {
  const Launch L = interior_launch(b, 16);
  k_update_xr<<<L.grid, L.block>>>(b, alpha, p, q, x, r, out);
}

void gpu_update_p(const DevBlock& b, double beta, const double* r, double* p) {
  const Launch L = interior_launch(b, 65535);
  k_update_p<<<L.grid, L.block>>>(b, beta, r, p);
}

void gpu_pack(const DevBlock& b, int d, int layer, const double* u, double* buf) {
  const long m = face_len(b, d);
  k_face<<<(unsigned)((m + kThreads - 1) / kThreads), kThreads>>>(b, d, layer, u, buf, 1);
}

void gpu_unpack(const DevBlock& b, int d, int layer, const double* buf, double* u) {
  const long m = face_len(b, d);
  k_face<<<(unsigned)((m + kThreads - 1) / kThreads), kThreads>>>(b, d, layer, buf, u, 0);
}

void gpu_check(const char* where) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error after %s: %s\n", where, cudaGetErrorString(e));
    std::abort();
  }
}
