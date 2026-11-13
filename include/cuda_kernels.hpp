// GPU kernels of the MPI+CUDA build (src/cuda/kernels.cu). All pointers are device pointers;
// every kernel runs on the default stream. The vector layout is the CPU Block layout.
#pragma once
#include "grid.hpp"

struct DevBlock {
  int dim, n0, n1, n2, k0, k1;
  long si, sj;
};
inline DevBlock dev_block(const Block& b) {
  return DevBlock{b.dim, b.n[0], b.n[1], b.n[2], b.k0, b.k1, b.si, b.sj};
}

struct DevCoef {
  double cx, cy, cz, dg;
};
inline DevCoef dev_coef(const Grid& g) { return DevCoef{g.c[0], g.c[1], g.c[2], g.diag}; }

// q = A u on all interior points (same expression and order as the CPU stencil).
void gpu_apply_A(const DevBlock& b, const DevCoef& c, const double* u, double* q);
// out[0] += x.y over the interior (out is a device scalar the caller zeroes).
void gpu_dot(const DevBlock& b, const double* x, const double* y, double* out);
// x += alpha p;  r -= alpha q;  out[0] += (new r).(new r)
void gpu_update_xr(const DevBlock& b, double alpha, const double* p, const double* q, double* x,
                   double* r, double* out);
// p = r + beta p
void gpu_update_p(const DevBlock& b, double beta, const double* r, double* p);
// Face d at interior/ghost layer `layer` <-> contiguous buffer (CPU pack_face order).
void gpu_pack(const DevBlock& b, int d, int layer, const double* u, double* buf);
void gpu_unpack(const DevBlock& b, int d, int layer, const double* buf, double* u);
// Aborts with a message if a kernel launch failed.
void gpu_check(const char* where);
