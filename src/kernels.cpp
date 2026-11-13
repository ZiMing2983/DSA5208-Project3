#include "kernels.hpp"

#include <cstdlib>
#include <new>

namespace {

// q = A u on the box [i0,i1] x [j0,j1] x [k0,k1] (inclusive; k ignored in 2D).
template <int DIM>
void stencil_box(const Grid& g, const Block& b, const double* __restrict u, double* __restrict q,
                 int i0, int i1, int j0, int j1, int k0, int k1) {
  if (i0 > i1 || j0 > j1 || (DIM == 3 && k0 > k1)) return;
  const long si = b.si, sj = b.sj;
  const double cx = g.c[0], cy = g.c[1], cz = g.c[2], dg = g.diag;
  if constexpr (DIM == 3) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = i0; i <= i1; ++i)
      for (int j = j0; j <= j1; ++j) {
        const long base = (long)i * si + (long)j * sj;
#pragma omp simd
        for (int k = k0; k <= k1; ++k) {
          const long id = base + k;
          q[id] = dg * u[id] - cx * (u[id - si] + u[id + si]) - cy * (u[id - sj] + u[id + sj]) -
                  cz * (u[id - 1] + u[id + 1]);
        }
      }
  } else {
    (void)k0;
    (void)k1;
    (void)cz;
#pragma omp parallel for schedule(static)
    for (int i = i0; i <= i1; ++i) {
      const long base = (long)i * si;
#pragma omp simd
      for (int j = j0; j <= j1; ++j) {
        const long id = base + j;
        q[id] = dg * u[id] - cx * (u[id - si] + u[id + si]) - cy * (u[id - 1] + u[id + 1]);
      }
    }
  }
}

template <int DIM>
void apply_t(const Grid& g, const Block& b, const double* u, double* q, Region region) {
  const int n0 = b.n[0], n1 = b.n[1], n2 = b.n[2];
  const int k0 = b.k0, k1 = b.k1;
  if (region == Region::All) {
    stencil_box<DIM>(g, b, u, q, 1, n0, 1, n1, k0, k1);
    return;
  }
  if (region == Region::Interior) {
    stencil_box<DIM>(g, b, u, q, 2, n0 - 1, 2, n1 - 1, 2, n2 - 1);
    return;
  }
  // Boundary shell, split into disjoint boxes: the i-faces, then the j-faces of the remaining
  // i range, then (3D) the k-faces of the remaining i and j ranges.
  stencil_box<DIM>(g, b, u, q, 1, 1, 1, n1, k0, k1);
  if (n0 > 1) stencil_box<DIM>(g, b, u, q, n0, n0, 1, n1, k0, k1);
  stencil_box<DIM>(g, b, u, q, 2, n0 - 1, 1, 1, k0, k1);
  if (n1 > 1) stencil_box<DIM>(g, b, u, q, 2, n0 - 1, n1, n1, k0, k1);
  if constexpr (DIM == 3) {
    stencil_box<DIM>(g, b, u, q, 2, n0 - 1, 2, n1 - 1, 1, 1);
    if (n2 > 1) stencil_box<DIM>(g, b, u, q, 2, n0 - 1, 2, n1 - 1, n2, n2);
  }
}

}  // namespace

void apply_A(const Grid& g, const Block& b, const double* u, double* q, Region region) {
  if (b.dim == 3)
    apply_t<3>(g, b, u, q, region);
  else
    apply_t<2>(g, b, u, q, region);
}

// The vector kernels below share one loop structure: (i, j) rows in parallel, the contiguous
// innermost index as a SIMD loop. In 2D the innermost index is j; in 3D it is k.
#define ROWS_BEGIN(b)                                                \
  const int n0_ = (b).n[0], n1_ = (b).n[1];                          \
  const int k0_ = (b).k0, k1_ = (b).k1;                              \
  const long si_ = (b).si, sj_ = (b).sj;                             \
  (void)k0_;                                                         \
  (void)k1_;

double dot(const Block& b, const double* x, const double* y) {
  ROWS_BEGIN(b)
  double s = 0.0;
  if (b.dim == 3) {
#pragma omp parallel for collapse(2) reduction(+ : s) schedule(static)
    for (int i = 1; i <= n0_; ++i)
      for (int j = 1; j <= n1_; ++j) {
        const long base = (long)i * si_ + (long)j * sj_;
        double t = 0.0;
#pragma omp simd reduction(+ : t)
        for (int k = k0_; k <= k1_; ++k) t += x[base + k] * y[base + k];
        s += t;
      }
  } else {
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (int i = 1; i <= n0_; ++i) {
      const long base = (long)i * si_;
      double t = 0.0;
#pragma omp simd reduction(+ : t)
      for (int j = 1; j <= n1_; ++j) t += x[base + j] * y[base + j];
      s += t;
    }
  }
  return s;
}

void dot2(const Block& b, const double* r, const double* w, double out[2]) {
  ROWS_BEGIN(b)
  double rr = 0.0, wr = 0.0;
  if (b.dim == 3) {
#pragma omp parallel for collapse(2) reduction(+ : rr, wr) schedule(static)
    for (int i = 1; i <= n0_; ++i)
      for (int j = 1; j <= n1_; ++j) {
        const long base = (long)i * si_ + (long)j * sj_;
        double t = 0.0, u = 0.0;
#pragma omp simd reduction(+ : t, u)
        for (int k = k0_; k <= k1_; ++k) {
          const double rv = r[base + k];
          t += rv * rv;
          u += w[base + k] * rv;
        }
        rr += t;
        wr += u;
      }
  } else {
#pragma omp parallel for reduction(+ : rr, wr) schedule(static)
    for (int i = 1; i <= n0_; ++i) {
      const long base = (long)i * si_;
      double t = 0.0, u = 0.0;
#pragma omp simd reduction(+ : t, u)
      for (int j = 1; j <= n1_; ++j) {
        const double rv = r[base + j];
        t += rv * rv;
        u += w[base + j] * rv;
      }
      rr += t;
      wr += u;
    }
  }
  out[0] = rr;
  out[1] = wr;
}

double update_xr(const Block& b, double alpha, const double* __restrict p,
                 const double* __restrict q, double* __restrict x, double* __restrict r) {
  ROWS_BEGIN(b)
  double s = 0.0;
  if (b.dim == 3) {
#pragma omp parallel for collapse(2) reduction(+ : s) schedule(static)
    for (int i = 1; i <= n0_; ++i)
      for (int j = 1; j <= n1_; ++j) {
        const long base = (long)i * si_ + (long)j * sj_;
        double t = 0.0;
#pragma omp simd reduction(+ : t)
        for (int k = k0_; k <= k1_; ++k) {
          const long id = base + k;
          x[id] += alpha * p[id];
          const double rn = r[id] - alpha * q[id];
          r[id] = rn;
          t += rn * rn;
        }
        s += t;
      }
  } else {
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (int i = 1; i <= n0_; ++i) {
      const long base = (long)i * si_;
      double t = 0.0;
#pragma omp simd reduction(+ : t)
      for (int j = 1; j <= n1_; ++j) {
        const long id = base + j;
        x[id] += alpha * p[id];
        const double rn = r[id] - alpha * q[id];
        r[id] = rn;
        t += rn * rn;
      }
      s += t;
    }
  }
  return s;
}

// ---- Reproducible (double-double) reductions ---------------------------------------------
// Each row is summed into four independent DD accumulators (for instruction-level
// parallelism) and the row results are combined with dd_add. Each thread keeps its own DD
// partial; the partials are added in a critical section (Lecture 9 synchronisation). An OpenMP
// user-defined reduction would be neater, but nvc++ (Vanda's compiler) does not support
// `declare reduction`. The order in which threads enter the critical section does not matter,
// because DD addition gives the same rounded result in any order.

namespace {

// Sum of f(id) over one contiguous row id = base + k, k in [k0, k1].
template <class F>
inline DD row_sum_dd(long base, int k0, int k1, F&& f) {
  DD a0, a1, a2, a3;
  int k = k0;
  for (; k + 3 <= k1; k += 4) {
    dd_acc(a0, f(base + k));
    dd_acc(a1, f(base + k + 1));
    dd_acc(a2, f(base + k + 2));
    dd_acc(a3, f(base + k + 3));
  }
  for (; k <= k1; ++k) dd_acc(a0, f(base + k));
  return dd_add(dd_add(a0, a1), dd_add(a2, a3));
}

// Sum over the interior of f(id); f may also update vectors (it is called once per point).
template <class F>
DD interior_sum_dd(const Block& b, F f) {
  const int n0 = b.n[0], n1 = b.n[1];
  const long si = b.si, sj = b.sj;
  DD total;
#pragma omp parallel
  {
    DD acc;  // this thread's partial sum
    if (b.dim == 3) {
      const int k0 = b.k0, k1 = b.k1;
#pragma omp for collapse(2) schedule(static) nowait
      for (int i = 1; i <= n0; ++i)
        for (int j = 1; j <= n1; ++j)
          acc = dd_add(acc, row_sum_dd((long)i * si + (long)j * sj, k0, k1, f));
    } else {
#pragma omp for schedule(static) nowait
      for (int i = 1; i <= n0; ++i) acc = dd_add(acc, row_sum_dd((long)i * si, 1, n1, f));
    }
#pragma omp critical(dd_sum)
    total = dd_add(total, acc);
  }
  return total;
}

}  // namespace

DD dot_dd(const Block& b, const double* x, const double* y) {
  return interior_sum_dd(b, [=](long id) { return x[id] * y[id]; });
}

void dot2_dd(const Block& b, const double* r, const double* w, DD out[2]) {
  // One pass over r and w (like dot2), two DD sums per row.
  const int n0 = b.n[0], n1 = b.n[1];
  const long si = b.si, sj = b.sj;
  const int k0 = (b.dim == 3) ? b.k0 : 1, k1 = (b.dim == 3) ? b.k1 : n1;
  const int jmax = (b.dim == 3) ? n1 : 1;
  DD rr_total, wr_total;
#pragma omp parallel
  {
    DD rr, wr;  // this thread's partial sums
#pragma omp for collapse(2) schedule(static) nowait
    for (int i = 1; i <= n0; ++i)
      for (int j = 1; j <= jmax; ++j) {
        const long base = (long)i * si + (b.dim == 3 ? (long)j * sj : 0);
        DD a0, a1, c0, c1;
        int k = k0;
        for (; k + 1 <= k1; k += 2) {
          const double r0 = r[base + k], r1 = r[base + k + 1];
          dd_acc(a0, r0 * r0);
          dd_acc(a1, r1 * r1);
          dd_acc(c0, w[base + k] * r0);
          dd_acc(c1, w[base + k + 1] * r1);
        }
        for (; k <= k1; ++k) {
          const double r0 = r[base + k];
          dd_acc(a0, r0 * r0);
          dd_acc(c0, w[base + k] * r0);
        }
        rr = dd_add(rr, dd_add(a0, a1));
        wr = dd_add(wr, dd_add(c0, c1));
      }
#pragma omp critical(dd_sum)
    {
      rr_total = dd_add(rr_total, rr);
      wr_total = dd_add(wr_total, wr);
    }
  }
  out[0] = rr_total;
  out[1] = wr_total;
}

DD update_xr_dd(const Block& b, double alpha, const double* p, const double* q, double* x,
                double* r) {
  return interior_sum_dd(b, [=](long id) {
    x[id] += alpha * p[id];
    const double rn = r[id] - alpha * q[id];
    r[id] = rn;
    return rn * rn;
  });
}

void update_p(const Block& b, double beta, const double* __restrict r, double* __restrict p) {
  ROWS_BEGIN(b)
  if (b.dim == 3) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= n0_; ++i)
      for (int j = 1; j <= n1_; ++j) {
        const long base = (long)i * si_ + (long)j * sj_;
#pragma omp simd
        for (int k = k0_; k <= k1_; ++k) p[base + k] = r[base + k] + beta * p[base + k];
      }
  } else {
#pragma omp parallel for schedule(static)
    for (int i = 1; i <= n0_; ++i) {
      const long base = (long)i * si_;
#pragma omp simd
      for (int j = 1; j <= n1_; ++j) p[base + j] = r[base + j] + beta * p[base + j];
    }
  }
}

void update_cg_fused(const Block& b, double alpha, double beta, const double* __restrict w,
                     double* __restrict r, double* __restrict p, double* __restrict s,
                     double* __restrict x) {
  ROWS_BEGIN(b)
  if (b.dim == 3) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= n0_; ++i)
      for (int j = 1; j <= n1_; ++j) {
        const long base = (long)i * si_ + (long)j * sj_;
#pragma omp simd
        for (int k = k0_; k <= k1_; ++k) {
          const long id = base + k;
          const double pn = r[id] + beta * p[id];
          const double sn = w[id] + beta * s[id];
          p[id] = pn;
          s[id] = sn;
          x[id] += alpha * pn;
          r[id] -= alpha * sn;
        }
      }
  } else {
#pragma omp parallel for schedule(static)
    for (int i = 1; i <= n0_; ++i) {
      const long base = (long)i * si_;
#pragma omp simd
      for (int j = 1; j <= n1_; ++j) {
        const long id = base + j;
        const double pn = r[id] + beta * p[id];
        const double sn = w[id] + beta * s[id];
        p[id] = pn;
        s[id] = sn;
        x[id] += alpha * pn;
        r[id] -= alpha * sn;
      }
    }
  }
}

void copy_interior(const Block& b, const double* __restrict src, double* __restrict dst) {
  ROWS_BEGIN(b)
  if (b.dim == 3) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= n0_; ++i)
      for (int j = 1; j <= n1_; ++j) {
        const long base = (long)i * si_ + (long)j * sj_;
#pragma omp simd
        for (int k = k0_; k <= k1_; ++k) dst[base + k] = src[base + k];
      }
  } else {
#pragma omp parallel for schedule(static)
    for (int i = 1; i <= n0_; ++i) {
      const long base = (long)i * si_;
#pragma omp simd
      for (int j = 1; j <= n1_; ++j) dst[base + j] = src[base + j];
    }
  }
}

void zero_interior(const Block& b, double* v) {
  ROWS_BEGIN(b)
  if (b.dim == 3) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= n0_; ++i)
      for (int j = 1; j <= n1_; ++j) {
        const long base = (long)i * si_ + (long)j * sj_;
#pragma omp simd
        for (int k = k0_; k <= k1_; ++k) v[base + k] = 0.0;
      }
  } else {
#pragma omp parallel for schedule(static)
    for (int i = 1; i <= n0_; ++i) {
      const long base = (long)i * si_;
#pragma omp simd
      for (int j = 1; j <= n1_; ++j) v[base + j] = 0.0;
    }
  }
}

double* alloc_vector(const Block& b) {
  const std::size_t bytes = ((b.size * sizeof(double) + 63) / 64) * 64;
  double* v = static_cast<double*>(std::aligned_alloc(64, bytes));
  if (!v) throw std::bad_alloc();
  // Zero everything (ghosts included) with the same static row split as the compute loops, so
  // each page is first touched, and therefore placed, by the thread that will use it.
  const int sx = b.sx, sy = b.sy, sz = b.sz;
  if (b.dim == 3) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < sx; ++i)
      for (int j = 0; j < sy; ++j) {
        double* row = v + (long)i * b.si + (long)j * b.sj;
        for (int k = 0; k < sz; ++k) row[k] = 0.0;
      }
  } else {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < sx; ++i) {
      double* row = v + (long)i * b.si;
      for (int j = 0; j < sy; ++j) row[j] = 0.0;
    }
  }
  // Rounding up to 64 bytes may leave a few doubles past b.size; they are never read.
  return v;
}

void free_vector(double* v) { std::free(v); }
