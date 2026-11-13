#include "cg.hpp"

#include <chrono>
#include <cmath>

#include "kernels.hpp"

double wtime() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void matvec(const Grid& g, const Block& b, Comm& comm, double* u, double* q, CGStats* t) {
  const double t0 = wtime();
  comm.halo_begin(u);
  if (comm.overlap()) {
    const double t1 = wtime();
    apply_A(g, b, u, q, Region::Interior);  // needs no ghost cells
    const double t2 = wtime();
    comm.halo_end(u);
    const double t3 = wtime();
    apply_A(g, b, u, q, Region::Boundary);
    const double t4 = wtime();
    if (t) {
      t->t_halo += (t1 - t0) + (t3 - t2);
      t->t_comp += (t2 - t1) + (t4 - t3);
    }
  } else {
    comm.halo_end(u);
    const double t1 = wtime();
    apply_A(g, b, u, q, Region::All);
    const double t2 = wtime();
    if (t) {
      t->t_halo += t1 - t0;
      t->t_comp += t2 - t1;
    }
  }
}

namespace {

// In fixed-iteration mode a small problem can converge long before the last iteration; the
// iterates then underflow towards denormals, which are very slow, and may end in 0/0.
// Restarting from x = 0 below this residual keeps the work per iteration unchanged.
constexpr double kRestartBelow = 1e-13;

struct Workspace {
  const Block& b;
  double* v[4] = {nullptr, nullptr, nullptr, nullptr};
  explicit Workspace(const Block& blk, int count) : b(blk) {
    for (int i = 0; i < count; ++i) v[i] = alloc_vector(b);
  }
  ~Workspace() {
    for (double* p : v)
      if (p) free_vector(p);
  }
};

// Global dot products over all ranks: the local part (compute time) and the Allreduce
// (reduction time) are added to *T when T != nullptr. With repro, the local sums are
// double-double and the Allreduce uses the custom double-double MPI_Op (dd.hpp).
struct Reducer {
  const Block& b;
  Comm& comm;
  bool repro;

  double dot(const double* x, const double* y, CGStats* T) {
    const double t0 = wtime();
    double v = 0.0;
    DD d;
    if (repro)
      d = dot_dd(b, x, y);
    else
      v = ::dot(b, x, y);
    const double t1 = wtime();
    if (repro) {
      comm.allreduce_dd(&d, 1);
      v = dd_value(d);
    } else {
      comm.allreduce_sum(&v, 1);
    }
    if (T) {
      T->t_comp += t1 - t0;
      T->t_reduce += wtime() - t1;
    }
    return v;
  }

  // x += alpha p; r -= alpha q; returns the global r.r of the new r.
  double update_xr(double alpha, const double* p, const double* q, double* x, double* r,
                   CGStats* T) {
    const double t0 = wtime();
    double v = 0.0;
    DD d;
    if (repro)
      d = update_xr_dd(b, alpha, p, q, x, r);
    else
      v = ::update_xr(b, alpha, p, q, x, r);
    const double t1 = wtime();
    if (repro) {
      comm.allreduce_dd(&d, 1);
      v = dd_value(d);
    } else {
      comm.allreduce_sum(&v, 1);
    }
    if (T) {
      T->t_comp += t1 - t0;
      T->t_reduce += wtime() - t1;
    }
    return v;
  }

  // {r.r, w.r} over all ranks, with one Allreduce of two values.
  void dot2(const double* r, const double* w, double out[2], CGStats* T) {
    const double t0 = wtime();
    DD d[2];
    if (repro)
      dot2_dd(b, r, w, d);
    else
      ::dot2(b, r, w, out);
    const double t1 = wtime();
    if (repro) {
      comm.allreduce_dd(d, 2);
      out[0] = dd_value(d[0]);
      out[1] = dd_value(d[1]);
    } else {
      comm.allreduce_sum(out, 2);
    }
    if (T) {
      T->t_comp += t1 - t0;
      T->t_reduce += wtime() - t1;
    }
  }
};

CGStats cg_standard(const Grid& g, const Block& b, Comm& comm, const double* rhs, double* x,
                    const CGOptions& opt) {
  Workspace ws(b, 3);
  double *r = ws.v[0], *p = ws.v[1], *q = ws.v[2];
  Reducer red{b, comm, opt.repro};
  const bool fixed = !(opt.tol > 0);
  const int warm = fixed ? opt.warmup : 0;
  const int total = warm + opt.iters;

  zero_interior(b, x);
  copy_interior(b, rhs, r);
  copy_interior(b, r, p);
  double rho = red.dot(r, r, nullptr);
  const double rho0 = rho, bnorm = std::sqrt(rho);

  CGStats st, discard;
  CGStats* T = (warm > 0) ? &discard : &st;
  if (bnorm == 0.0) return st;
  double t_start = wtime(), rel = 1.0;
  for (int it = 0; it < total; ++it) {
    if (it == warm && warm > 0) {
      T = &st;
      t_start = wtime();
    }
    matvec(g, b, comm, p, q, T);
    const double pq = red.dot(p, q, T);
    const double alpha = rho / pq;
    const double rr = red.update_xr(alpha, p, q, x, r, T);
    if (it >= warm) ++st.iters;
    rel = std::sqrt(rr) / bnorm;
    if (!fixed && rel < opt.tol) break;
    const double t5 = wtime();
    if (fixed && rel < kRestartBelow) {
      zero_interior(b, x);
      copy_interior(b, rhs, r);
      copy_interior(b, r, p);
      rho = rho0;
    } else {
      update_p(b, rr / rho, r, p);
      rho = rr;
    }
    T->t_comp += wtime() - t5;
  }
  st.t_total = wtime() - t_start;
  st.rel_res = rel;
  return st;
}

// Chronopoulos-Gear CG (one Allreduce of two values per iteration):
//   r = b, w = A r, gamma = r.r, delta = w.r, alpha = gamma/delta, beta = 0
//   loop: p = r + beta p; s = w + beta s; x += alpha p; r -= alpha s; w = A r
//         gamma' = r.r, delta = w.r                                  [one Allreduce]
//         beta = gamma'/gamma; alpha = gamma' / (delta - beta gamma' / alpha)
CGStats cg_fused(const Grid& g, const Block& b, Comm& comm, const double* rhs, double* x,
                 const CGOptions& opt) {
  Workspace ws(b, 4);
  double *r = ws.v[0], *w = ws.v[1], *p = ws.v[2], *s = ws.v[3];
  Reducer red{b, comm, opt.repro};
  const bool fixed = !(opt.tol > 0);
  const int warm = fixed ? opt.warmup : 0;
  const int total = warm + opt.iters;

  CGStats st, discard;
  CGStats* T = (warm > 0) ? &discard : &st;
  double gamma = 0.0, alpha = 0.0, beta = 0.0, bnorm = 0.0;
  auto start = [&](CGStats* timers) {  // (re)start from x = 0
    zero_interior(b, x);
    zero_interior(b, p);
    zero_interior(b, s);
    copy_interior(b, rhs, r);
    matvec(g, b, comm, r, w, timers);
    double gd[2];
    red.dot2(r, w, gd, timers);
    gamma = gd[0];
    alpha = gd[0] / gd[1];
    beta = 0.0;
    return std::sqrt(gd[0]);
  };
  bnorm = start(nullptr);
  if (bnorm == 0.0) return st;

  double t_start = wtime(), rel = 1.0;
  for (int it = 0; it < total; ++it) {
    if (it == warm && warm > 0) {
      T = &st;
      t_start = wtime();
    }
    const double t0 = wtime();
    update_cg_fused(b, alpha, beta, w, r, p, s, x);
    T->t_comp += wtime() - t0;
    matvec(g, b, comm, r, w, T);
    double gd[2];
    red.dot2(r, w, gd, T);
    if (it >= warm) ++st.iters;
    rel = std::sqrt(gd[0]) / bnorm;
    if (!fixed && rel < opt.tol) break;
    if (fixed && rel < kRestartBelow) {
      start(T);
      continue;
    }
    beta = gd[0] / gamma;
    alpha = gd[0] / (gd[1] - beta * gd[0] / alpha);
    gamma = gd[0];
  }
  st.t_total = wtime() - t_start;
  st.rel_res = rel;
  return st;
}

}  // namespace

CGStats cg_solve(const Grid& g, const Block& b, Comm& comm, const double* rhs, double* x,
                 const CGOptions& opt) {
  return opt.fused ? cg_fused(g, b, comm, rhs, x, opt) : cg_standard(g, b, comm, rhs, x, opt);
}
