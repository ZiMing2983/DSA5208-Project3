// Matrix-free Conjugate Gradient on one block. Communication goes through the Comm interface:
// the serial build uses SerialComm (no-ops), the MPI build uses MpiComm (halo.hpp).
#pragma once
#include "dd.hpp"
#include "grid.hpp"

struct Comm {
  virtual ~Comm() = default;
  // Fill u's ghost cells from the neighbouring blocks. halo_begin starts the exchange and
  // halo_end completes it; the overlap variant computes interior points in between.
  virtual void halo_begin(double* u) = 0;
  virtual void halo_end(double* u) = 0;
  virtual bool overlap() const { return false; }
  virtual void allreduce_sum(double* v, int n) = 0;  // in place, over all ranks
  virtual void allreduce_dd(DD* v, int n) = 0;       // reproducible (double-double) sum
};

struct SerialComm final : Comm {
  void halo_begin(double*) override {}
  void halo_end(double*) override {}
  void allreduce_sum(double*, int) override {}
  void allreduce_dd(DD*, int) override {}
};

struct CGOptions {
  bool fused = false;  // false: standard CG, 2 Allreduces per iteration
                       // true: Chronopoulos-Gear CG, one 2-value Allreduce per iteration
  double tol = 1e-8;   // > 0: iterate until ||r|| / ||b|| < tol (at most max_iters)
                       // = 0: fixed-iteration mode (performance runs): warmup + iters
  int iters = 100000;  // max iterations (tol > 0) or timed iterations (tol = 0)
  int warmup = 10;     // untimed iterations before the timed ones (fixed mode only)
  bool repro = false;  // double-double dot products: bit-identical results for any p/threads
};

struct CGStats {
  int iters = 0;         // iterations counted (timed iterations in fixed mode)
  double rel_res = 0.0;  // ||r|| / ||b|| from the CG recurrence
  double t_total = 0.0;  // wall time of the counted iterations, seconds
  double t_comp = 0.0;   // stencil + vector updates + local dot products
  double t_halo = 0.0;   // halo exchange (in the overlap variant: only the unhidden part)
  double t_reduce = 0.0; // MPI_Allreduce
};

double wtime();  // seconds, monotonic

// q = A u including the halo exchange of u; adds the phase times to *t when t != nullptr.
void matvec(const Grid& g, const Block& b, Comm& comm, double* u, double* q, CGStats* t);

// Solves A x = rhs from x = 0. rhs and x are block vectors (alloc_vector).
CGStats cg_solve(const Grid& g, const Block& b, Comm& comm, const double* rhs, double* x,
                 const CGOptions& opt);
