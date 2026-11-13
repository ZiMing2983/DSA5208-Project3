// MPI (+OpenMP) driver.
//   mpirun -np 64 poisson_mpi --dim 3 --n 512 --iters 200 --csv results/strong.csv
//   mpirun -np 8  poisson_mpi --dim 3 --nlocal 128 --iters 200          weak scaling
//   mpirun -np 6  poisson_mpi --dim 3 --nx 37 --ny 29 --nz 23 --test apply --halo overlap
//   mpirun -np 6  poisson_mpi --dim 2 --n 101 --test solve --cg fused
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstdio>
#include <vector>

#include "cg.hpp"
#include "grid.hpp"
#include "halo.hpp"
#include "kernels.hpp"
#include "options.hpp"

namespace {

int num_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

CGOptions cg_options(const Options& o) {
  CGOptions c;
  c.fused = (o.cg == "fused");
  c.repro = (o.dot == "repro");
  c.tol = o.tol;
  c.iters = o.iters;
  c.warmup = o.warmup;
  return c;
}

struct Problem {
  Grid g;
  Decomp d;
};

Problem setup(const Options& o, int nx, int ny, int nz) {
  int p = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  int dims[3];
  decomp_dims(p, o.dim, o.decomp, dims);
  if (o.nlocal > 0) {
    nx = o.nlocal * dims[0];
    ny = o.nlocal * dims[1];
    nz = o.nlocal * dims[2];
  }
  Problem pr;
  pr.g = make_grid(o.dim, nx, ny, nz);
  pr.d = make_decomp(pr.g, MPI_COMM_WORLD, dims);
  return pr;
}

struct Solve {
  CGStats st;      // timers: t_total is the maximum over ranks, the phases are rank averages
  double err = 0;  // max |x - u_ex| over all ranks
};

Solve distributed_solve(const Options& o, Problem& pr, std::vector<double>* gathered) {
  const Block& b = pr.d.block;
  double* rhs = alloc_vector(b);
  double* x = alloc_vector(b);
  fill_rhs(pr.g, b, rhs);
  Solve s;
  {
    MpiComm comm(pr.d, parse_halo(o.halo));
    MPI_Barrier(pr.d.cart);
    s.st = cg_solve(pr.g, b, comm, rhs, x, cg_options(o));
  }
  s.err = max_error(pr.g, b, x);
  MPI_Allreduce(MPI_IN_PLACE, &s.err, 1, MPI_DOUBLE, MPI_MAX, pr.d.cart);
  // t_total: the slowest rank. Phases: averages over ranks, so they add up to about t_total;
  // a rank waiting for slower ranks spends that time inside the Allreduce (t_reduce).
  MPI_Allreduce(MPI_IN_PLACE, &s.st.t_total, 1, MPI_DOUBLE, MPI_MAX, pr.d.cart);
  double ph[3] = {s.st.t_comp, s.st.t_halo, s.st.t_reduce};
  MPI_Allreduce(MPI_IN_PLACE, ph, 3, MPI_DOUBLE, MPI_SUM, pr.d.cart);
  s.st.t_comp = ph[0] / pr.d.size;
  s.st.t_halo = ph[1] / pr.d.size;
  s.st.t_reduce = ph[2] / pr.d.size;
  if (gathered) *gathered = gather_to_root(pr.g, pr.d, x);
  free_vector(rhs);
  free_vector(x);
  return s;
}

Result make_result(const Options& o, const Problem& pr, const Solve& s) {
  Result r;
  r.machine = machine_name(o);
  r.threads = num_threads();
  r.backend = r.threads > 1 ? "mpi+omp" : "mpi";
  r.dim = pr.g.dim;
  r.nx = pr.g.N[0];
  r.ny = pr.g.N[1];
  r.nz = pr.g.N[2];
  r.n = pr.g.n();
  r.nnz = pr.g.nnz();
  r.p = pr.d.size;
  r.px = pr.d.dims[0];
  r.py = pr.d.dims[1];
  r.pz = pr.d.dims[2];
  r.decomp = o.decomp;
  r.halo = o.halo;
  r.cg = o.cg;
  r.iters = s.st.iters;
  r.rel_res = s.st.rel_res;
  r.err_inf = s.err;
  r.t_total = s.st.t_total;
  r.t_comp = s.st.t_comp;
  r.t_halo = s.st.t_halo;
  r.t_reduce = s.st.t_reduce;
  r.warmup = o.fixed_mode() ? o.warmup : 0;
  r.tol = o.tol;
  r.tag = o.tag;
  r.dot = o.dot;
  return r;
}

int run(const Options& o) {
  Problem pr = setup(o, o.Nx(), o.Ny(), o.Nz());
  const Solve s = distributed_solve(o, pr, nullptr);
  if (pr.d.rank == 0) {
    const Result r = make_result(o, pr, s);
    if (!o.quiet) print_result(r);
    if (!o.csv.empty()) append_csv(o.csv, r);
  }
  free_decomp(pr.d);
  return 0;
}

// V2a: one application of A to a non-smooth vector, compared bit for bit with the serial
// stencil applied by rank 0 to the whole grid.
int test_apply(const Options& o) {
  Problem pr = setup(o, o.Nx(), o.Ny(), o.Nz());
  const Block& b = pr.d.block;
  double* v = alloc_vector(b);
  double* q = alloc_vector(b);
  fill_test_vector(pr.g, b, v);
  {
    MpiComm comm(pr.d, parse_halo(o.halo));
    matvec(pr.g, b, comm, v, q, nullptr);
  }
  const std::vector<double> got = gather_to_root(pr.g, pr.d, q);
  int pass = 1;
  if (pr.d.rank == 0) {
    const Block gb = serial_block(pr.g);
    double* vs = alloc_vector(gb);
    double* qs = alloc_vector(gb);
    fill_test_vector(pr.g, gb, vs);
    SerialComm sc;
    matvec(pr.g, gb, sc, vs, qs, nullptr);
    long long mismatches = 0;
    double maxdiff = 0.0;
    for (int i = 1; i <= gb.n[0]; ++i)
      for (int j = 1; j <= gb.n[1]; ++j)
        for (int k = gb.k0; k <= gb.k1; ++k) {
          const long id = gb.idx(i, j, k);
          if (got[id] != qs[id]) {
            ++mismatches;
            maxdiff = std::fmax(maxdiff, std::fabs(got[id] - qs[id]));
          }
        }
    pass = (mismatches == 0);
    std::printf("V2a apply %dD %dx%dx%d p=%d [%dx%dx%d] decomp=%s halo=%s: %lld of %lld values "
                "differ (max |diff| %.3e) -> %s\n",
                pr.g.dim, pr.g.N[0], pr.g.N[1], pr.g.N[2], pr.d.size, pr.d.dims[0],
                pr.d.dims[1], pr.d.dims[2], o.decomp.c_str(), o.halo.c_str(), mismatches,
                pr.g.n(), maxdiff, pass ? "PASS" : "FAIL");
    free_vector(vs);
    free_vector(qs);
  }
  MPI_Bcast(&pass, 1, MPI_INT, 0, pr.d.cart);
  free_vector(v);
  free_vector(q);
  free_decomp(pr.d);
  return pass ? 0 : 1;
}

// V2b: a full CG solve compared with rank 0's serial solve: same iteration count (+-1) and
// ||x_p - x_serial|| / ||x_serial|| below --check-tol (only the reduction order differs).
int test_solve(const Options& o_in) {
  Options o = o_in;
  if (o.fixed_mode()) {
    o.tol = 1e-10;
    o.iters = 100000;
  }
  Problem pr = setup(o, o.Nx(), o.Ny(), o.Nz());
  std::vector<double> xp;
  const Solve s = distributed_solve(o, pr, &xp);
  int pass = 1;
  if (pr.d.rank == 0) {
    const Block gb = serial_block(pr.g);
    double* rhs = alloc_vector(gb);
    double* xs = alloc_vector(gb);
    fill_rhs(pr.g, gb, rhs);
    SerialComm sc;
    const CGStats ss = cg_solve(pr.g, gb, sc, rhs, xs, cg_options(o));
    double num = 0.0, den = 0.0;
    for (int i = 1; i <= gb.n[0]; ++i)
      for (int j = 1; j <= gb.n[1]; ++j)
        for (int k = gb.k0; k <= gb.k1; ++k) {
          const long id = gb.idx(i, j, k);
          num += (xp[id] - xs[id]) * (xp[id] - xs[id]);
          den += xs[id] * xs[id];
        }
    const double rel = std::sqrt(num / den);
    // With reproducible dot products the parallel solve must match serial bit for bit.
    const bool repro = (o.dot == "repro");
    pass = repro ? (s.st.iters == ss.iters && num == 0.0)
                 : (std::abs(s.st.iters - ss.iters) <= 1) && rel <= o.check_tol;
    std::printf("V2b solve %dD %dx%dx%d p=%d [%dx%dx%d] threads=%d cg=%s halo=%s dot=%s: iters "
                "%d vs serial %d, ||x_p - x_s||/||x_s|| = %.3e%s, err_inf %.4e -> %s\n",
                pr.g.dim, pr.g.N[0], pr.g.N[1], pr.g.N[2], pr.d.size, pr.d.dims[0],
                pr.d.dims[1], pr.d.dims[2], num_threads(), o.cg.c_str(), o.halo.c_str(),
                o.dot.c_str(), s.st.iters, ss.iters, rel, num == 0.0 ? " (bit-identical)" : "",
                s.err, pass ? "PASS" : "FAIL");
    free_vector(rhs);
    free_vector(xs);
  }
  MPI_Bcast(&pass, 1, MPI_INT, 0, pr.d.cart);
  free_decomp(pr.d);
  return pass ? 0 : 1;
}

// V1 with the parallel build (also V3 for OpenMP): N + 1 = 16, 32, ..., order -> 2.
int test_mms(const Options& o_in) {
  Options o = o_in;
  if (o.fixed_mode() || o.tol > 1e-10) o.tol = 1e-10;
  o.iters = 100000;
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0) {
    std::printf("V1 (manufactured solution), MPI p=%d threads=%d, %dD, cg=%s, tol=%.0e\n", size,
                num_threads(), o.dim, o.cg.c_str(), o.tol);
    std::printf("%8s %12s %8s %12s %12s %8s\n", "N", "h", "iters", "rel_res", "err_inf",
                "order");
  }
  double prev_err = 0.0, prev_h = 0.0, order = 0.0;
  for (int np1 = 16; np1 - 1 <= o.max_n; np1 *= 2) {
    const int N = np1 - 1;
    Problem pr = setup(o, N, N, N);
    const Solve s = distributed_solve(o, pr, nullptr);
    order = (prev_err > 0) ? std::log(prev_err / s.err) / std::log(prev_h / pr.g.h[0]) : 0.0;
    if (rank == 0)
      std::printf("%8d %12.4e %8d %12.3e %12.4e %8.3f\n", N, pr.g.h[0], s.st.iters,
                  s.st.rel_res, s.err, order);
    prev_err = s.err;
    prev_h = pr.g.h[0];
    free_decomp(pr.d);
  }
  const bool pass = std::fabs(order - 2.0) < 0.1;
  if (rank == 0)
    std::printf("V1 %dD (MPI p=%d threads=%d): observed order %.3f -> %s\n", o.dim, size,
                num_threads(), order, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (provided < MPI_THREAD_FUNNELED && rank == 0)
    std::fprintf(stderr, "warning: MPI provides thread level %d < FUNNELED\n", provided);
  const Options o = parse_options(argc, argv, "poisson_mpi", rank == 0);
  int rc = 0;
  if (o.test == "apply")
    rc = test_apply(o);
  else if (o.test == "solve")
    rc = test_solve(o);
  else if (o.test == "mms")
    rc = test_mms(o);
  else
    rc = run(o);
  MPI_Finalize();
  return rc;
}
