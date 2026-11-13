// Serial driver: the speedup baseline (no MPI, no OpenMP threads) and validation test V1.
//   poisson_serial --dim 3 --n 128 --iters 100            fixed-iteration timing run
//   poisson_serial --dim 2 --n 1023 --tol 1e-8            time to solution
//   poisson_serial --dim 3 --test mms --max-n 255         V1: error ~ h^2
#include <cmath>
#include <cstdio>
#include <vector>

#include "cg.hpp"
#include "grid.hpp"
#include "kernels.hpp"
#include "options.hpp"

namespace {

CGOptions cg_options(const Options& o) {
  CGOptions c;
  c.fused = (o.cg == "fused");
  c.repro = (o.dot == "repro");
  c.tol = o.tol;
  c.iters = o.iters;
  c.warmup = o.warmup;
  return c;
}

struct Solve {
  CGStats st;
  double err = 0.0;
};

Solve solve(const Grid& g, const CGOptions& co) {
  const Block b = serial_block(g);
  double* rhs = alloc_vector(b);
  double* x = alloc_vector(b);
  fill_rhs(g, b, rhs);
  SerialComm comm;
  Solve s;
  s.st = cg_solve(g, b, comm, rhs, x, co);
  s.err = max_error(g, b, x);
  free_vector(rhs);
  free_vector(x);
  return s;
}

// V1: N + 1 = 16, 32, ... so that h halves exactly; the observed order should approach 2.
int test_mms(const Options& o) {
  CGOptions co = cg_options(o);
  if (o.fixed_mode() || o.tol > 1e-10) co.tol = 1e-10;
  co.iters = 100000;
  std::printf("V1 (manufactured solution), serial, %dD, cg=%s, tol=%.0e\n", o.dim,
              o.cg.c_str(), co.tol);
  std::printf("%8s %12s %8s %12s %12s %8s\n", "N", "h", "iters", "rel_res", "err_inf", "order");
  double prev_err = 0.0, prev_h = 0.0, order = 0.0;
  for (int np1 = 16; np1 - 1 <= o.max_n; np1 *= 2) {
    const int N = np1 - 1;
    const Grid g = make_grid(o.dim, N, N, N);
    const Solve s = solve(g, co);
    order = (prev_err > 0) ? std::log(prev_err / s.err) / std::log(prev_h / g.h[0]) : 0.0;
    std::printf("%8d %12.4e %8d %12.3e %12.4e %8.3f\n", N, g.h[0], s.st.iters, s.st.rel_res,
                s.err, order);
    prev_err = s.err;
    prev_h = g.h[0];
  }
  const bool pass = std::fabs(order - 2.0) < 0.1;
  std::printf("V1 %dD: observed order %.3f -> %s\n", o.dim, order, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const Options o = parse_options(argc, argv, "poisson_serial", true);
  if (o.test == "mms") return test_mms(o);
  if (!o.test.empty()) {
    std::fprintf(stderr, "poisson_serial: --test %s needs the MPI build\n", o.test.c_str());
    return 2;
  }
  const Grid g = make_grid(o.dim, o.Nx(), o.Ny(), o.Nz());
  const Solve s = solve(g, cg_options(o));

  Result r;
  r.machine = machine_name(o);
  r.backend = "serial";
  r.dim = g.dim;
  r.nx = g.N[0];
  r.ny = g.N[1];
  r.nz = g.N[2];
  r.n = g.n();
  r.nnz = g.nnz();
  r.decomp = "none";
  r.halo = "none";
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
  if (!o.quiet) print_result(r);
  if (!o.csv.empty()) append_csv(o.csv, r);
  return 0;
}
