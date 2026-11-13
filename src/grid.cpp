#include "grid.hpp"

#include <vector>

Grid make_grid(int dim, int nx, int ny, int nz) {
  Grid g;
  g.dim = dim;
  g.N[0] = nx;
  g.N[1] = ny;
  g.N[2] = (dim == 3) ? nz : 1;
  g.diag = 0.0;
  for (int d = 0; d < 3; ++d) {
    g.h[d] = 1.0 / (g.N[d] + 1);
    g.c[d] = (d < dim) ? 1.0 / (g.h[d] * g.h[d]) : 0.0;
    g.diag += 2.0 * g.c[d];
  }
  return g;
}

// Every interior point has a diagonal entry, plus one entry per interior neighbour:
// nnz = n + 2 * sum_d (N_d - 1) * prod_{e != d} N_e   (= 5N^2-4N in 2D, 7N^3-6N^2 in 3D).
long long Grid::nnz() const {
  long long total = n();
  for (int d = 0; d < dim; ++d) total += 2LL * (N[d] - 1) * (n() / N[d]);
  return total;
}

Block make_block(const Grid& g, const int n[3], const int off[3]) {
  Block b;
  b.dim = g.dim;
  for (int d = 0; d < 3; ++d) {
    b.n[d] = (d < g.dim) ? n[d] : 1;
    b.off[d] = (d < g.dim) ? off[d] : 0;
  }
  b.sx = b.n[0] + 2;
  b.sy = b.n[1] + 2;
  b.sz = (g.dim == 3) ? b.n[2] + 2 : 1;
  b.sj = b.sz;
  b.si = (long)b.sy * b.sz;
  b.k0 = (g.dim == 3) ? 1 : 0;
  b.k1 = (g.dim == 3) ? b.n[2] : 0;
  b.size = (std::size_t)b.sx * b.sy * b.sz;
  return b;
}

Block serial_block(const Grid& g) {
  const int off[3] = {0, 0, 0};
  return make_block(g, g.N, off);
}

namespace {

// g(x) and q(x) at the block's interior points along dimension d (index 1..n[d]).
struct Profile {
  std::vector<double> g, q;
};

Profile profile(const Grid& gr, const Block& b, int d) {
  Profile p;
  p.g.assign(b.n[d] + 2, 0.0);
  p.q.assign(b.n[d] + 2, 0.0);
  for (int i = 1; i <= b.n[d]; ++i) {
    const double x = (b.off[d] + i) * gr.h[d];
    p.g[i] = mms_g(x);
    p.q[i] = mms_q(x);
  }
  return p;
}

}  // namespace

void fill_rhs(const Grid& gr, const Block& b, double* f) {
  const Profile X = profile(gr, b, 0), Y = profile(gr, b, 1);
  if (b.dim == 2) {
#pragma omp parallel for schedule(static)
    for (int i = 1; i <= b.n[0]; ++i)
      for (int j = 1; j <= b.n[1]; ++j)
        f[b.idx(i, j, 0)] = X.q[i] * Y.g[j] + X.g[i] * Y.q[j];
  } else {
    const Profile Z = profile(gr, b, 2);
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= b.n[0]; ++i)
      for (int j = 1; j <= b.n[1]; ++j)
        for (int k = 1; k <= b.n[2]; ++k)
          f[b.idx(i, j, k)] = X.q[i] * Y.g[j] * Z.g[k] + X.g[i] * Y.q[j] * Z.g[k] +
                              X.g[i] * Y.g[j] * Z.q[k];
  }
}

double max_error(const Grid& gr, const Block& b, const double* x) {
  const Profile X = profile(gr, b, 0), Y = profile(gr, b, 1);
  double e = 0.0;
  if (b.dim == 2) {
#pragma omp parallel for reduction(max : e) schedule(static)
    for (int i = 1; i <= b.n[0]; ++i)
      for (int j = 1; j <= b.n[1]; ++j)
        e = std::fmax(e, std::fabs(x[b.idx(i, j, 0)] - X.g[i] * Y.g[j]));
  } else {
    const Profile Z = profile(gr, b, 2);
#pragma omp parallel for collapse(2) reduction(max : e) schedule(static)
    for (int i = 1; i <= b.n[0]; ++i)
      for (int j = 1; j <= b.n[1]; ++j)
        for (int k = 1; k <= b.n[2]; ++k)
          e = std::fmax(e, std::fabs(x[b.idx(i, j, k)] - X.g[i] * Y.g[j] * Z.g[k]));
  }
  return e;
}

void fill_test_vector(const Grid& gr, const Block& b, double* v) {
  (void)gr;
  for (int i = 1; i <= b.n[0]; ++i)
    for (int j = 1; j <= b.n[1]; ++j)
      for (int k = b.k0; k <= b.k1; ++k) {
        const long long gi = b.off[0] + i, gj = b.off[1] + j, gk = b.off[2] + k;
        const long long hsh = (gi * 7919 + gj * 104729 + gk * 1299709 + gi * gj * 31) % 2001;
        v[b.idx(i, j, k)] = (double)(hsh - 1000) / 997.0;  // integer arithmetic, then one rounding
      }
}
